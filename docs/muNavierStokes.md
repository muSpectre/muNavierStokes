# muNavierStokes

A direct numerical simulation (DNS) of the incompressible Navier–Stokes
equations using a **pseudo-spectral** method. Fast Fourier transforms, field
storage and (optionally MPI-parallel) domain decomposition are provided by
[µGrid](https://github.com/muSpectre/muGrid).

This document describes the physics, the implementation, and the µGrid API the
code relies on. It also records the migration from µFFT to µGrid.

---

## 1. Physics

### 1.1 Governing equations

For an incompressible fluid of constant density with kinematic viscosity `ν`,

```
∂u/∂t + (u·∇)u = −∇p + ν ∇²u ,      ∇·u = 0 .
```

The code integrates the **rotational form** of the advection term. Using the
identity `(u·∇)u = (∇×u)×u + ∇(½|u|²)` and folding the gradient term into a
modified pressure `P = p + ½|u|²`,

```
∂u/∂t = u×ω − ∇P + ν ∇²u ,      ω = ∇×u  (vorticity).
```

### 1.2 Fourier representation

The velocity field is expanded in Fourier modes on a triply-periodic box. With
`û(k)` the Fourier coefficient of `u` and wavevector `k`, derivatives become
algebraic:

* gradient → multiplication by `i k`
* Laplacian → multiplication by `−|k|²`
* vorticity → `ω̂ = i k × û`

Incompressibility `∇·u = 0` becomes `k·û = 0`: the velocity is transverse to
its wavevector. The pressure is eliminated by **projecting** the right-hand
side onto the divergence-free subspace (Leray/Helmholtz projection),

```
P_⊥ f = f − k (k·f)/|k|² .
```

Applying the projection to the momentum equation removes `∇P` and yields the
evolution equation actually integrated:

```
∂û/∂t = P_⊥( (u×ω)^ ) − ν |k|² û .
```

The nonlinear term `(u×ω)^` is computed **pseudo-spectrally**: transform `û`
and `ω̂` to real space, form the cross product point-wise, and transform back.

### 1.3 Dealiasing (the 2/3 rule)

The point-wise product of two band-limited fields produces wavenumbers up to
twice the maximum resolved wavenumber. On a finite grid these fold back
("alias") onto the resolved modes and corrupt the solution. The **2/3 rule**
removes this error: zero all modes with `|kᵢ| ≥ (2/3) k_max` *before* forming
the product, so the aliased content lands only in the (discarded) upper third
of the spectrum.

`NavierStokes.dudt` applies the dealiasing mask in two places:

1. **On the inputs** `û` and `ω̂`, before the inverse transforms. This is what
   keeps the resolved modes free of aliasing error. It is essential whenever
   the field carries energy in the cut band — e.g. the random `turbulence`
   initial condition.
2. **On the nonlinear output** `(u×ω)^`, after the forward transform. This
   prevents spurious aliased energy from being injected into the cut-band
   modes, which then merely decay viscously and stay decoupled from the
   resolved dynamics.

The cut-off wavenumber is `k_c = 2π/(3 Δx)`, i.e. `(2/3) k_max` with
`k_max = π/Δx` the Nyquist wavenumber.

---

## 2. Implementation

### 2.1 `muNavierStokes.NavierStokes`

```python
NavierStokes(nb_grid_pts, physical_size=(1,1,1), viscosity=0.001,
             dealias=True, communicator=None)
```

| argument        | meaning                                                              |
|-----------------|---------------------------------------------------------------------|
| `nb_grid_pts`   | grid resolution, e.g. `(64, 64, 64)`                                |
| `physical_size` | box size in each direction (sets the grid spacing `Δx`)             |
| `viscosity`     | kinematic viscosity `ν`                                             |
| `dealias`       | enable the 2/3-rule dealiasing                                      |
| `communicator`  | an `mpi4py` communicator for parallel runs, or `None` for serial    |

Precomputed in `_init_fft`:

* `_wavevector_cqks` — the angular wavevectors `k = 2π · fftfreq / Δx`, shape
  `(3, *fourier_subdomain)`.
* `_wavevector_sq_qks` — `|k|²` (exposed read-only as the `wavevector_sq`
  property).
* `_inv_wavevector_cqks` — `k/|k|²` (with the `k=0` mode regularised), used for
  the projection.
* `_dealias_qks` — boolean mask of the retained (lower 2/3) modes.

Public accessors `fft` (the `FFTEngine`), `wavevector_sq` (`|k|²`), and `parnp`
(the MPI-aware reduction helper) let drivers build initial conditions and
diagnostics without touching private attributes.

Key methods:

* **`dudt(t, uarr_cqks)`** — the right-hand side of `∂û/∂t`. Reuses six named
  µGrid fields (allocated once, reused every call) for the transforms.
* **`power(u_cqks, mask=None)`** — total (or masked) spectral energy via
  Parseval's theorem, with the factor-of-2 bookkeeping of the half-complex
  (r2c) representation. Both self-conjugate planes — `kx = 0` and (for even Nx)
  `kx = Nyquist` — are counted once, so the result is exact on odd *and* even
  grids. Reductions are MPI- and GPU-aware through µGrid's `Communicator`.
* **`to_incompressible(u_cqks)`** — apply the projection `P_⊥` to make an
  arbitrary field divergence-free (used to build initial conditions).

### 2.2 Time integration — `muNavierStokes.rk4`

A classical fourth-order Runge–Kutta step. `rk4(f, t, y, dt)` returns the
*increment* `Δy` such that `y_{n+1} = y_n + Δy`.

### 2.3 The driver — `simulate.py`

A single command-line executable owns everything that used to be duplicated
across the milestone scripts: parameter handling, the initial condition
(`taylor-green` or `turbulence`), the time loop, screen diagnostics, and the
NetCDF velocity output. The initial-condition helpers (`taylor_green`,
`turbulence`) use only the public `NavierStokes` interface. See §4.2 for usage.

### 2.4 Verification

The `tests/` directory holds a `pytest` suite. `test_navier_stokes.py` checks
the solver against analytic references:

* the spectral curl operator (vanishing in-plane components of a rigid rotation),
* `to_incompressible` and `dudt` are divergence-free (`k·u = 0`),
* `power()` equals the real-space energy on **odd and even** grids (the latter
  exercises the Nyquist-plane correction), and the mask partitions the total,
* the 2D Taylor–Green vortex decays at the analytic rate `2ν(2π)²`,
* dealiasing zeroes the cut-band **and** reproduces a zero-padded, alias-free
  reference on the retained modes to machine precision (the definitive 2/3-rule
  check),
* energy is conserved at `ν = 0`, incompressibility is preserved during
  integration, and `rk4` exhibits fourth-order convergence.

`test_simulate.py` covers the driver: the initial-condition helpers (the
Taylor–Green field is verified divergence-free) and an end-to-end run whose
NetCDF output is checked to contain only the `velocity` field. See §4.4.

---

## 3. Migration from µFFT to µGrid

The solver previously used `muFFT.FFT`. It now uses `muGrid.FFTEngine`. The
table summarises the API mapping:

| µFFT                                   | µGrid                                            |
|----------------------------------------|--------------------------------------------------|
| `from muFFT import FFT`                 | `from muGrid import FFTEngine`                   |
| `FFT(n, engine=..., communicator=c)`    | `FFTEngine(n, communicator=c)` — no `engine` arg |
| `fft.real_space_field(name, 3)`         | same (int or tuple `(3,)` for components)        |
| `fft.fourier_space_field(name, 3)`      | same                                             |
| `fft.fft(real, fourier)` / `ifft(...)`  | same (field-based, in-place)                     |
| `fft.fftfreq`, `fft.coords`             | same                                             |
| `fft.normalisation`                     | same                                             |
| `fft.real_field_collection`             | **`fft.real_space_collection`**                  |
| `fft.nb_fourier_grid_pts` (local)       | `nb_fourier_subdomain_grid_pts` (local) / `nb_fourier_grid_pts` (global) |
| `field.p = array`                       | **`field.p[...] = array`** (no `.p` setter)      |

The three behavioural points that required code changes:

1. **No `engine` selection.** µGrid picks the backend automatically (PocketFFT
   on CPU; cuFFT/rocFFT on GPU via a `device=` argument), so the `engine`
   argument was dropped from `NavierStokes` entirely.
2. **`field.p` has no setter.** The numpy view returned by `.p` is writable in
   place, so every `field.p = value` became `field.p[...] = value`.
3. **Local vs. global Fourier grid.** In µGrid, `nb_fourier_grid_pts` is the
   **global** shape and `nb_fourier_subdomain_grid_pts` is the **local** (per
   rank) shape. Arrays that live alongside the distributed Fourier field
   (e.g. the explicit `uarr_cqks` for the turbulence initial condition) must be
   sized with the *subdomain* shape, or parallel runs mismatch the wavevector
   arrays.

The functional one-shot transform `fft.fft(array) -> array` of older µFFT no
longer exists; the curl check (now in `tests/`) and `scripts/spectrum.py` use
the field-based form (copy into a field, transform, read `.p`).

---

## 4. Running

### 4.1 Dependencies

* `muGrid` (with FFT support; **≥ the version that ships `FFTEngine`**, and
  built with NetCDF for the file I/O)
* `mpi4py`
* `numpy`
* `matplotlib`, `netCDF4` (post-processing scripts only)

### 4.2 Running the simulation

Everything is driven by the single executable `simulate.py`:

```bash
python simulate.py [options]              # serial
mpirun -np 4 python simulate.py [options] # parallel
```

| option                         | meaning                                          | default      |
|--------------------------------|--------------------------------------------------|--------------|
| `-i, --initial-condition`      | `taylor-green` or `turbulence`                   | taylor-green |
| `-n, --nb-grid-pts NX NY NZ`   | grid resolution                                  | 32 32 32     |
| `-v, --viscosity`              | kinematic viscosity `ν`                          | 1/1600       |
| `-t, --timestep`               | time step                                        | 1e-3         |
| `-N, --nb-steps`               | number of steps                                  | 100000       |
| `-a, --amplitude`              | velocity amplitude of the initial condition      | 1            |
| `--no-dealias`                 | disable 2/3-rule dealiasing                       | (on)         |
| `--seed`                       | RNG seed for the `turbulence` initial condition  | none         |
| `-o, --output`                 | NetCDF output file                               | navier_stokes.nc |
| `--dump-interval`              | write the velocity every N steps                 | 100          |
| `--screen-interval`            | print diagnostics every N steps                  | 100          |

* `taylor-green` reproduces the classic Taylor–Green vortex (use a thin grid,
  e.g. `-n 32 32 4`, for the 2D variant and a viscosity measurement).
* `turbulence` seeds a random incompressible field with a `k^(-5/3)` spectrum
  and forces the flow by freezing the lowest-wavenumber amplitudes after every
  step.

The driver writes **only** the real-space velocity field (variable
`velocity`) to the NetCDF file via µGrid's parallel writer:

```python
file = FileIONetCDF(output, OpenMode.Overwrite, communicator=comm)
file.register_field_collection(ns.fft.real_space_collection, field_names=['velocity'])
...
file.append_frame().write()
```

The `field_names=['velocity']` filter keeps `dudt`'s scratch fields, which
share the same field collection, out of the file.

### 4.3 Post-processing (`scripts/`)

All scripts read the `velocity` variable written by `simulate.py`.

* `slice.py` — extract a 2D slice from the 3D output into a smaller file.
* `plot.py` — animate a slice to an `.mp4`.
* `spectrum.py` — energy/dissipation spectra.
* `eval_taylor_green.py` — fit the viscous decay rate of a Taylor–Green run.

### 4.4 Running the test suite

With muGrid importable, run `pytest` from the repository root:

```bash
pytest                      # run the suite
pytest --cov                # with a coverage report
```

Configuration lives in `pyproject.toml` (`pytest` adds the repository root to
the path; coverage measures `muNavierStokes` and `simulate`). The suite runs in
CI via GitHub Actions (`.github/workflows/tests.yml`) on every push: a matrix of
Python versions runs `pytest` with coverage (uploaded to Codecov), and a
separate job verifies that a 2-rank MPI run reproduces the serial result.
`tests/test_gpu.py` adds device tests that skip automatically without a GPU.

### 4.5 GPU execution

µGrid can place fields in GPU memory (field buffers become CuPy arrays instead
of numpy). Pass a device to the solver or the driver:

```python
ns = NavierStokes(nb_grid_pts, viscosity=nu, device="cuda")   # or "cuda:N"
```
```bash
python simulate.py --device cuda -i taylor-green
```

The solver is written to be device-agnostic:

* `NavierStokes` discovers the array module (numpy or CuPy) of its fields and
  moves every precomputed coefficient array (wavevectors, projection operator,
  dealiasing/Nyquist masks) onto that device, so `dudt` never mixes host and
  device memory.
* It uses array *methods* (`.sum()`, `.conj()`, `.real`) and the array module
  (`xp.cross`) rather than the numpy free functions, which would force host
  execution or host/device mixing.
* The RK4 integrator is pure array arithmetic and already device-agnostic.
* The driver builds its initial conditions on the device (CuPy RNG and trig)
  and collapses scalar diagnostics with `float(...)`.

The caller must keep the integration array `uarr_cqks` on the same device as
the solver (the driver does; `taylor_green`/`turbulence` return device arrays).

---

## 5. Notes & possible improvements

* **Forcing.** The `turbulence` initial condition forces the flow by re-imposing
  fixed low-wavenumber amplitudes after every step. This is a simple, robust
  scheme; a constant-energy-injection forcing would be a natural extension.
* **MPI reproducibility.** The seeded-random `turbulence` initial condition is
  generated per rank, so a parallel run does not reproduce the serial field
  bit-for-bit (the Taylor–Green one, built from coordinates, does). A
  domain-decomposition-independent seeding would restore that.
* **GPU reductions.** Reductions go through µGrid's GPU-aware `Communicator`
  via its `.reduction` adapter (`sum`/`min`/`max`/`mean`): it reduces
  CuPy buffers on-device and short-circuits serial runs entirely (no MPI call,
  so no CUDA-aware MPI needed for a single rank). Multi-GPU runs still require a
  CUDA-aware `mpi4py` for the cross-rank `Allreduce`. (This replaced the earlier
  `NuMPI.Tools.Reduction`, which was numpy-oriented and invoked MPI even for a
  single rank.) Remaining checks for multi-GPU: confirm `FileIONetCDF` staging
  of device fields. The array math itself (everything in
  `dudt`/`power`/`to_incompressible`) is device-ready.
