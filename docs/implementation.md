# Implementation

## `muNavierStokes.NavierStokes`

```python
NavierStokes(nb_grid_pts, physical_size=(1, 1, 1), viscosity=0.001,
             dealias=True, communicator=None, device=None)
```

| argument | meaning |
|---|---|
| `nb_grid_pts` | grid resolution, e.g. `(64, 64, 64)` |
| `physical_size` | box size in each direction (sets the grid spacing \(\Delta x\)) |
| `viscosity` | kinematic viscosity \(\nu\) |
| `dealias` | enable the 2/3-rule dealiasing |
| `communicator` | an `mpi4py` communicator for parallel runs, or `None` for serial |
| `device` | `None`/`"cpu"`, or `"cuda"`/`"cuda:N"` for the GPU |

Precomputed once in `_init_fft` (and moved onto the compute device):

- `_wavevector_cqks` — the angular wavevectors \(k = 2\pi\,\text{fftfreq}/\Delta x\),
  shape `(3, *fourier_subdomain)`.
- `_wavevector_sq_qks` — \(|k|^2\) (exposed read-only as the `wavevector_sq`
  property).
- `_inv_wavevector_cqks` — \(k/|k|^2\) (with the \(k=0\) mode regularised), used
  for the projection.
- `_dealias_qks` — boolean mask of the retained (lower 2/3) modes.
- constant Fourier symbols `i k` (curl) and \(\nu|k|^2\) (viscous), so they are
  not rebuilt every step.

Public accessors `fft` (the `FFTEngine`), `wavevector_sq` (\(|k|^2\)),
`array_module` (numpy or cupy), and `parnp` (the MPI-aware reduction helper) let
drivers build initial conditions and diagnostics without touching private
attributes.

### Two right-hand-side paths

The solver exposes a **performance path** and a **convenience path**:

- **`dudt_into(t, y, out)`** + **`rk4_step(y, t, dt)`** — the hot path. They
  operate on µGrid **fields** (not plain arrays), reuse scratch fields allocated
  once in `_init_fft`, and use µGrid's fused, in-place BLAS-like field operations
  (`copy`/`scal`/`axpy`/`cross`/`leray_project`). A full RK4 step therefore
  allocates **nothing** per stage. The cross products and the per-pixel pressure
  projection are the only remaining element-wise temporaries inside `dudt_into`.
- **`dudt(t, uarr_cqks)`** — a thin array-in/array-out wrapper around
  `dudt_into`, for tests and diagnostics. The in-loop integrator uses
  `rk4_step` instead.

Other methods:

- **`power(u_cqks, mask=None)`** — total (or masked) spectral energy via
  Parseval's theorem, with the factor-of-2 bookkeeping of the half-complex (r2c)
  representation. Both self-conjugate planes — \(k_x = 0\) and (for even \(N_x\))
  \(k_x = \text{Nyquist}\) — are counted once, so the result is exact on odd *and*
  even grids. Reductions are MPI- and GPU-aware through µGrid's `Communicator`.
- **`to_incompressible(u_cqks)`** — apply the projection \(P_\perp\) to make an
  arbitrary field divergence-free (used to build initial conditions).

## Time integration

`rk4_step` is a classical fourth-order Runge–Kutta step performed **in place** on
a Fourier-space field, with the stage combinations expressed as fused
`copy`/`axpy` field operations:

\[
y_{n+1} = y_n + \tfrac{\Delta t}{6}\,(k_1 + 2k_2 + 2k_3 + k_4).
\]

A standalone array-based `rk4(f, t, y, dt)` (returning the increment
\(\Delta y\)) is also provided for the convenience/test path.

## The µGrid field interface

A few µGrid conventions shape the code:

- **Fields, not arrays.** Fields come from the engine with
  `real_space_field(name, ncomp)` and `fourier_space_field(name, ncomp)`.
  Repeated calls with the same name return the *same* field (reusing memory),
  which is how the solver pre-allocates scratch space.
- **`.p` has no setter.** A field's numpy/cupy view is `field.p`; it is writable
  in place, so always `field.p[...] = value`, never `field.p = value`.
- **Local vs. global Fourier grid.** `nb_fourier_grid_pts` is the **global**
  shape; `nb_fourier_subdomain_grid_pts` is the **local** (per-rank) shape.
  Arrays that live alongside a distributed Fourier field (e.g. the explicit
  `turbulence` initial condition) must be sized with the *subdomain* shape, or
  parallel runs mismatch the wavevector arrays.

### Array-name suffixes

Variable names encode array layout: `_cqks` is a complex Fourier-space field
(component, \(q_x, q_y, q_z\)), `_cxyz` is real-space components, and `_qks` is a
single-component Fourier-space symbol or mask.

## Migration from µFFT to µGrid

The solver previously used `muFFT.FFT`; it now uses `muGrid.FFTEngine`. The key
behavioural differences that required code changes:

1. **No backend `engine` argument.** µGrid picks the backend automatically
   (PocketFFT on CPU; cuFFT/rocFFT on GPU via `device=`), so the `engine`
   argument was dropped.
2. **`field.p` has no setter** — every `field.p = value` became
   `field.p[...] = value`.
3. **Local vs. global Fourier grid** — distributed arrays must use the subdomain
   shape (see above).
4. **No one-shot functional transform.** The old `fft.fft(array) -> array` no
   longer exists; the field-based form (copy into a field, transform, read `.p`)
   is used throughout.

| µFFT | µGrid |
|---|---|
| `from muFFT import FFT` | `from muGrid import FFTEngine` |
| `FFT(n, engine=..., communicator=c)` | `FFTEngine(n, communicator=c, device=...)` |
| `fft.real_field_collection` | `fft.real_space_collection` |
| `fft.nb_fourier_grid_pts` (local) | `nb_fourier_subdomain_grid_pts` (local) / `nb_fourier_grid_pts` (global) |
| `field.p = array` | `field.p[...] = array` |
