# Usage

Everything is driven by a single command-line executable, `simulate.py`. It runs
either a Taylor–Green vortex (the classic transition-to-turbulence benchmark) or
forced isotropic turbulence, and streams the velocity field to a NetCDF file
through µGrid. It runs serially or under MPI:

```bash
python simulate.py [options]                # serial
mpirun -np 4 python simulate.py [options]   # parallel
python simulate.py --device cuda [options]  # on the GPU
python simulate.py --help                   # all options
```

## Options

| option | meaning | default |
|---|---|---|
| `-i, --initial-condition` | `taylor-green` or `turbulence` | taylor-green |
| `-n, --nb-grid-pts NX NY NZ` | grid resolution | 32 32 32 |
| `-v, --viscosity` | kinematic viscosity \(\nu\) | 1/1600 |
| `-t, --timestep` | time step | 1e-3 |
| `-N, --nb-steps` | number of steps | 100000 |
| `-a, --amplitude` | velocity amplitude of the initial condition | 1 |
| `--no-dealias` | disable 2/3-rule dealiasing | (on) |
| `--device` | `cpu` (default) or `cuda`/`cuda:N` | cpu |
| `--seed` | RNG seed for the `turbulence` initial condition | none |
| `-o, --output` | NetCDF output file | navier_stokes.nc |
| `--dump-interval` | write the velocity every N steps | 100 |
| `--screen-interval` | print diagnostics every N steps | 100 |

## Initial conditions

- **`taylor-green`** reproduces the classic Taylor–Green vortex. Use a thin grid
  (e.g. `-n 32 32 4`) for the 2D variant and a viscosity measurement. The field
  is built from coordinates, so it is identical across MPI ranks and a parallel
  run reproduces the serial result bit-for-bit.
- **`turbulence`** seeds a random incompressible field with a \(k^{-5/3}\)
  (Kolmogorov) spectrum and forces the flow by **freezing the lowest-wavenumber
  amplitudes** after every step. The random field is generated per rank, so a
  parallel run does *not* reproduce the serial field bit-for-bit.

## Diagnostics

Every `--screen-interval` steps the driver prints the step, time, the
min/mean/max of the velocity, the total spectral power (energy), and a
frames/second rate. A hierarchical, MPI-aware timer prints a breakdown of where
wall time went (`rk4_step`, diagnostics, output) at the end of the run.

## Output

The driver writes **only** the real-space velocity field (variable `velocity`)
to the NetCDF file via µGrid's parallel writer:

```python
file = FileIONetCDF(output, OpenMode.Overwrite, communicator=comm)
file.register_field_collection(ns.fft.real_space_collection,
                               field_names=["velocity"])
...
file.append_frame().write()
```

The `field_names=["velocity"]` filter keeps the right-hand side's scratch fields,
which share the same field collection, out of the file.

## Post-processing (`scripts/`)

All scripts read the `velocity` variable written by `simulate.py`.

- `slice.py` — extract a 2D slice from the 3D output into a smaller file.
- `plot.py` — animate a slice to an `.mp4`.
- `spectrum.py` — energy/dissipation spectra.
- `eval_taylor_green.py` — fit the viscous decay rate of a Taylor–Green run.

## Testing

```bash
pytest            # run the suite
pytest --cov      # with a coverage report
```

The suite checks the solver against analytic references (the spectral curl, a
divergence-free right-hand side, exact spectral energy on odd *and* even grids,
the analytic Taylor–Green decay rate, the 2/3-rule against a zero-padded
reference, energy conservation at \(\nu=0\), and fourth-order RK4 convergence). It
runs in CI on every push, including a 2-rank MPI consistency check.
