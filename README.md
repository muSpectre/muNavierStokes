# muNavierStokes

Direct numerical simulation (DNS) of the incompressible Navier–Stokes equations
with a pseudo-spectral method. Fast Fourier transforms and the (optionally
MPI-parallel) field storage are provided by
[µGrid](https://github.com/muSpectre/muGrid).

## Quick start

A single executable, `simulate.py`, runs the calculation and writes the
velocity field to a NetCDF file. It runs serially or under MPI:

```bash
# Taylor-Green vortex on a 64^3 grid (serial)
python simulate.py --initial-condition taylor-green -n 64 64 64

# forced isotropic turbulence on 4 MPI ranks
mpirun -np 4 python simulate.py --initial-condition turbulence

# see all options
python simulate.py --help
```

## Documentation

See [`docs/muNavierStokes.md`](docs/muNavierStokes.md) for the physics, the
implementation, the µFFT→µGrid migration notes, and how to run the simulation
and tests.

## Layout

| path                        | contents                                               |
|-----------------------------|--------------------------------------------------------|
| `simulate.py`               | the simulation driver (single executable)              |
| `muNavierStokes/`           | the solver (`NavierStokes`) and the RK4 integrator     |
| `scripts/`                  | post-processing (slicing, plotting, spectra, decay fit)|
| `tests/test_correctness.py` | analytic correctness checks for the solver             |

## Dependencies

`muGrid` (with FFT + NetCDF support), `NuMPI`, `mpi4py`, `numpy`; plus
`matplotlib` and `netCDF4` for the post-processing scripts.
