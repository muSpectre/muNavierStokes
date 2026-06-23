# muNavierStokes

[![Tests](https://github.com/muSpectre/muNavierStokes/actions/workflows/tests.yml/badge.svg)](https://github.com/muSpectre/muNavierStokes/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/muSpectre/muNavierStokes/branch/main/graph/badge.svg)](https://codecov.io/gh/muSpectre/muNavierStokes)

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

Full documentation is published at
**<https://muSpectre.github.io/muNavierStokes/>** (built with MkDocs from the
[`docs/`](docs/) directory): the [physics](docs/physics.md), the
[implementation](docs/implementation.md) (including the µFFT→µGrid migration
notes), [usage](docs/usage.md), [GPU](docs/gpu.md) execution, and a
[benchmark](docs/benchmark.md) (single-CPU, multi-CPU MPI, single-GPU).

## Layout

| path                        | contents                                               |
|-----------------------------|--------------------------------------------------------|
| `simulate.py`               | the simulation driver (single executable)              |
| `muNavierStokes/`           | the solver (`NavierStokes`) and the RK4 integrator     |
| `scripts/`                  | post-processing (slicing, plotting, spectra, decay fit)|
| `tests/`                    | `pytest` correctness and functional tests              |

## Testing

```bash
pytest            # run the test suite
pytest --cov      # with a coverage report
```

Tests run in CI on every push (see `.github/workflows/tests.yml`).

## Dependencies

`muGrid` (with FFT + NetCDF support), `mpi4py`, `numpy`; plus
`matplotlib` and `netCDF4` for the post-processing scripts. Install the test
extras with `pip install -e ".[test]"`.
