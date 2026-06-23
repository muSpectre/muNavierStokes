# Installation

muNavierStokes is a small pure-Python package. Its one non-trivial dependency is
[µGrid](https://github.com/muSpectre/muGrid) **with FFT support** — and, for
parallel or GPU runs, built with MPI/CUDA. Everything else (`numpy`, `mpi4py`)
is standard.

## Dependencies

| package | needed for |
|---|---|
| `muGrid` (with `FFTEngine`; NetCDF for file I/O) | the solver and output |
| `numpy` | array math |
| `mpi4py` | MPI-parallel runs |
| `matplotlib`, `netCDF4` | the post-processing scripts |

## Installing muGrid

µGrid with the `FFTEngine` may not yet be on PyPI as a release wheel, so it is
usually **built from source**. A minimal CPU build:

```bash
pip install -v "muGrid @ git+https://github.com/muSpectre/muGrid.git"
```

µGrid autodetects MPI and NetCDF from the system; install their development
packages first to enable parallel runs and file I/O, e.g. on Debian/Ubuntu:

```bash
sudo apt-get install -y cmake ninja-build \
    openmpi-bin libopenmpi-dev libnetcdf-dev libpnetcdf-dev
CC=mpicc pip install --no-binary mpi4py mpi4py
pip install -v "muGrid @ git+https://github.com/muSpectre/muGrid.git"
```

For a **GPU** build, compile µGrid with CUDA enabled (see [GPU](gpu.md)).

## Installing muNavierStokes

!!! warning "Use `--no-deps` against a source-built µGrid"
    `pyproject.toml` pins `muGrid>=0.109.0`. A git/source build of µGrid carries
    a setuptools-scm *dev* version that does **not** satisfy that pin, so a normal
    `pip install -e .` would try to pull a release wheel from PyPI and clobber
    your MPI/NetCDF-enabled build. Install without re-resolving dependencies:

```bash
# numpy / mpi4py / muGrid already present from the steps above
pip install --no-deps -e .
pip install pytest pytest-cov netCDF4   # optional: tests + post-processing
```

## Verifying the install

```bash
python -c "import muGrid; print('mpi', muGrid.has_mpi, 'gpu', muGrid.has_gpu, 'netcdf', muGrid.has_netcdf)"
python -c "import muNavierStokes; print(muNavierStokes.__version__)"
pytest            # run the test suite
```

The suite includes GPU tests (`tests/test_gpu.py`) that skip automatically when
no GPU/CuPy is present, so `pytest` is a no-op for those on CPU-only machines.
