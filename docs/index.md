# muNavierStokes

muNavierStokes is a direct numerical simulation (DNS) of the incompressible
Navier–Stokes equations using a **pseudo-spectral** method on a triply-periodic
box. Fast Fourier transforms, field storage and the (optionally MPI-parallel)
domain decomposition are provided by
[µGrid](https://github.com/muSpectre/muGrid), which also supplies the GPU
backend.

## Highlights

- **Pseudo-spectral solver.** The velocity field is advanced in its Fourier
  representation; the nonlinear term is formed in real space and the pressure is
  eliminated by a divergence-free (Leray) projection.
- **Rotational form + RK4.** The advection term uses the rotational form
  \(u\times\omega\), integrated in time with a classical fourth-order
  Runge–Kutta step that allocates nothing per stage.
- **2/3-rule dealiasing.** Aliasing error from the nonlinear product is removed
  by band-limiting to the lower two thirds of the spectrum.
- **MPI-parallel and GPU-ready.** The same code runs serially, across MPI ranks
  (pencil-decomposed FFTs), or on the GPU — selected at run time.
- **One executable.** [`simulate.py`](usage.md) runs a Taylor–Green vortex or
  forced isotropic turbulence and streams the velocity field to NetCDF.

## A first run

```bash
# Taylor-Green vortex on a 64^3 grid (serial)
python simulate.py --initial-condition taylor-green -n 64 64 64

# forced isotropic turbulence on 4 MPI ranks
mpirun -np 4 python simulate.py --initial-condition turbulence

# on the GPU
python simulate.py --device cuda -i taylor-green
```

## Where to go next

- [Installation](installation.md) — dependencies and how to build the
  MPI/GPU-enabled µGrid this solver needs.
- [Usage](usage.md) — the `simulate.py` driver, its options, and the
  post-processing scripts.
- [Physics](physics.md) — governing equations, the Fourier representation, and
  dealiasing.
- [Implementation](implementation.md) — the `NavierStokes` solver, the RK4
  stepper, and the µGrid field interface.
- [GPU](gpu.md) — how the solver runs device-agnostically on CPU or GPU.
- [Benchmark](benchmark.md) — single-CPU, multi-CPU (MPI) and single-GPU timing
  on a real machine.

## History

muNavierStokes is part of the µSpectre project, an open-source platform for
FFT-based continuum mesoscale modelling. It is built on the standalone
[µGrid](https://github.com/muSpectre/muGrid) grid library.

## License

muNavierStokes is free software, distributed under the terms of the GNU Lesser
General Public License (see `LICENSE`).
