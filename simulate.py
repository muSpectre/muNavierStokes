#!/usr/bin/env python3
"""
Pseudo-spectral direct numerical simulation (DNS) of the incompressible
Navier-Stokes equations.

This single entry point runs either a Taylor-Green vortex (the classic
transition-to-turbulence benchmark) or forced isotropic turbulence, and streams
the velocity field to a NetCDF file through µGrid. It runs serially or under MPI:

    python simulate.py --initial-condition taylor-green -n 64 64 64
    mpirun -np 4 python simulate.py --initial-condition turbulence
"""

import argparse
import time

import numpy as np
from muGrid import FileIONetCDF, OpenMode
from muTimer import Timer

try:
    # mpi4py supplies COMM_WORLD for parallel runs. muGrid accepts an mpi4py
    # communicator (even a size-1 one on a non-MPI build); reductions go through
    # muGrid's own Communicator inside the solver.
    from mpi4py import MPI
except ImportError:  # serial fall-back when mpi4py is not installed
    MPI = None

from muNavierStokes import NavierStokes

# Box size; all wavevectors are derived from this and the grid resolution.
PHYSICAL_SIZE = (1, 1, 1)


def taylor_green(ns, amplitude):
    """Taylor-Green vortex. Returns the Fourier-space velocity field.

    The z-amplitude is -2 so that the field is divergence-free: for the
    cos/sin/sin, sin/cos/sin, sin/sin/cos structure with equal wavenumbers the
    component amplitudes must sum to zero (1 + 1 - 2 = 0).
    """
    xp = ns.array_module
    # coords is a host array; move it to the compute device
    x, y, z = (xp.asarray(c) for c in ns.fft.coords)
    u_cxyz = ns.fft.real_space_field("velocity", 3)
    u_cxyz.p[...] = amplitude * xp.array(
        [
            xp.cos(2 * np.pi * x) * xp.sin(2 * np.pi * y) * xp.sin(2 * np.pi * z),
            xp.sin(2 * np.pi * x) * xp.cos(2 * np.pi * y) * xp.sin(2 * np.pi * z),
            -2.0
            * xp.sin(2 * np.pi * x)
            * xp.sin(2 * np.pi * y)
            * xp.cos(2 * np.pi * z),
        ]
    )
    u_cqks = ns.fft.fourier_space_field("velocity_k", 3)
    ns.fft.fft(u_cxyz, u_cqks)
    return u_cqks.p * ns.fft.normalisation


def turbulence(ns, amplitude, seed):
    """Random incompressible field with a k^(-5/3) (Kolmogorov) spectrum.

    Returns ``(uarr_cqks, freeze_mask, frozen_amplitudes)``, where the mask and
    amplitudes are used to force the lowest wavenumbers (excluding the mean
    flow) by re-imposing them after every time step.
    """
    xp = ns.array_module
    shape = (3,) + ns.fft.nb_fourier_subdomain_grid_pts
    # xp.random.default_rng works for both numpy and cupy
    rng = xp.random.default_rng(seed)
    uarr_cqks = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # Energy ~ k^(-5/3) corresponds to an amplitude ~ k^(-5/6)
    k_sq = ns.wavevector_sq
    nonzero = k_sq > 0
    factor = xp.zeros_like(k_sq)
    factor[nonzero] = amplitude * k_sq[nonzero] ** (-5 / 6)
    uarr_cqks *= factor
    uarr_cqks = ns.to_incompressible(uarr_cqks)

    freeze_wavevector = 2 * np.pi * 3 / np.mean(PHYSICAL_SIZE)
    freeze_mask = (k_sq < freeze_wavevector**2) & nonzero
    return uarr_cqks, freeze_mask, uarr_cqks[:, freeze_mask].copy()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--initial-condition",
        "-i",
        choices=["taylor-green", "turbulence"],
        default="taylor-green",
        help="initial condition (default: taylor-green)",
    )
    p.add_argument(
        "--nb-grid-pts",
        "-n",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=[32, 32, 32],
        help="grid resolution (default: 32 32 32)",
    )
    p.add_argument(
        "--viscosity",
        "-v",
        type=float,
        default=1 / 1600,
        help="kinematic viscosity (default: 1/1600)",
    )
    p.add_argument(
        "--timestep", "-t", type=float, default=1e-3, help="time step (default: 1e-3)"
    )
    p.add_argument(
        "--nb-steps",
        "-N",
        type=int,
        default=100000,
        help="number of steps (default: 100000)",
    )
    p.add_argument(
        "--amplitude",
        "-a",
        type=float,
        default=1.0,
        help="velocity amplitude (default: 1)",
    )
    p.add_argument(
        "--no-dealias", action="store_true", help="disable 2/3-rule dealiasing"
    )
    p.add_argument(
        "--device",
        default=None,
        help="compute device: 'cpu' (default) or 'cuda'/'cuda:N' for the GPU",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the turbulence initial condition",
    )
    p.add_argument(
        "--output", "-o", default="navier_stokes.nc", help="NetCDF output file"
    )
    p.add_argument(
        "--dump-interval", type=int, default=100, help="write velocity every N steps"
    )
    p.add_argument(
        "--screen-interval",
        type=int,
        default=100,
        help="report to screen every N steps",
    )
    return p.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = comm.Get_rank() if comm is not None else 0

    ns = NavierStokes(
        tuple(args.nb_grid_pts),
        PHYSICAL_SIZE,
        args.viscosity,
        dealias=not args.no_dealias,
        communicator=comm,
        device=args.device,
    )

    # Initial condition (turbulence additionally provides the forcing mask)
    forcing = None
    if args.initial_condition == "taylor-green":
        uarr_cqks = taylor_green(ns, args.amplitude)
    else:
        uarr_cqks, freeze_mask, frozen = turbulence(ns, args.amplitude, args.seed)
        forcing = (freeze_mask, frozen)

    # The integration state is a Fourier-space field, advanced in place by the
    # field-based RK4 stepper (no per-step allocation).
    state = ns.fft.fourier_space_field("state", 3)
    state.p[...] = uarr_cqks

    # Real-space velocity field for output (the inverse transform of `state`).
    velocity = ns.fft.real_space_field("velocity", 3)

    # Open the output file and register *only* the velocity field for writing
    # (the collection also holds dudt's scratch fields, which we skip).
    file = FileIONetCDF(args.output, OpenMode.Overwrite, communicator=comm)
    file.register_field_collection(
        ns.fft.real_space_collection, field_names=["velocity"]
    )

    if rank == 0:
        print(
            f"# {ns.fft.backend_name} FFT engine, grid {tuple(args.nb_grid_pts)}, "
            f"nu = {args.viscosity:g}, dealias = {not args.no_dealias}, "
            f"IC = {args.initial_condition}",
            flush=True,
        )
        print(
            "#    step        time        min /      mean /       max"
            "          power     frames/s",
            flush=True,
        )

    # Hierarchical timer for a breakdown of where wall time goes. It is
    # MPI-aware (gathers to rank 0) and prints nothing until print_summary().
    timer = Timer(comm=comm)

    last_time = None
    with timer("time_loop"):
        for n in range(args.nb_steps):
            if n % args.screen_interval == 0:
                with timer("diagnostics"):
                    ns.fft.ifft(state, velocity)
                    # float() collapses 0-d numpy/cupy results to a host scalar
                    # so the formatting below works on both CPU and GPU.
                    umin = float(ns.parnp.min(velocity.p))
                    umean = float(ns.parnp.mean(velocity.p))
                    umax = float(ns.parnp.max(velocity.p))
                    power = float(ns.power(state.p))
                    if rank == 0:
                        fps = (
                            ""
                            if last_time is None
                            else f"{args.screen_interval / (time.time() - last_time):11.4g}"
                        )
                        print(
                            f"{n:9d} {n * args.timestep:11.4g}   {umin:9.3g} / "
                            f"{umean:9.3g} / {umax:9.3g}   {power:12.5g} {fps}",
                            flush=True,
                        )
                    last_time = time.time()

            # Integrate one step (in place, on the device)
            with timer("rk4_step"):
                ns.rk4_step(state, n * args.timestep, args.timestep)

            # Forcing: re-impose the frozen low-wavenumber amplitudes
            if forcing is not None:
                with timer("forcing"):
                    freeze_mask, frozen = forcing
                    state.p[:, freeze_mask] = frozen

            # Output to file
            if n % args.dump_interval == 0:
                with timer("output"):
                    ns.fft.ifft(state, velocity)
                    file.append_frame().write()

    file.close()

    timer.print_summary(title="Navier-Stokes timing breakdown", comm=comm)


if __name__ == "__main__":
    main()
