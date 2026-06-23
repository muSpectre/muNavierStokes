#!/usr/bin/env python3
"""Timed RK4-stepping workload for the muNavierStokes benchmark driver.

Builds a Taylor-Green initial condition on an ``n x n x n`` grid and advances it
with the pseudo-spectral solver's in-place :meth:`NavierStokes.rk4_step` for a
fixed number of steps, reporting the mean wall time **per step** as a single
JSON object on stdout. It runs serially, on the GPU (``--device cuda``), or under
MPI (one rank per process; only rank 0 prints).

This is the per-data-point executable that ``benchmark.py`` launches as a
subprocess (under ``mpiexec`` for the MPI configurations), mirroring how µGrid's
benchmark driver runs ``poisson.py``/``homogenization.py``. Timing covers only
the integration loop (no I/O, no diagnostics): a fixed number of RK4 steps is the
solver's fixed work budget, so every configuration does identical arithmetic.

Example
-------
    python bench_workload.py -n 64 -d cpu --warmup 2 --steps 10 --json
    mpiexec -n 8 python bench_workload.py -n 64 -d cpu --steps 10 --json
"""

import argparse
import json
import sys
import time

import numpy as np

from muNavierStokes import NavierStokes

try:
    from mpi4py import MPI
except ImportError:  # serial fall-back when mpi4py is not installed
    MPI = None

PHYSICAL_SIZE = (1, 1, 1)


def taylor_green(ns, amplitude=1.0):
    """Divergence-free Taylor-Green vortex as a Fourier-space velocity array."""
    xp = ns.array_module
    x, y, z = (xp.asarray(c) for c in ns.fft.coords)
    u_cxyz = ns.fft.real_space_field("velocity", 3)
    u_cxyz.p[...] = amplitude * xp.array([
        xp.cos(2 * np.pi * x) * xp.sin(2 * np.pi * y) * xp.sin(2 * np.pi * z),
        xp.sin(2 * np.pi * x) * xp.cos(2 * np.pi * y) * xp.sin(2 * np.pi * z),
        -2.0 * xp.sin(2 * np.pi * x) * xp.sin(2 * np.pi * y) * xp.cos(2 * np.pi * z),
    ])
    u_cqks = ns.fft.fourier_space_field("velocity_k", 3)
    ns.fft.fft(u_cxyz, u_cqks)
    return u_cqks.p * ns.fft.normalisation


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--nb-grid-pts", type=int, default=64,
                    help="per-axis grid size (the grid is n x n x n)")
    ap.add_argument("-d", "--device", default="cpu",
                    help="'cpu' (default) or 'cuda'/'cuda:N' for the GPU")
    ap.add_argument("-v", "--viscosity", type=float, default=1 / 1600)
    ap.add_argument("-t", "--timestep", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed steps to run before timing (JIT/plan warmup)")
    ap.add_argument("--steps", type=int, default=10, help="timed steps")
    ap.add_argument("--json", action="store_true", help="emit a JSON result line")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = comm.Get_rank() if comm is not None else 0
    nranks = comm.Get_size() if comm is not None else 1
    device = None if args.device == "cpu" else args.device

    n = args.nb_grid_pts
    ns = NavierStokes((n, n, n), PHYSICAL_SIZE, args.viscosity,
                      communicator=comm, device=device)
    xp = ns.array_module

    state = ns.fft.fourier_space_field("state", 3)
    state.p[...] = taylor_green(ns)

    def sync():
        # Block until queued GPU work has actually completed before stopping the
        # clock (a no-op on the CPU, where xp is numpy).
        if xp is not np:
            xp.cuda.runtime.deviceSynchronize()

    dt = args.timestep
    for _ in range(args.warmup):
        ns.rk4_step(state, 0.0, dt)
    sync()
    if comm is not None:
        comm.Barrier()

    t0 = time.perf_counter()
    for i in range(args.steps):
        ns.rk4_step(state, i * dt, dt)
    sync()
    elapsed = time.perf_counter() - t0

    # The slowest rank governs the wall time of a synchronous step.
    if comm is not None:
        elapsed = comm.allreduce(elapsed, op=MPI.MAX)
    secs = elapsed / args.steps

    npts = int(np.prod(ns.fft.nb_domain_grid_pts))
    result = {
        "config": {
            "device": args.device, "nranks": nranks,
            "n": n, "npts": npts, "dim": 3,
            "warmup": args.warmup, "steps": args.steps,
        },
        "results": {
            "secs_per_step": secs,
            "steps_per_sec": 1.0 / secs if secs else float("nan"),
            "mpoints_per_sec": npts / secs / 1e6 if secs else float("nan"),
        },
    }
    if rank == 0:
        if args.json:
            print(json.dumps(result))
        else:
            r = result["results"]
            print(f"n={n}^3 ({npts} pts) device={args.device} "
                  f"ranks={nranks}: {r['secs_per_step'] * 1e3:.3f} ms/step, "
                  f"{r['mpoints_per_sec']:.1f} Mpoint/s", file=sys.stderr)


if __name__ == "__main__":
    main()
