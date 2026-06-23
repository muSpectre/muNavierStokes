#!/usr/bin/env python3
"""Scaling benchmark for the muNavierStokes pseudo-spectral solver.

Times one classical RK4 step of the incompressible Navier-Stokes solver
(`bench_workload.py`) across log-spaced 3D grid sizes, and (re)generates the
documentation [Benchmark](../docs/benchmark.md) page. It has two parts:

1. **Time vs. grid size** — one merged plot comparing, at each grid size, a
   *single CPU core*, the *full machine via MPI* (one rank per logical core), and
   a *single GPU*. On a multi-GPU host a *multi-GPU MPI* curve is added
   automatically (one rank per GPU).
2. **MPI strong scaling** — speedup vs. rank count at a fixed problem size.

Data collection and page generation are separate. A run executes
`bench_workload.py` as a subprocess for each data point (under `mpiexec` for the
MPI configurations) and **appends** the results — with date, code version, and
machine — to the shared benchmark database (`benchmarks/results.csv`, see
`benchmark_db.py`). Tables and plots are then rendered *from the database*, so the
page can be regenerated at any time, and historical runs stay reproducible.

muGrid build selection
----------------------
The non-MPI configurations (`cpu1`, `gpu1`) use whatever muGrid is on the default
`PYTHONPATH` (e.g. the pip-installed, GPU-enabled, non-MPI build). The MPI
configurations need an **MPI-enabled** muGrid, which is prepended to `PYTHONPATH`
for the `mpiexec` subprocesses only — pass `--mpi-pythonpath` (default points at a
sibling `../muGrid` MPI build tree).

Examples
--------
    # run benchmarks, append to the DB, and (re)generate the page:
    python benchmarks/benchmark.py --doc-out docs/benchmark.md

    # just re-render the page from the latest run already in the DB:
    python benchmarks/benchmark.py --render-only --doc-out docs/benchmark.md
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_db as db  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WORKLOAD = os.path.join(HERE, "bench_workload.py")
BENCHMARK = "navier_stokes"
CONFIG_META = db.CONFIG_META

# Default MPI-enabled muGrid build (compiled extension + source wrappers) for a
# sibling ../muGrid checkout. Override with --mpi-pythonpath.
DEFAULT_MPI_PYTHONPATH = os.pathsep.join([
    os.path.join(REPO_ROOT, "..", "muGrid", "build-mpi", "language_bindings",
                 "python"),
    os.path.join(REPO_ROOT, "..", "muGrid", "language_bindings", "python"),
])


# --------------------------------------------------------------------------- #
# Running the workload
# --------------------------------------------------------------------------- #
def run(device, n, steps, warmup, nranks, mpi_pythonpath):
    """Run one timed workload; return a dict of metrics or None."""
    base = [WORKLOAD, "-n", str(n), "-d", device,
            "--steps", str(steps), "--warmup", str(warmup), "--json"]
    # Make muNavierStokes importable; the non-MPI configs inherit the ambient
    # muGrid, the MPI configs prepend an MPI-enabled build.
    env = dict(os.environ)
    pp = [REPO_ROOT]
    if nranks > 1:
        pp = [mpi_pythonpath, REPO_ROOT]
        cmd = ["mpiexec", "-n", str(nranks), sys.executable] + base
    else:
        cmd = [sys.executable] + base
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(pp + ([existing] if existing else []))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                             env=env)
    except subprocess.SubprocessError:
        return None
    m = re.search(r"\{.*\}", out.stdout, re.DOTALL)
    if not m:
        sys.stderr.write(f"  [{device} n={n} ranks={nranks}] no JSON\n"
                         f"{out.stderr[-500:]}\n")
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["npts"], secs=r["secs_per_step"],
                mpoints=r["mpoints_per_sec"])


def collect(args, prov):
    """Run all data points and return DB rows (does not write)."""
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    rows = []

    # Time vs. size, one curve per device/MPI config.
    for n in args.sizes:
        for key, device, nranks in configs:
            r = run(device, n, args.steps, args.warmup, nranks,
                    args.mpi_pythonpath)
            label = CONFIG_META[key]["label"](nranks)
            if r is None:
                sys.stderr.write(f"  {label} {n}^3: skipped (failed / OOM)\n")
                continue
            rows.append({**prov, "benchmark": BENCHMARK, "study": "time_vs_size",
                         "label": key, "device": device, "nranks": nranks,
                         "dim": 3, "n": n, "npts": r["npts"],
                         "steps": args.steps, "secs": r["secs"],
                         "mpoints": r["mpoints"]})
            sys.stderr.write(f"  {label} {n}^3 ({r['npts']} pts): "
                             f"{r['secs'] * 1e3:.2f} ms/step, "
                             f"{r['mpoints']:.1f} Mpoint/s\n")

    # MPI strong scaling (CPU).
    for n in args.scaling_sizes:
        for R in args.scaling_ranks:
            r = run("cpu", n, args.steps, args.warmup, R, args.mpi_pythonpath)
            if r is None:
                sys.stderr.write(f"  scaling {n}^3 ranks={R}: skipped\n")
                continue
            rows.append({**prov, "benchmark": BENCHMARK, "study": "mpi_scaling",
                         "label": str(R), "device": "cpu", "nranks": R,
                         "dim": 3, "n": n, "npts": r["npts"],
                         "steps": args.steps, "secs": r["secs"],
                         "mpoints": r["mpoints"]})
            sys.stderr.write(f"  scaling {n}^3 ranks={R}: "
                             f"{r['secs'] * 1e3:.2f} ms/step\n")
    return rows


# --------------------------------------------------------------------------- #
# Re-shaping DB rows for rendering
# --------------------------------------------------------------------------- #
def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(int(npts))


def merged_from_rows(rows):
    """{config_key: {n: row}}."""
    d = {}
    for r in rows:
        if r["study"] != "time_vs_size":
            continue
        d.setdefault(r["label"], {})[r["n"]] = r
    return d


def scaling_from_rows(rows):
    """{n: {ranks: row}}."""
    d = {}
    for r in rows:
        if r["study"] != "mpi_scaling":
            continue
        d.setdefault(r["n"], {})[int(r["nranks"])] = r
    return d


def sizes_in(rows, study):
    return sorted({r["n"] for r in rows if r["study"] == study})


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def table_markdown(sizes, configs, merged):
    cols = [n for n in sizes if any(n in merged.get(k, {}) for k, *_ in configs)]
    header = ("| Configuration | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for key, label, _style in configs:
        res = merged.get(key)
        if not res:
            continue
        cells = " | ".join(
            (f"{res[n]['secs'] * 1e3:.3g}" if n in res else "—") for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def scaling_tables_markdown(scaling, ranks):
    out = []
    for n in sorted(scaling):
        rows = scaling[n]
        t1 = rows.get(1, {}).get("secs")
        out.append(f"**{n}³ ({n ** 3:,} points)**\n")
        out.append("| Ranks | ms/step | Speedup | Parallel eff. |")
        out.append("|---|---|---|---|")
        for R in ranks:
            if R not in rows:
                continue
            t = rows[R]["secs"]
            sp = t1 / t if t1 else float("nan")
            out.append(f"| {R} | {t * 1e3:.2f} | {sp:.2f}× | {sp / R * 100:.0f}% |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_merged_plot(configs, merged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, style in configs:
        pts = sorted((r["npts"], r["secs"]) for r in merged.get(key, {}).values())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, label=label, **style)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Time per RK4 step (s)")
    ax.set_title("Navier-Stokes (pseudo-spectral): time per step vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_scaling_plot(scaling, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    all_ranks = sorted({R for rows in scaling.values() for R in rows})
    rmax = max(all_ranks) if all_ranks else 1
    ax.plot([1, rmax], [1, rmax], ls="--", color="0.6", label="ideal (linear)")
    markers = {32: "v", 64: "o", 96: "s", 128: "^"}
    for n in sorted(scaling):
        rows = scaling[n]
        t1 = rows.get(1, {}).get("secs")
        if not t1:
            continue
        xs = sorted(rows)
        ys = [t1 / rows[R]["secs"] for R in xs]
        ax.plot(xs, ys, marker=markers.get(n, "o"), label=f"{n}³")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    if all_ranks:
        ax.set_xticks(all_ranks)
        ax.set_xticklabels([str(R) for R in all_ranks])
        ax.set_yticks(all_ranks)
        ax.set_yticklabels([str(R) for R in all_ranks])
    ax.set_xlabel("MPI ranks (CPU cores)")
    ax.set_ylabel("Speedup vs. 1 rank")
    ax.set_title("Navier-Stokes (pseudo-spectral): MPI strong scaling")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Doc page
# --------------------------------------------------------------------------- #
DOC_TEMPLATE = """# Benchmark

Wall time of a single classical fourth-order Runge-Kutta step of the
pseudo-spectral [Navier-Stokes solver](implementation.md) (`benchmarks/bench_workload.py`,
Taylor-Green initial condition), across log-spaced **3D** grid sizes. Lower is
better.

!!! info "Test machine & code version"
    - **CPU:** {cpu}
    - **GPU:** {gpu}
    - **muNavierStokes:** `{version}` — run {timestamp}

Run configuration: triply-periodic box, kinematic viscosity `ν = 1/1600`,
time step `1e-3`, 2/3-rule dealiasing on. Each data point times {steps} RK4 steps
(after warm-up) and reports the mean — i.e. a **fixed work budget**, so every
configuration performs identical arithmetic. Timing covers only the integration
loop (no file I/O, no diagnostics). One RK4 step evaluates the right-hand side
four times; each evaluation runs several 3-component forward/inverse FFTs plus the
fused per-pixel curl, dealiasing, viscous, and Leray-projection kernels.

## Time vs. grid size

The plot below merges the ways of running the *same* solver on this machine:

- **CPU (1 core)** — a single core, MPI disabled (the non-MPI muGrid build).
  muGrid's compute kernels carry no OpenMP, so a non-MPI CPU run uses one core.
- **CPU ({ncores} cores, MPI)** — the whole CPU via MPI pencil decomposition
  (`mpiexec -n {ncores}`), the grid split into per-rank subdomains whose FFTs
  exchange data each transform.
- **GPU (1 device)** — the whole GPU (cuFFT plus the fused device kernels).
{gpu_mpi_bullet}
{table}

(values are **milliseconds per RK4 step**)

![Navier-Stokes time per step vs. number of grid points]({plot_name})

The workload is dominated by the multidimensional FFTs, which are
memory-bandwidth-bound, so the time tracks memory throughput rather than peak
FLOPs. Two regimes are visible. At **small grids** the GPU is overhead-bound: a
fixed per-step cost of kernel launches and host/device synchronisation (a few
milliseconds) dominates, so the GPU is actually the *slowest* configuration while
the single CPU core, with almost no fixed overhead, wins on the tiniest grids.
Past the crossover (here around \(10^5\) grid points) the picture flips: the
GPU's high memory bandwidth takes over and it becomes the fastest by a growing
margin, while the full CPU via MPI sits in between — well ahead of one core but
short of the GPU.

!!! note "Multi-GPU"
    The solver binds each MPI rank to the communicator it is given, so on a host
    with several GPUs `mpiexec -n <#GPUs> python bench_workload.py -d cuda` runs
    one rank per device. This benchmark adds a *GPU (N devices, MPI)* curve
    automatically when more than one GPU is present. **{gpu_count_note}**

## MPI strong scaling (CPU)

Strong scaling of the same step (fixed problem size, increasing MPI ranks) on the
{ncores}-core CPU.

{scaling_tables}
![Navier-Stokes MPI strong scaling]({scaling_plot_name})

Scaling is strong at low rank counts and then tapers: the pseudo-spectral step is
memory-bandwidth-bound and the distributed FFT needs an all-to-all transpose each
transform, so once per-rank subdomains get small the transpose communication and
the CG-style reductions begin to dominate. Larger grids keep scaling further
because they keep more work per rank.

All data points live in the shared benchmark database `benchmarks/results.csv`
(date, code version, machine, parameters, results). This page is generated by
`benchmarks/benchmark.py`; re-render it from the database (no recompute) with
`--render-only`, or run a fresh measurement that appends a new dated row set:

```bash
python benchmarks/benchmark.py --doc-out docs/benchmark.md
```
"""


def write_doc_page(path, plot_path, scaling_plot_path, table, scaling_tables,
                   meta, ncores, multi_gpu):
    if multi_gpu:
        gpu_mpi_bullet = ("- **GPU (N devices, MPI)** — all GPUs, one rank per "
                          "device.\n")
        gpu_count_note = "This run used several GPUs, so the multi-GPU curve is shown."
    else:
        gpu_mpi_bullet = ""
        gpu_count_note = ("This run used a single GPU, so only the single-GPU "
                          "curve is shown; the script produces the multi-GPU "
                          "curve on a multi-GPU host with no changes.")
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=meta["cpu"], gpu=meta["gpu"], version=meta["version"],
            timestamp=meta["timestamp"], table=table, steps=meta["steps"],
            ncores=ncores, gpu_mpi_bullet=gpu_mpi_bullet,
            gpu_count_note=gpu_count_note,
            plot_name=os.path.basename(plot_path),
            scaling_tables=scaling_tables,
            scaling_plot_name=os.path.basename(scaling_plot_path)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count(),
                    help="Ranks for the full-machine MPI CPU curve")
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--scaling-sizes", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--scaling-ranks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=10, help="timed steps per point")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up steps")
    ap.add_argument("--mpi-pythonpath", default=DEFAULT_MPI_PYTHONPATH,
                    help="PYTHONPATH prepended for MPI (mpiexec) subprocesses; "
                         "must point at an MPI-enabled muGrid build")
    ap.add_argument("--render-only", action="store_true",
                    help="Skip running; render from the database")
    ap.add_argument("--timestamp", default=None,
                    help="Render this run (timestamp prefix / date) instead of "
                         "the latest")
    ap.add_argument("--db", default=db.DB_PATH, help="Benchmark CSV path")
    ap.add_argument("--doc-out", default=None)
    ap.add_argument("--plot-out",
                    default=os.path.join(REPO_ROOT, "docs", "benchmark.png"))
    ap.add_argument("--scaling-plot-out",
                    default=os.path.join(REPO_ROOT, "docs", "benchmark_mpi.png"))
    args = ap.parse_args()

    if not args.render_only:
        prov = db.run_provenance()
        rows = collect(args, prov)
        if not rows:
            sys.exit("No successful runs — nothing to record.")
        db.append_rows(rows, args.db)
        sys.stderr.write(f"appended {len(rows)} rows to {args.db}\n")
        select_ts = prov["timestamp"]
    else:
        select_ts = args.timestamp

    rows = db.select(db.load(args.db), BENCHMARK, select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    meta = {k: rows[0][k] for k in ("cpu", "gpu", "version", "timestamp")}
    meta["steps"] = next((r["steps"] for r in rows
                          if r["study"] == "time_vs_size"), args.steps)
    configs = db.render_configs(rows, "time_vs_size")
    merged = merged_from_rows(rows)
    scaling = scaling_from_rows(rows)
    ncores = next((r["nranks"] for r in rows if r["label"] == "cpuN"),
                  args.mpi_cpu_ranks)
    multi_gpu = any(r["label"] == "gpuN" for r in rows)

    sizes = sizes_in(rows, "time_vs_size")
    table = table_markdown(sizes, configs, merged)
    scaling_ranks = sorted({int(r["nranks"]) for r in rows
                            if r["study"] == "mpi_scaling"})
    scaling_tables = scaling_tables_markdown(scaling, scaling_ranks)
    print("\n" + table + "\n\n" + scaling_tables)

    plot_out = os.path.abspath(args.plot_out)
    scaling_plot_out = os.path.abspath(args.scaling_plot_out)
    make_merged_plot(configs, merged, plot_out)
    make_scaling_plot(scaling, scaling_plot_out)
    sys.stderr.write(f"wrote {plot_out}\nwrote {scaling_plot_out}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, scaling_plot_out, table,
                       scaling_tables, meta, ncores, multi_gpu)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
