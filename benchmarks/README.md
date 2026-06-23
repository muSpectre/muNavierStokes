# Benchmark database

`results.csv` is a small, git-friendly, **append-only** database of benchmark
measurements. Every benchmark run adds a fresh batch of rows stamped with the
date, code version, and machine, so the file is a growing, diffable history that
is meant to be committed and updated regularly.

Page generation is fully separated from data collection: `benchmark.py` reads
rows back from this file and renders the documentation table and plots, so the
[Benchmark](../docs/benchmark.md) page can be regenerated at any time — and any
historical run reproduced — without re-measuring. This mirrors µGrid's benchmark
setup; the CSV schema is shared (see `benchmark_db.py`).

## Files

| file | role |
|---|---|
| `results.csv` | the append-only measurement database |
| `benchmark_db.py` | schema, provenance capture, CSV I/O, device/MPI config vocabulary |
| `bench_workload.py` | the timed workload (one RK4 step loop); run per data point, prints JSON |
| `benchmark.py` | the driver: runs the workload across sizes/configs, appends rows, renders the page |

## Format

One row per measured data point ("long" format), so new studies and series just
add rows (and leave unrelated columns blank) rather than reshaping the file.
Columns:

| group | columns | meaning |
|---|---|---|
| provenance | `timestamp`, `version`, `commit`, `dirty`, `cpu`, `gpu` | when/what/where — identical across all rows of one run |
| identity | `benchmark`, `study`, `label` | which plot/series the point belongs to |
| parameters | `device`, `nranks`, `dim`, `n`, `npts`, `steps` | enough to reproduce the run |
| results | `secs` (per RK4 step), `mpoints` (Mpoint/s), `gbps` | the measurement |

- `version` is `git describe --tags --always --dirty`; `commit` is the short
  hash; `dirty=1` flags an uncommitted working tree (avoid for runs you intend to
  keep).
- A **run** is one invocation of `benchmark.py`: all its rows share one
  `timestamp`. Rendering selects the most recent run by default
  (`--timestamp <date-or-prefix>` picks an older one).

Current `benchmark`/`study` values:

| benchmark | study | series (`label`) |
|---|---|---|
| `navier_stokes` | `time_vs_size` | config key: `cpu1`, `cpuN`, `gpu1`, `gpuN` |
| `navier_stokes` | `mpi_scaling` | rank count |

## Workflow

The non-MPI configurations (`cpu1`, `gpu1`) use whatever muGrid is on the default
`PYTHONPATH` (e.g. the pip-installed, GPU-enabled, non-MPI build). The MPI
configurations need an **MPI-enabled** muGrid, prepended to `PYTHONPATH` for the
`mpiexec` subprocesses only via `--mpi-pythonpath` (default points at a sibling
`../muGrid` MPI build tree).

```bash
# Measure (appends a new dated run) and regenerate the page + plots:
python benchmarks/benchmark.py --doc-out docs/benchmark.md

# Re-render the page from the latest run already in the DB (no measuring):
python benchmarks/benchmark.py --render-only --doc-out docs/benchmark.md

# Point at a specific MPI-enabled muGrid build for the MPI runs:
python benchmarks/benchmark.py --doc-out docs/benchmark.md \
    --mpi-pythonpath /path/to/muGrid/build-mpi/language_bindings/python:/path/to/muGrid/language_bindings/python

# Then commit the updated database and pages:
git add benchmarks/results.csv docs/benchmark.md docs/benchmark.png docs/benchmark_mpi.png
```
