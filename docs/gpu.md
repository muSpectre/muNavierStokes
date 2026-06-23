# GPU

µGrid can place fields in GPU memory (field buffers become [CuPy](https://cupy.dev/)
arrays instead of numpy, and FFTs run on cuFFT/rocFFT). muNavierStokes is written
to be **device-agnostic**: the same solver code runs on the CPU or the GPU,
selected at run time.

## Building muGrid with GPU support

Compile µGrid from source with CUDA (NVIDIA) or HIP (AMD) enabled, e.g.:

```bash
CMAKE_ARGS="-DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75" \
    pip install -v "muGrid @ git+https://github.com/muSpectre/muGrid.git"
```

Set `CMAKE_CUDA_ARCHITECTURES` to your device's compute capability (e.g.
`70`=V100, `80`=A100, `90`=H100). You also need `cupy` matching your CUDA
toolkit (e.g. `pip install cupy-cuda12x`). Verify with:

```bash
python -c "import muGrid; print('gpu', muGrid.has_gpu, muGrid.is_gpu_available())"
```

## Running on the GPU

Pass a device to the solver or the driver:

```python
ns = NavierStokes(nb_grid_pts, viscosity=nu, device="cuda")   # or "cuda:N"
```

```bash
python simulate.py --device cuda -i taylor-green
```

## How device-agnosticism works

- **`NavierStokes` discovers the array module** (numpy or CuPy) of its fields
  (`self._xp`) and moves every precomputed coefficient array — wavevectors, the
  projection operator, the dealiasing and Nyquist masks — onto that device, so
  `dudt` never mixes host and device memory.
- It uses array *methods* (`.sum()`, `.conj()`, `.real`) and the array module
  (`xp.cross`) rather than the numpy free functions, which would force host
  execution or host/device mixing.
- The **RK4 integrator** is pure (fused) field arithmetic and is already
  device-agnostic.
- The **driver** builds its initial conditions on the device (CuPy RNG and trig)
  and collapses scalar diagnostics with `float(...)` so the formatting works on
  both CPU and GPU.

The caller must keep the integration array on the **same device** as the solver
(the driver does; `taylor_green`/`turbulence` return device arrays).

## Reductions

Global reductions (`min`/`max`/`mean`/`sum` behind `ns.parnp` and `power`) go
through µGrid's GPU-aware `Communicator` via its `.reduction` adapter: it reduces
CuPy buffers on-device and short-circuits serial runs entirely (no MPI call, so
no CUDA-aware MPI is needed for a single rank). Multi-GPU runs still require a
CUDA-aware `mpi4py` for the cross-rank all-reduce.

## Multi-GPU

The solver binds to whatever communicator it is given, so on a host with several
GPUs `mpirun -np <#GPUs> python simulate.py -d cuda` runs one rank per device.
See the [Benchmark](benchmark.md) page for the single-GPU timing on the reference
machine and the automatically-added multi-GPU curve on multi-GPU hosts.
