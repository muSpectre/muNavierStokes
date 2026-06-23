"""
GPU device tests. These are skipped automatically when no GPU (or CuPy) is
available, so the suite is a no-op on CPU-only machines and CI. On a GPU host
they verify that the solver runs on the device and reproduces the CPU result.
"""
import numpy as np
import pytest

import muGrid

from muNavierStokes import NavierStokes, rk4


def _gpu_available():
    if not getattr(muGrid, "has_gpu", False):
        return False
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return muGrid.is_gpu_available()


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="no GPU / CuPy available")


def _random_incompressible(ns, xp, seed=0):
    shape = (3,) + ns.fft.nb_fourier_subdomain_grid_pts
    rng = xp.random.default_rng(seed)
    u = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return ns.to_incompressible(u)


def test_solver_runs_on_gpu():
    import cupy as cp

    ns = NavierStokes((16, 16, 16), (1, 1, 1), 1 / 1600, dealias=True, device="cuda")
    # Coefficient arrays and the array module live on the device
    assert ns.array_module is cp
    assert isinstance(ns.wavevector_sq, cp.ndarray)

    u = _random_incompressible(ns, cp)
    d = ns.dudt(0.0, u)
    assert isinstance(d, cp.ndarray)

    # dudt is divergence-free on the device
    div = (ns._wavevector_cqks * d).sum(axis=0)
    assert float(cp.abs(div).max()) < 1e-8

    # A few RK4 steps stay finite (and on the device)
    for _ in range(5):
        u += rk4(ns.dudt, 0, u, 1e-3)
    assert isinstance(u, cp.ndarray)
    assert bool(cp.isfinite(u.real).all()) and bool(cp.isfinite(u.imag).all())


def test_cpu_and_gpu_agree():
    """The same initial field must give the same dudt and power on CPU and GPU."""
    import cupy as cp

    ns_cpu = NavierStokes((16, 16, 16), (1, 1, 1), 1 / 1600, dealias=True)
    ns_gpu = NavierStokes((16, 16, 16), (1, 1, 1), 1 / 1600, dealias=True, device="cuda")

    u_cpu = _random_incompressible(ns_cpu, np, seed=1)
    u_gpu = cp.asarray(u_cpu)

    d_cpu = ns_cpu.dudt(0.0, u_cpu)
    d_gpu = ns_gpu.dudt(0.0, u_gpu)
    # The resolved modes are O(1e2-1e3), so rtol governs them; the k=0 mode is
    # the (near-zero) spatial mean of u x omega, whose value is pure FFT round-off
    # (~1e-12). atol must clear that floor, which differs slightly between the CPU
    # (PocketFFT) and GPU (cuFFT) transforms.
    np.testing.assert_allclose(cp.asnumpy(d_gpu), d_cpu, rtol=1e-10, atol=1e-9)

    np.testing.assert_allclose(
        float(ns_gpu.power(u_gpu)), float(ns_cpu.power(u_cpu)), rtol=1e-10
    )


def test_gpu_reductions_match_numpy():
    """The parallel-reduction helper (``ns.parnp``, backed by muGrid's
    GPU-aware Communicator) reduces a CuPy buffer on-device and agrees with
    numpy. This exercises the sum/min/max/mean paths the screen diagnostics
    and ``power`` rely on."""
    import cupy as cp

    ns = NavierStokes((8, 8, 8), (1, 1, 1), 1 / 1600, dealias=True, device="cuda")
    red = ns.parnp

    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((3, 8, 8, 8)).astype(np.float64)
    a_cp = cp.asarray(a_np)

    np.testing.assert_allclose(float(red.sum(a_cp)), a_np.sum(), rtol=1e-12)
    np.testing.assert_allclose(float(red.min(a_cp)), a_np.min(), rtol=1e-12)
    np.testing.assert_allclose(float(red.max(a_cp)), a_np.max(), rtol=1e-12)
    np.testing.assert_allclose(float(red.mean(a_cp)), a_np.mean(), rtol=1e-12)
