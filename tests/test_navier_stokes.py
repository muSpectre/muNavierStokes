"""
Functional and physical correctness tests for the muGrid-based Navier-Stokes
solver. Run with ``pytest`` (the muGrid build must be importable).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from muNavierStokes import NavierStokes, rk4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_incompressible(ns, seed=0):
    """A divergence-free random field with a k^(-5/3) spectrum."""
    shape = (3,) + ns.fft.nb_fourier_subdomain_grid_pts
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    k_sq = ns.wavevector_sq
    nonzero = k_sq > 0
    factor = np.zeros_like(k_sq)
    factor[nonzero] = k_sq[nonzero] ** (-5 / 6)
    return ns.to_incompressible(u * factor)


def to_real(ns, arr_cqks):
    """Inverse transform a Fourier-space array to a real-space numpy array."""
    k = ns.fft.fourier_space_field("_k", 3)
    k.p[...] = arr_cqks
    r = ns.fft.real_space_field("_r", 3)
    ns.fft.ifft(k, r)
    return r.p.copy()


# --------------------------------------------------------------------------- #
# Spectral operators
# --------------------------------------------------------------------------- #
def test_curl_rigid_rotation_inplane_components_vanish():
    """curl of u = e_z x (r - 1/2): the x/y components are identically zero
    (the z component rings because (x-1/2) is a non-periodic ramp)."""
    ns = NavierStokes((32, 32, 4), (1, 1, 1), viscosity=0.0, dealias=False)
    fft = ns.fft
    u = fft.real_space_field("u", 3)
    uk = fft.fourier_space_field("uk", 3)
    ck = fft.fourier_space_field("ck", 3)
    c = fft.real_space_field("c", 3)

    u.p[...] = np.cross(np.array([0, 0, 1]), fft.coords - 0.5, axis=0)
    fft.fft(u, uk)
    ck.p[...] = 1j * np.cross(ns._wavevector_cqks, uk.p * fft.normalisation, axis=0)
    fft.ifft(ck, c)

    assert_allclose(c.p[0], 0, atol=1e-10)
    assert_allclose(c.p[1], 0, atol=1e-10)


def test_to_incompressible_is_divergence_free():
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=False)
    u = random_incompressible(ns, seed=1)
    div = np.sum(ns._wavevector_cqks * u, axis=0)  # proportional to k . u
    assert_allclose(div, 0, atol=1e-10)


def test_dudt_is_divergence_free():
    ns = NavierStokes((24, 24, 24), (1, 1, 1), viscosity=1 / 1600, dealias=True)
    u = random_incompressible(ns, seed=2)
    div = np.sum(ns._wavevector_cqks * ns.dudt(0.0, u), axis=0)
    assert_allclose(div, 0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Energy / Parseval
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [15, 16])  # odd and even (Nyquist plane) grids
def test_power_equals_real_space_energy(n):
    """power() must equal sum |u|^2 for both odd and even grids (the latter
    exercises the Nyquist-plane correction)."""
    ns = NavierStokes((n, n, n), (1, 1, 1), viscosity=0.0, dealias=False)
    rng = np.random.default_rng(3)
    r = ns.fft.real_space_field("r", 3)
    q = ns.fft.fourier_space_field("q", 3)
    r.p[...] = rng.standard_normal(r.p.shape)
    ns.fft.fft(r, q)
    uarr = q.p * ns.fft.normalisation
    assert_allclose(ns.power(uarr), np.sum(r.p ** 2), rtol=1e-10)


def test_power_mask_partitions_total():
    """The energy in a mask plus its complement equals the total."""
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=False)
    u = random_incompressible(ns, seed=4)
    mask = ns.wavevector_sq < (2 * np.pi * 2) ** 2
    total = ns.power(u)
    assert_allclose(ns.power(u, mask) + ns.power(u, ~mask), total, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Dealiasing
# --------------------------------------------------------------------------- #
def test_dealiasing_zeroes_cut_band():
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=True)
    u = random_incompressible(ns, seed=5)
    d = ns.dudt(0.0, u)
    cut = ~ns._dealias_qks
    assert np.max(np.abs(d[:, cut])) < 1e-12


def test_dealiasing_matches_zero_padded_reference():
    """The dealiased nonlinear term must equal an alias-free (zero-padded)
    reference on the retained modes. This is the definitive 2/3-rule check."""
    N, N2, L = 24, 48, 1.0  # N/3 = 8: retained modes have all |m_i| < 8
    ns = NavierStokes((N, N, N), (L, L, L), 0.0, dealias=True)
    ns2 = NavierStokes((N2, N2, N2), (L, L, L), 0.0, dealias=False)

    # Divergence-free single modes c cos(2 pi m . x) with c . m = 0
    retained = [((7, 0, 0), (0, 1, 0)), ((0, 5, 0), (0, 0, 1)),
                ((3, 4, 0), (0, 0, 1)), ((0, 0, 6), (1, 0, 0))]
    cut = [((10, 0, 0), (0, 1, 0)), ((2, 9, 0), (0, 0, 1))]  # in the cut band

    def build(coords, modes):
        cx, cy, cz = coords
        u = np.zeros((3,) + cx.shape)
        for m, c in modes:
            phase = np.cos(2 * np.pi * (m[0] * cx + m[1] * cy + m[2] * cz))
            for i in range(3):
                u[i] += c[i] * phase
        return u

    def advect(solver, modes):
        u = solver.fft.real_space_field("u", 3)
        u.p[...] = build(solver.fft.coords, modes)
        uk = solver.fft.fourier_space_field("uk", 3)
        solver.fft.fft(u, uk)
        return solver.dudt(0.0, uk.p * solver.fft.normalisation)

    cutoff = 2 * np.pi / (3 * (L / N))  # physical |k| cutoff of the coarse grid
    mask2 = np.all((np.abs(ns2._wavevector_cqks).T < cutoff).T, axis=0)

    reference = to_real(ns2, advect(ns2, retained) * mask2)[:, ::2, ::2, ::2]
    dealiased = to_real(ns, advect(ns, retained + cut))  # full field, dealias on

    scale = np.max(np.abs(reference))
    assert np.max(np.abs(dealiased - reference)) / scale < 1e-12


# --------------------------------------------------------------------------- #
# Physics: time integration
# --------------------------------------------------------------------------- #
def test_taylor_green_viscous_decay_rate():
    """The 2D Taylor-Green vortex decays at the analytic rate 2 nu (2 pi)^2."""
    visc = 1 / 100
    ns = NavierStokes((32, 32, 4), (1, 1, 1), viscosity=visc, dealias=False)
    x, y, z = ns.fft.coords
    u = ns.fft.real_space_field("u", 3)
    uk = ns.fft.fourier_space_field("uk", 3)
    u.p[...] = np.array([
        np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y),
        -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
        np.zeros_like(x),
    ])
    ns.fft.fft(u, uk)
    uarr = uk.p * ns.fft.normalisation

    a0 = np.max(np.abs(uarr))
    dt, nsteps = 1e-3, 200
    for _ in range(nsteps):
        uarr += rk4(ns.dudt, 0, uarr, dt)
    rate = -np.log(np.max(np.abs(uarr)) / a0) / (nsteps * dt)
    assert_allclose(rate, 2 * visc * (2 * np.pi) ** 2, rtol=2e-3)


def test_inviscid_energy_is_conserved():
    """With nu = 0 the (projected, dealiased) advection does no net work, so the
    total kinetic energy is conserved up to the time-integration error."""
    ns = NavierStokes((24, 24, 24), (1, 1, 1), viscosity=0.0, dealias=True)
    u = random_incompressible(ns, seed=6)
    p0 = ns.power(u)
    for _ in range(50):
        u += rk4(ns.dudt, 0, u, 1e-3)
    assert_allclose(ns.power(u), p0, rtol=1e-4)


def test_incompressibility_preserved_during_integration():
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=1 / 800, dealias=True)
    u = random_incompressible(ns, seed=7)
    for _ in range(20):
        u += rk4(ns.dudt, 0, u, 1e-3)
    div = np.sum(ns._wavevector_cqks * u, axis=0)
    assert np.max(np.abs(div)) < 1e-8


# --------------------------------------------------------------------------- #
# The RK4 integrator
# --------------------------------------------------------------------------- #
def test_field_rk4_step_matches_array_rk4():
    """The in-place, field-based RK4 stepper must reproduce the array-based
    rk4(ns.dudt, ...) to within floating-point reordering."""
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=1 / 800, dealias=True)
    u0 = random_incompressible(ns, seed=11)

    # Array path
    y_array = u0 + rk4(ns.dudt, 0.0, u0, 1e-3)

    # Field path: advance an in-place state field with the BLAS-style stepper
    state = ns.fft.fourier_space_field("state", 3)
    state.p[...] = u0
    ns.rk4_step(state, 0.0, 1e-3)

    np.testing.assert_allclose(state.p, y_array, rtol=1e-10, atol=1e-12)


def test_rk4_is_fourth_order():
    """Halving the step must cut the error of dy/dt = -y by roughly 2^4 = 16."""
    def f(t, y):
        return -y

    def integrate(dt, T=1.0):
        y = np.array([1.0])
        for n in range(round(T / dt)):
            y = y + rk4(f, n * dt, y, dt)
        return y[0]

    exact = np.exp(-1.0)
    e_coarse = abs(integrate(0.1) - exact)
    e_fine = abs(integrate(0.05) - exact)
    assert e_fine < e_coarse
    assert 12 < e_coarse / e_fine < 20
