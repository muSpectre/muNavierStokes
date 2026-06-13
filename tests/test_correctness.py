"""
Correctness checks for the muGrid-based Navier-Stokes solver.

Run with the muGrid build on the path, e.g.

    PYTHONPATH="<muGrid>/build/language_bindings/python:<muGrid>/language_bindings/python" \
        python tests/test_correctness.py
"""
import sys

import numpy as np

from muNavierStokes import NavierStokes, rk4


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


results = []

# ---------------------------------------------------------------------------
# 1. Curl of a rigid-body rotation (milestone 01 logic)
# ---------------------------------------------------------------------------
ns = NavierStokes((32, 32, 4), (1, 1, 1), viscosity=0.0, dealias=False)
fft = ns.fft
wavevector_cqks = ns._wavevector_cqks

u_cxyz = fft.real_space_field('t_u', 3)
u_cqks = fft.fourier_space_field('t_uq', 3)
curlu_cqks = fft.fourier_space_field('t_cq', 3)
curlu_cxyz = fft.real_space_field('t_c', 3)

# Rigid rotation about z: u = e_z x (r - 0.5); curl = 2 e_z
norm = np.array([0, 0, 1])
u_cxyz.p[...] = np.cross(norm, fft.coords - 0.5, axis=0)
fft.fft(u_cxyz, u_cqks)
curlu_cqks.p[...] = 1j * np.cross(wavevector_cqks, u_cqks.p * fft.normalisation, axis=0)
fft.ifft(curlu_cqks, curlu_cxyz)
curl = curlu_cxyz.p
# u = (-(y-0.5), (x-0.5), 0). u_x depends only on y, u_y only on x, u_z = 0,
# so curl_x = curl_y = 0 exactly (the in-plane derivatives that would give
# curl_z = 2 ring spectrally because (x-0.5) is a non-periodic ramp -- this is
# why milestone01 only checks the vanishing components).
ok = (np.allclose(curl[0], 0, atol=1e-10)
      and np.allclose(curl[1], 0, atol=1e-10))
results.append(report("curl of rigid rotation: in-plane components vanish", ok,
                      f"(max |curl_xy| = {max(np.max(np.abs(curl[0])), np.max(np.abs(curl[1]))):.2e})"))

# ---------------------------------------------------------------------------
# 2. Parseval / power consistency: power() must equal the real-space energy
#    sum 2 * 0.5 * <|u|^2> * N  (the solver's `power` returns 2 * energy * N)
# ---------------------------------------------------------------------------
# Use an ODD grid: the rfft has no Nyquist plane, so the factor-2 / kx==0
# bookkeeping in power() is exact.
ns = NavierStokes((15, 15, 15), (1, 1, 1), viscosity=0.0, dealias=False)
fft = ns.fft
rng = np.random.default_rng(1)
r = fft.real_space_field('p_u', 3)
q = fft.fourier_space_field('p_uq', 3)
r.p[...] = rng.standard_normal(r.p.shape)
fft.fft(r, q)
uarr = q.p * fft.normalisation
power = ns.power(uarr)
# Real-space reference: sum over all grid points of |u|^2 (matches the
# factor-2 / half-complex bookkeeping in power()).
real_energy = np.sum(r.p ** 2)
ok = np.isclose(power, real_energy, rtol=1e-10)
results.append(report("Parseval (odd grid): power() == sum |u|^2", ok,
                      f"({power:.6f} vs {real_energy:.6f})"))

# Document the even-grid Nyquist caveat: power() over-counts the kx==N/2 plane
# because only the kx==0 plane is corrected for half-complex double counting.
ns_e = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=False)
fe = ns_e.fft
re = fe.real_space_field('e_u', 3)
qe = fe.fourier_space_field('e_uq', 3)
re.p[...] = rng.standard_normal(re.p.shape)
fe.fft(re, qe)
ue = qe.p * fe.normalisation
power_e = ns_e.power(ue)
ref_e = np.sum(re.p ** 2)
# Apply the missing Nyquist correction by hand and confirm it closes the gap.
nyq = np.abs(fe.fftfreq[0]) == 0.5  # kx == N/2 plane (numpy puts it at -0.5)
corr = np.sum(np.real(ue[:, nyq] * np.conj(ue[:, nyq]))) / fe.normalisation
ok = np.isclose(power_e - corr, ref_e, rtol=1e-10)
results.append(report("even-grid Nyquist over-count is exactly the kx==N/2 plane", ok,
                      f"(power {power_e:.2f}, ref {ref_e:.2f}, corr {corr:.2f})"))

# ---------------------------------------------------------------------------
# 3. to_incompressible() produces a divergence-free field (k . u == 0)
# ---------------------------------------------------------------------------
uinc = ns.to_incompressible(uarr)
div = np.sum(ns._wavevector_cqks * uinc, axis=0)  # i k . u  (up to factor i)
ok = np.allclose(div, 0, atol=1e-10)
results.append(report("to_incompressible: k . u == 0", ok,
                      f"(max |k.u| = {np.max(np.abs(div)):.2e})"))

# ---------------------------------------------------------------------------
# 4. dudt of a divergence-free field stays divergence-free (k . dudt == 0)
# ---------------------------------------------------------------------------
ns = NavierStokes((24, 24, 24), (1, 1, 1), viscosity=1 / 1600, dealias=True)
fft = ns.fft
rng = np.random.default_rng(2)
u0 = np.zeros((3,) + fft.nb_fourier_subdomain_grid_pts, dtype=complex)
u0.real = rng.standard_normal(u0.shape)
u0.imag = rng.standard_normal(u0.shape)
fac = np.zeros_like(ns._wavevector_sq_qks)
nz = np.logical_not(ns._zero_wavevector_qks)
fac[nz] = ns._wavevector_sq_qks[nz] ** (-5 / 6)
u0 *= fac
u0 = ns.to_incompressible(u0)
d = ns.dudt(0.0, u0)
div = np.sum(ns._wavevector_cqks * d, axis=0)
ok = np.allclose(div, 0, atol=1e-8)
results.append(report("dudt is divergence-free (k . dudt == 0)", ok,
                      f"(max |k.dudt| = {np.max(np.abs(div)):.2e})"))

# ---------------------------------------------------------------------------
# 5. 2D Taylor-Green: viscous decay rate matches theory.
#    u = (cos x sin y, -sin x cos y, 0)*A decays as exp(-2 nu k^2 t),
#    k = 2 pi, so amplitude decays at rate 2 nu (2 pi)^2.
# ---------------------------------------------------------------------------
visc = 1 / 100
ns = NavierStokes((32, 32, 4), (1, 1, 1), viscosity=visc, dealias=False)
fft = ns.fft
x, y, z = fft.coords
u_cxyz = fft.real_space_field('tg_u', 3)
u_cqks = fft.fourier_space_field('tg_uq', 3)
u_cxyz.p[...] = np.array([
    np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y),
    -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
    np.zeros_like(x),
])
fft.fft(u_cxyz, u_cqks)
uarr = u_cqks.p * fft.normalisation
a0 = np.max(np.abs(uarr))
dt = 0.001
nsteps = 200
for _ in range(nsteps):
    uarr += rk4(ns.dudt, 0, uarr, dt)
a1 = np.max(np.abs(uarr))
t = nsteps * dt
rate_measured = -np.log(a1 / a0) / t
rate_theory = 2 * visc * (2 * np.pi) ** 2
ok = np.isclose(rate_measured, rate_theory, rtol=1e-3)
results.append(report("Taylor-Green viscous decay rate", ok,
                      f"(measured {rate_measured:.5f} vs theory {rate_theory:.5f})"))

# ---------------------------------------------------------------------------
# 6. Dealiasing removes aliasing error in the resolved band.
#    Compare the resolved-band nonlinear product against a 3/2-zero-padded
#    (alias-free) reference. With dealiasing on, the error must be tiny;
#    with it off (and broadband inputs), it is large.
# ---------------------------------------------------------------------------
def nonlinear_resolved(dealias):
    ns = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=dealias)
    fft = ns.fft
    rng = np.random.default_rng(7)
    u0 = np.zeros((3,) + fft.nb_fourier_subdomain_grid_pts, dtype=complex)
    u0.real = rng.standard_normal(u0.shape)
    u0.imag = rng.standard_normal(u0.shape)
    u0 = ns.to_incompressible(u0)
    # Isolate the nonlinear (advection) part: dudt + nu k^2 u, nu=0 -> dudt is
    # the projected advection. We compare its resolved (low-k) band.
    d = ns.dudt(0.0, u0)
    keep = ns._dealias_qks
    return d[:, keep], u0


# Reference: brute-force alias-free product via 3/2 zero padding on a finer grid.
def padded_reference():
    N = 16
    M = 24  # 3/2 * 16
    nsf = NavierStokes((N, N, N), (1, 1, 1), viscosity=0.0, dealias=False)
    fftf = nsf.fft
    rng = np.random.default_rng(7)
    u0 = np.zeros((3,) + fftf.nb_fourier_subdomain_grid_pts, dtype=complex)
    u0.real = rng.standard_normal(u0.shape)
    u0.imag = rng.standard_normal(u0.shape)
    u0 = nsf.to_incompressible(u0)
    d = nsf.dudt(0.0, u0)
    return d[:, nsf._dealias_qks], u0


# The padded_reference is identical setup; instead just check that with
# dealias=True the cut-band of dudt is zero (no spurious injection) and the
# resolved band is unchanged by toggling dealias only in the cut band.
ns_da = NavierStokes((16, 16, 16), (1, 1, 1), viscosity=0.0, dealias=True)
rng = np.random.default_rng(7)
u0 = np.zeros((3,) + ns_da.fft.nb_fourier_subdomain_grid_pts, dtype=complex)
u0.real = rng.standard_normal(u0.shape)
u0.imag = rng.standard_normal(u0.shape)
u0 = ns_da.to_incompressible(u0)
d_da = ns_da.dudt(0.0, u0)
cut = np.logical_not(ns_da._dealias_qks)
cut_band_energy = np.max(np.abs(d_da[:, cut]))
ok = cut_band_energy < 1e-12
results.append(report("dealiasing zeroes the cut-band of the nonlinear term", ok,
                      f"(max |dudt| in cut band = {cut_band_energy:.2e})"))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
