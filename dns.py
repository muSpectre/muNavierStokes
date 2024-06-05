import numpy as np
import muGrid
from muFFT import FFT

viscosity = 1
nb_grid_pts = [32, 32, 32]
fft = FFT(nb_grid_pts)

# Velocity field
u_cqks = fft.fourier_space_field('u_cqks', 3)
u_cqks.p = np.random.random(u_cqks.p.shape) - 0.5


def dudt(u_cqks):
    # Get fields; this will allocate on first call
    u_cxyz = fft.real_space_field('u_cxyz', 3)
    uu_cxyz = fft.real_space_field('uu_cxyz', 3)
    uu_cqks = fft.fourier_space_field('uu_cqks', 3)

    # Compute u x (nabla x u)
    fft.ifft(u_cqks, u_cxyz)
    uu_cqks.p = np.cross(fft.fftfreq, u_cqks.p, axis=0)
    fft.ifft(uu_cqks, uu_cxyz)
    uu_cxyz.p = np.cross(u_cxyz.p, uu_cxyz.p, axis=0)
    fft.fft(uu_cxyz, uu_cqks)

    # Compute dudt
    wavevector_cqks = fft.fftfreq
    zero_wavevector_qks = (wavevector_cqks.T == np.zeros(3, dtype=int)).T.all(axis=0)
    wavevector_sq_qks = np.sum(wavevector_cqks ** 2, axis=0)
    wavevector_sq_qks[zero_wavevector_qks] = 1.0  # to avoid divide by zero
    return viscosity * wavevector_sq_qks * u_cqks.p + wavevector_cqks * np.sum(wavevector_cqks * uu_cqks.p,
                                                                               axis=0) / wavevector_sq_qks - uu_cqks.p


dudt_cxyz = dudt(u_cqks)
