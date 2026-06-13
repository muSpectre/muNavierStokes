import numpy as np
import matplotlib.pyplot as plt

from muGrid import FFTEngine
from netCDF4 import Dataset

nb_bins = 20
physical_size = (1, 1, 1)

plt.figure()

with Dataset('navier_stokes.nc', 'r') as file:
    nb_grid_pts = (file.dimensions['nx'].size, file.dimensions['ny'].size, file.dimensions['nz'].size)
    fft = FFTEngine(nb_grid_pts)
    grid_spacing = np.array(physical_size) / np.array(nb_grid_pts)
    max_wavevector = np.mean(2 / 3 * np.pi / grid_spacing)
    wavevector_cqks = (2 * np.pi * fft.fftfreq.T / grid_spacing).T
    abs_wavevector_qks = np.sqrt(np.sum(wavevector_cqks ** 2, axis=0))

    bin_width = max_wavevector / nb_bins
    nb_values_q = np.bincount((abs_wavevector_qks.flatten() / bin_width).astype(int), minlength=nb_bins)
    print(nb_values_q)
    wavevector_q = np.bincount((abs_wavevector_qks.flatten() / bin_width).astype(int),
                               weights=np.abs(abs_wavevector_qks.flatten()) ** 2,
                               minlength=nb_bins) / nb_values_q

    u_cxyz = fft.real_space_field('u_cxyz', 3)
    u_cqks_field = fft.fourier_space_field('u_cqks', 3)
    for frame, u_csxyz in enumerate(file.variables['velocity'][::1]):
        data = u_csxyz[0]
        u_ampl = np.std(u_csxyz)
        u_cxyz.p[...] = u_csxyz[:, 0, :, :, :]
        fft.fft(u_cxyz, u_cqks_field)
        u_cqks = u_cqks_field.p
        energy_spectrum_q = np.bincount((abs_wavevector_qks.flatten() / bin_width).astype(int),
                                        weights=np.real(np.sum(u_cqks * np.conj(u_cqks), axis=0)).flatten(),
                                        minlength=nb_bins) / nb_values_q
        dissipation_spectrum_q = np.bincount((abs_wavevector_qks.flatten() / bin_width).astype(int),
                                             weights=(abs_wavevector_qks ** 2 * np.real(
                                                 np.sum(u_cqks * np.conj(u_cqks), axis=0))).flatten(),
                                             minlength=nb_bins) / nb_values_q

        plt.loglog(wavevector_q, energy_spectrum_q, label=f'frame {frame}')

x = np.logspace(1, 4, 101)
plt.plot(x, 1e8 * x ** (-5 / 3), 'k--', label=r'$k^{-5/3}$')

plt.ylim(1, 10 ** 7)

plt.legend()
plt.show()
