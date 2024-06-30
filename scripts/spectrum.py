import numpy as np
import matplotlib.pyplot as plt

from muFFT import FFT
from netCDF4 import Dataset

nb_bins = 20
physical_size = (1, 1, 1)

plt.figure()

with Dataset('navier_stokes.nc', 'r') as file:
    nb_grid_pts = (file.dimensions['nx'].size, file.dimensions['ny'].size, file.dimensions['nz'].size)
    fft = FFT(nb_grid_pts, engine='pocketfft')
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

    for frame, u_csxyz in enumerate(file.variables['velocity'][100::100]):
        data = u_csxyz[0]
        u_ampl = np.std(u_csxyz)
        u_cqks = fft.fft(u_csxyz[:, 0, :, :, :])
        spectrum_q = np.bincount((abs_wavevector_qks.flatten() / bin_width).astype(int),
                                 weights=(abs_wavevector_qks ** 2 * np.real(
                                     np.sum(u_cqks * np.conj(u_cqks), axis=0))).flatten(),
                                 minlength=nb_bins) / nb_values_q

        plt.plot(wavevector_q, spectrum_q, label=f'frame {frame}')

plt.xlim(0, 1000)
plt.legend()
plt.show()
