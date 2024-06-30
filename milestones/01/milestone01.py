import numpy as np
import matplotlib.pyplot as plt

from muFFT import FFT

nb_grid_pts = (32, 32, 2)
physical_size = (1, 1, 1)
grid_spacing = np.array(physical_size) / np.array(nb_grid_pts)

fft = FFT(nb_grid_pts, engine='pocketfft')

# Pre-compute wavevectors
wavevector_cqks = (2 * np.pi * fft.fftfreq.T / grid_spacing).T


def curl(u_cxyz):
    """Computes the curl of a vector field in real space."""
    u_cqks = fft.fft(u_cxyz) * fft.normalisation
    return fft.ifft(1j * np.cross(wavevector_cqks, u_cqks, axis=0))


u_cxyz = np.ones([3, *fft.nb_subdomain_grid_pts])
curlu_cxyz = curl(u_cxyz)

np.testing.assert_allclose(curlu_cxyz, 0)

norm = np.array([0, 0, 1])
u_cxyz = np.cross(norm, fft.coords - 0.5, axis=0)
curlu_cxyz = curl(u_cxyz)

np.testing.assert_allclose(curlu_cxyz[0], 0)
np.testing.assert_allclose(curlu_cxyz[1], 0)

plt.quiver(*fft.coords[0:2, :, :, 0], u_cxyz[0, :, :, 0], u_cxyz[1, :, :, 0])
plt.show()

plt.pcolormesh(curlu_cxyz[2, :, :, 0])
plt.colorbar()
plt.show()

print(curlu_cxyz[2, 14:18, 14:18, 0])