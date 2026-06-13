# Evaluate a 2D Taylor-Green vortex to compute the viscosity
# Vortex is initialized via
# u_cxyz.p = velocity_amplitude * np.array([
#     np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),  # * np.cos(2 * np.pi * z),
#     -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y),  # * np.cos(2 * np.pi * z),
#     np.zeros_like(x)
# ])

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset

nb_bins = 20
physical_size = (1, 1, 1)
timestep = 0.1

plt.figure()

ampl = []
with Dataset('navier_stokes.nc', 'r') as file:
    for frame, u_csxyz in enumerate(file.variables['velocity']):
        ampl += [u_csxyz.max()]

t = np.arange(len(ampl)) * timestep
b, a = np.polyfit(t, np.log(ampl), 1)
viscosity = -b / (2 * np.pi) ** 2 / 2
print(f'viscosity = {viscosity}')

tfine = np.arange(10 * len(ampl)) * timestep / 10
plt.plot(t, ampl, 'kx')
plt.plot(tfine, np.exp(a + b * tfine), 'r-')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
