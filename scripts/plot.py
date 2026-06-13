import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset

if os.path.exists('navier_stokes_slice.nc'):
    file = Dataset('navier_stokes_slice.nc', 'r')
else:
    file = Dataset('navier_stokes.nc', 'r')

u_fcsxyz = file.variables['velocity']

fig = plt.figure()
data = u_fcsxyz[-1]
if len(data.shape) == 5:
    pmesh = plt.pcolormesh(data[0, 0, :, :, 0])
else:
    pmesh = plt.pcolormesh(data)


# plt.colorbar()

def animate(i):
    data = u_fcsxyz[i]
    if len(data.shape) == 5:
        pmesh.set_array(data[0, 0, :, :, 0].flatten())
    else:
        pmesh.set_array(data.flatten())
    return pmesh


ani = animation.FuncAnimation(fig, animate, frames=u_fcsxyz.shape[0], interval=100)
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('navier_stokes.mp4', writer=ffwriter)
