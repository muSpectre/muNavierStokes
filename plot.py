import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset

file = Dataset('navier_stokes.nc', 'r')

u_fcsxyz = file.variables['velocity']

fig = plt.figure()
pmesh = plt.pcolormesh(u_fcsxyz[0, 0, 0, :, :, 0])


# plt.colorbar()

def animate(i):
    pmesh.set_array(u_fcsxyz[i, 0, 0, :, :, 0].flatten())
    return pmesh


ani = animation.FuncAnimation(fig, animate, frames=u_fcsxyz.shape[0], interval=100)
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('navier_stokes.mp4', writer=ffwriter)
