import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset

file = Dataset('navier_stokes_slice.nc', 'r')

u_fxy = file.variables['velocity']

fig = plt.figure()
data = u_fxy[-1]
pmesh = plt.pcolormesh(data)


# plt.colorbar()

def animate(i):
    data = u_fxy[i]
    pmesh.set_array(data.flatten())
    return pmesh


ani = animation.FuncAnimation(fig, animate, frames=u_fxy.shape[0], interval=100)
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('navier_stokes.mp4', writer=ffwriter)
