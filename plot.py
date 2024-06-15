import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset

file = Dataset('navier_stokes.nc', 'r')

u_fcsxyz = file.variables['velocity']

fig = plt.figure()
data = u_fcsxyz[0, 0, 0, :, :, 0]
pmesh = plt.pcolormesh(data / np.std(data))


# plt.colorbar()

def animate(i):
    data = u_fcsxyz[i, 0, 0, :, :, 0]
    pmesh.set_array(data.flatten() / np.std(data))
    return pmesh


ani = animation.FuncAnimation(fig, animate, frames=u_fcsxyz.shape[0], interval=100)
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('navier_stokes.mp4', writer=ffwriter)
