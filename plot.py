import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

file = Dataset('navier_stokes.nc', 'r')

u_fcxyz = file.variables['u_cxyz']

u_cxyz = u_fcxyz[10, 0, ...]
abs_u_qks = np.sqrt(np.sum(np.abs(u_cxyz) ** 2, axis=0))
plt.pcolormesh(abs_u_qks[0, :, :])
plt.colorbar()
plt.show()