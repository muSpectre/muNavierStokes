import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

file = Dataset('navier_stokes.nc', 'r')

u_fcxyz = file.variables['u_cxyz']

for u_csxyz in u_fcxyz:
    u_cxyz = u_csxyz[:, 0, ...]
    abs_u_qks = np.sqrt(np.sum(np.abs(u_cxyz) ** 2, axis=0))
    plt.figure()
    plt.pcolormesh(u_cxyz[0, :, :, 0])
    plt.colorbar()
plt.show()
