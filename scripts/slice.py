from netCDF4 import Dataset

file = Dataset('navier_stokes.nc', 'r')

with Dataset('navier_stokes_slice.nc', 'w') as slice:
    slice.createDimension('frame', None)
    slice.createDimension('nx', len(file.dimensions['nx']))
    slice.createDimension('ny', len(file.dimensions['ny']))
    slice.createVariable('velocity', 'f8', ('frame', 'nx', 'ny') )
    for frame, u_csxyz in enumerate(file.variables['velocity']):
        data = u_csxyz[0, 0, :, :, 0]
        slice.variables['velocity'][frame] = data
