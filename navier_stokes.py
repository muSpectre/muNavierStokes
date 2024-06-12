import sys

import numpy as np
from mpi4py import MPI

from muGrid import FileIONetCDF, OpenMode
from muFFT import FFT

rank = MPI.COMM_WORLD.Get_rank()

viscosity = 1/1600
#viscosity = 0.01
nb_grid_pts = (32, 32, 32)
physical_size = (1, 1, 1)
grid_spacing = np.array(physical_size) / np.array(nb_grid_pts)

#nb_steps = 10000
nb_steps = 10
screen_interval = 100  # output to screen every `screen_interval` steps
dump_interval = 1000  # dump every `dump_interval` steps
timestep = 0.001
#timestep = 0.01

fft = FFT(nb_grid_pts, engine='pocketfft')
x, y, z = fft.coords

# Velocity field
velocity_amplitude = 0.1
u_cxyz = fft.real_space_field('u_cxyz', 3)
u_cxyz.p = velocity_amplitude * np.array([
    np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z),
    -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z),
    np.zeros_like(x)
])

sys.stdout.write(f'Initial - {np.min(u_cxyz)}/{np.max(u_cxyz)}\n')

# Fourier space velocity field
u_cqks = fft.fourier_space_field('u_cqks', 3)
fft.fft(u_cxyz, u_cqks)
uarr_cqks = u_cqks.p * fft.normalisation

# Pre-compute wavevectors
wavevector_cqks = (2 * np.pi * fft.fftfreq.T / grid_spacing).T
zero_wavevector_qks = (wavevector_cqks.T == np.zeros(3, dtype=int)).T.all(axis=0)
wavevector_sq_qks = np.sum(wavevector_cqks ** 2, axis=0)
wavevector0_sq_qks = wavevector_sq_qks.copy()
wavevector0_sq_qks[zero_wavevector_qks] = 1.0  # to avoid divide by zero
inv_wavevector_cqks = wavevector_cqks / wavevector0_sq_qks  # k / |k|^2

def dudt(t, uarr_cqks):
    # Get fields
    u_cqks = fft.fourier_space_field('u_cqks', 3)
    u_cxyz = fft.real_space_field('u_cxyz', 3)
    curlu_cqks = fft.fourier_space_field('curlu_cqks', 3)
    curlu_cxyz = fft.real_space_field('curlu_cxyz', 3)
    ucurlu_cqks = fft.fourier_space_field('ucurlu_cqks', 3)
    ucurlu_cxyz = fft.real_space_field('ucurlu_cxyz', 3)

    # Copy numpy array to field
    u_cqks.p = uarr_cqks

    print('---')
    print('u_cqks:', np.abs(u_cqks.p).max())

    # Compute u x (nabla x u) = u x (curl u)
    curlu_cqks.p = np.cross(wavevector_cqks * 1j, u_cqks.p, axis=0)
    fft.ifft(curlu_cqks, curlu_cxyz)
    print('curlu_cxyz:', np.abs(curlu_cxyz.p).max())
    fft.ifft(u_cqks, u_cxyz)
    u_cxyz.p *= fft.normalisation
    print('u_cxyz:', np.abs(u_cxyz.p).max())
    ucurlu_cxyz.p = np.cross(u_cxyz.p, curlu_cxyz.p, axis=0)
    print('ucurlu_cxyz:', np.abs(ucurlu_cxyz.p).max())
    fft.fft(ucurlu_cxyz, ucurlu_cqks)
    print('ucurlu_cqks:', np.abs(ucurlu_cqks.p).max())

    # Compute dudt
    return ucurlu_cqks.p \
        - viscosity * wavevector_sq_qks * u_cqks.p \
        - wavevector_cqks * np.sum(inv_wavevector_cqks * ucurlu_cqks.p, axis=0)


def rk4(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


file = FileIONetCDF('navier_stokes.nc', OpenMode.Overwrite)
file.register_field_collection(fft.real_field_collection)

for n in range(nb_steps):
    if rank == 0 and n % screen_interval == 0:
        u_cqks.p = uarr_cqks
        fft.ifft(u_cqks, u_cxyz)
        sys.stdout.write(
            f'Step {n}/{nb_steps} - {np.min(u_cxyz.p):>7.3} / {np.mean(u_cxyz.p):>7.3} / {np.max(u_cxyz.p):>7.3}\n')
        sys.stdout.flush()
    uarr_cqks += rk4(dudt, 0, uarr_cqks, timestep)
    if n % dump_interval == 0:
        u_cqks.p = uarr_cqks
        fft.ifft(u_cqks, u_cxyz)
        file.append_frame().write()

file.close()
