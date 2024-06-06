import sys

import numpy as np
from mpi4py import MPI

from muGrid import FileIONetCDF, OpenMode
from muFFT import FFT

rank = MPI.COMM_WORLD.Get_rank()

viscosity = 0.01
nb_grid_pts = (32, 32, 32)
physical_size = (1, 1, 1)
grid_spacing = np.array(physical_size) / np.array(nb_grid_pts)

nb_steps = 10000
nb_dump = 1000  # dump every nb_dump steps
timestep = 0.01

fft = FFT(nb_grid_pts, engine='pocketfft')

# Velocity field
u_cxyz = 0.1 * (np.random.random((3,) + nb_grid_pts) - 0.5)

sys.stdout.write(f'Initial - {np.min(u_cxyz)}/{np.max(u_cxyz)}\n')

# Fourier space velocity field
print(f'{np.min(u_cxyz)}/{np.max(u_cxyz)}')
u_cqks = fft.fft(u_cxyz)


def dudt(t, u):
    # Get fields; this will allocate on first call
    u_cqks = fft.fourier_space_field('u_cqks', 3)
    u_cxyz = fft.real_space_field('u_cxyz', 3)
    uu_cxyz = fft.real_space_field('uu_cxyz', 3)
    uu_cqks = fft.fourier_space_field('uu_cqks', 3)

    # Get wavevectors
    wavevector_cqks = (2 * np.pi * fft.fftfreq.T / grid_spacing).T * fft.normalisation
    zero_wavevector_qks = (wavevector_cqks.T == np.zeros(3, dtype=int)).T.all(axis=0)
    wavevector_sq_qks = np.sum(wavevector_cqks ** 2, axis=0)
    wavevector0_sq_qks = wavevector_sq_qks.copy()
    wavevector0_sq_qks[zero_wavevector_qks] = 1.0  # to avoid divide by zero

    # Compute u x (nabla x u)
    u_cqks.p = u
    fft.ifft(u_cqks, u_cxyz)
    u_cxyz.p *= fft.normalisation
    uu_cqks.p = np.cross(wavevector_cqks * 1j, u_cqks.p, axis=0)
    fft.ifft(uu_cqks, uu_cxyz)
    uu_cxyz.p = np.cross(u_cxyz.p, uu_cxyz.p, axis=0)
    fft.fft(uu_cxyz, uu_cqks)

    # Compute dudt
    return uu_cqks.p \
        - viscosity * wavevector_cqks * u_cqks.p \
        - wavevector_cqks * np.sum(wavevector_cqks * uu_cqks.p, axis=0) / wavevector0_sq_qks


def rk4(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


file = FileIONetCDF('navier_stokes.nc', OpenMode.Overwrite)
u_field_cxyz = fft.real_space_field('u_cxyz', 3)
u_cqks += rk4(dudt, 0, u_cqks, timestep)  # Create fields
file.register_field_collection(fft.real_field_collection)
for n in range(nb_steps):
    if rank == 0:
        sys.stdout.write(
            f'Step {n}/{nb_steps} - {np.min(u_field_cxyz.p):>7.3} / {np.mean(u_field_cxyz.p):>7.3} / {np.max(u_field_cxyz.p):>7.3}\r')
        sys.stdout.flush()
    u_cqks += rk4(dudt, 0, u_cqks, timestep)
    if n % nb_dump == 0:
        file.append_frame().write()
file.close()
