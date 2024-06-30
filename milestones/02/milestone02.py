import sys
import time

import numpy as np
from mpi4py import MPI

from muNavierStokes import NavierStokes, rk4
from muGrid import FileIONetCDF, OpenMode

# Simulation parameters
viscosity = 1 / 1600
nb_grid_pts = (32, 32, 3)
physical_size = (1, 1, 1)
timestep = 0.001

# Initial condition
velocity_amplitude = 1

# I/O parameters
nb_steps = 100000
screen_interval = 100  # output to screen every `screen_interval` steps
dump_interval = 100  # dump every `dump_interval` steps

# Store rank into a convenience variable
rank = MPI.COMM_WORLD.Get_rank()

# Setup Navier-Stokes solver
ns = NavierStokes(nb_grid_pts, physical_size, viscosity, dealias=False, engine='pfft', communicator=MPI.COMM_WORLD)

# Print which FFT engine we are using
if rank == 0:
    print(f'FFT engine: {ns.fft.__class__.__name__}')

# Get spatial coordinates
x, y, z = ns.fft.coords

# Initialize velocity field
u_cxyz = ns.fft.real_space_field('u_cxyz', 3)
u_cxyz.p = velocity_amplitude * np.array([
    np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),  # * np.cos(2 * np.pi * z),
    -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y),  # * np.cos(2 * np.pi * z),
    np.zeros_like(x)
])
u_cqks = ns.fft.fourier_space_field('u_cqks', 3)
ns.fft.fft(u_cxyz, u_cqks)
uarr_cqks = u_cqks.p * ns.fft.normalisation

# Open file for writing velocity field; this uses parallel I/O
file = FileIONetCDF('navier_stokes.nc', OpenMode.Overwrite, communicator=MPI.COMM_WORLD)
# Register the field collection of the FFT object
file.register_field_collection(ns.fft.real_field_collection)

# This holds a timestamp used for reporting the frame rate
last_time = None
u_cqks = ns.fft.fourier_space_field('u_cqks', 3)
for n in range(nb_steps):
    # Output to screen
    if n % screen_interval == 0:
        # Compute velocity field in real space
        u_cqks.p = uarr_cqks
        ns.fft.ifft(u_cqks, u_cxyz)
        if rank == 0:
            if last_time is not None:
                frames_per_second = screen_interval / (time.time() - last_time)
                sys.stdout.write(
                    f'{n * 100 / nb_steps:>5.3}% - {n * timestep:>9.3} - {np.min(u_cxyz.p):>9.3} / {np.mean(u_cxyz.p):>9.3} / {np.max(u_cxyz.p):>9.3} - {ns.power(uarr_cqks):>9.3} - {frames_per_second:10.5} frames/s\n')
            else:
                sys.stdout.write(
                    f'{n * 100 / nb_steps:>5.3}% - {n * timestep:>9.3} - {np.min(u_cxyz.p):>9.3} / {np.mean(u_cxyz.p):>9.3} / {np.max(u_cxyz.p):>9.3} - {ns.power(uarr_cqks):>9.3}\n')
            sys.stdout.flush()
        last_time = time.time()

    # Integrate velocity field
    uarr_cqks += rk4(ns.dudt, 0, uarr_cqks, timestep)

    # Output to file
    if n % dump_interval == 0:
        # Compute velocity field in real space
        u_cqks.p = uarr_cqks
        ns.fft.ifft(u_cqks, u_cxyz)
        # Append frame to file and write all registered fields
        file.append_frame().write()

# Close file
file.close()
