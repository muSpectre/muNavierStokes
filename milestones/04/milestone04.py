import sys
import time

import numpy as np
from mpi4py import MPI

from muNavierStokes import NavierStokes, rk4
from muGrid import FileIONetCDF, OpenMode

# Simulation parameters
viscosity = 1 / 1600
nb_grid_pts = (64, 64, 64)
physical_size = (1, 1, 1)
timestep = 0.001
velocity_amplitude = 1
freeze_wavevector = 2 * np.pi * 3 / np.mean(physical_size)

# I/O parameters
nb_steps = 100000
screen_interval = 100  # output to screen every `screen_interval` steps
dump_interval = 100  # dump every `dump_interval` steps

# Store rank into a convenience variable
rank = MPI.COMM_WORLD.Get_rank()

# Setup Navier-Stokes solver
ns = NavierStokes(nb_grid_pts, physical_size, viscosity, dealias=False, engine='pfft', communicator=MPI.COMM_WORLD)

# Get spatial coordinates
x, y, z = ns.fft.coords

# Fourier space velocity field
uarr_cqks = np.zeros((3,) + ns.fft.nb_fourier_grid_pts, dtype=complex)
rng = np.random.default_rng()
uarr_cqks.real = rng.standard_normal(uarr_cqks.shape)
uarr_cqks.imag = rng.standard_normal(uarr_cqks.shape)
# Initial velocity field should decay as k^(-5/3) for the Kolmogorov spectrum
fac_qks = np.zeros_like(ns._wavevector_sq_qks)
fac_qks[np.logical_not(ns._zero_wavevector_qks)] = velocity_amplitude * ns._wavevector_sq_qks[
    np.logical_not(ns._zero_wavevector_qks)] ** (-5 / 6)
uarr_cqks *= fac_qks
# Project
uarr_cqks = ns.to_incompressible(uarr_cqks)

# Store frozen wavevectors
freeze_mask = ns._wavevector_sq_qks < freeze_wavevector ** 2
freeze_mask[ns._zero_wavevector_qks] = False  # Don't include the average velocity
assert ns._parnp.sum(freeze_mask) > 0
frozen_velocities = uarr_cqks[:, freeze_mask].copy()

if rank == 0:
    print(f'freezing wavevector: {freeze_wavevector}')

# Open file for writing velocity field; this uses parallel I/O
file = FileIONetCDF('navier_stokes.nc', OpenMode.Overwrite, communicator=MPI.COMM_WORLD)
# Register the field collection of the FFT object
u_cxyz = ns.fft.real_space_field('velocity', 3)
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
        frozen_power = ns.power(uarr_cqks, freeze_mask)
        total_power = ns.power(uarr_cqks)
        min_u = ns._parnp.min(u_cxyz.p)
        mean_u = ns._parnp.mean(u_cxyz.p)
        max_u = ns._parnp.max(u_cxyz.p)
        if rank == 0:
            if last_time is not None:
                frames_per_second = screen_interval / (time.time() - last_time)
                sys.stdout.write(
                    f'{n * 100 / nb_steps:>5.3}% - {n * timestep:>9.3} - {min_u:>9.3} / {mean_u:>9.3} / {max_u:>9.3} - {frozen_power:>9.3} / {total_power:>9.3} - {frames_per_second:10.5} frames/s\n')
            else:
                sys.stdout.write(
                    f'{n * 100 / nb_steps:>5.3}% - {n * timestep:>9.3} - {min_u:>9.3} / {mean_u:>9.3} / {max_u:>9.3} - {frozen_power:>9.3} / {total_power:>9.3}\n')
            sys.stdout.flush()
        last_time = time.time()

    # Integrate velocity field
    uarr_cqks += rk4(ns.dudt, 0, uarr_cqks, timestep)

    # Forcing
    # freeze_long_wavelength_amplitudes(uarr_cqks, freeze_amplitude ** 2)
    uarr_cqks[:, freeze_mask] = frozen_velocities

    # Output to file
    if n % dump_interval == 0:
        # Compute velocity field in real space
        u_cqks.p = uarr_cqks
        ns.fft.ifft(u_cqks, u_cxyz)
        # Append frame to file and write all registered fields
        file.append_frame().write()

# Close file
file.close()
