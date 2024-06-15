import sys
import time

import numpy as np
from mpi4py import MPI

from muGrid import FileIONetCDF, OpenMode
from muFFT import FFT

# Simulation parameters
viscosity = 1 / 1600
nb_grid_pts = (32, 32, 32)
physical_size = (1, 1, 1)
grid_spacing = np.array(physical_size) / np.array(nb_grid_pts)
timestep = 0.01

# I/O parameters
nb_steps = 100000
screen_interval = 100  # output to screen every `screen_interval` steps
dump_interval = 100  # dump every `dump_interval` steps

# Store rank into a convenience variable
rank = MPI.COMM_WORLD.Get_rank()

# Create FFT engine
fft = FFT(nb_grid_pts, engine='pocketfft', communicator=MPI.COMM_WORLD)

# Print which FFT engine we are using
if rank == 0:
    print(f'FFT engine: {fft.__class__.__name__}')

# Get spatial coordinates
x, y, z = fft.coords

# Initialize velocity field
velocity_amplitude = 1
u_cxyz = fft.real_space_field('velocity', 3)
u_cxyz.p = velocity_amplitude * np.array([
    np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z),
    -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z),
    np.zeros_like(x)
])

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

# Dealiasing field
max_wavevector_c = 2 / 3 * np.pi / grid_spacing
dealias_qks = np.all((np.abs(wavevector_cqks).T < max_wavevector_c).T, axis=0)


def dudt(t, uarr_cqks):
    """
    This function implements the incompressible Navier-Stokes equation in its
    rotational form. It computes the time derivative of the Fourier-representation
    of the velocity field.

    Parameters
    ----------
    t : float
        The current time.
    uarr_cqks : array_like
        The current value of the Fourier-representation of the velocity field.

    Returns
    -------
    array_like
        The time derivative of the Fourier-representation of the velocity field.
    """
    # Get fields; this will reuse the same memory upon every call
    u_cqks = fft.fourier_space_field('u_cqks', 3)
    u_cxyz = fft.real_space_field('u_cxyz', 3)
    curlu_cqks = fft.fourier_space_field('curlu_cqks', 3)
    curlu_cxyz = fft.real_space_field('curlu_cxyz', 3)
    ucurlu_cqks = fft.fourier_space_field('ucurlu_cqks', 3)
    ucurlu_cxyz = fft.real_space_field('ucurlu_cxyz', 3)

    # Copy numpy array to field
    u_cqks.p = uarr_cqks

    # Compute u x (nabla x u) = u x (curl u)
    curlu_cqks.p = np.cross(wavevector_cqks * 1j, u_cqks.p, axis=0)
    fft.ifft(curlu_cqks, curlu_cxyz)
    fft.ifft(u_cqks, u_cxyz)
    ucurlu_cxyz.p = np.cross(u_cxyz.p, curlu_cxyz.p, axis=0)
    fft.fft(ucurlu_cxyz, ucurlu_cqks)
    # Multiply result with dealiasing field to eliminate Gibbs ringing
    ucurlu_cqks.p *= fft.normalisation * dealias_qks

    # Navier-Stokes equation
    return ucurlu_cqks.p \
        - viscosity * wavevector_sq_qks * u_cqks.p \
        - wavevector_cqks * np.sum(inv_wavevector_cqks * ucurlu_cqks.p, axis=0)


def rk4(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Implements the fourth-order Runge-Kutta method for numerical integration
    of multidimensional fields.

    Parameters
    ----------
    f : function
        The function to be integrated. It should take two arguments: time t
        and field y.
    t : float
        The current time.
    y : array_like
        The current value of the field.
    dt : float
        The time step for the integration.

    Returns
    -------
    np.ndarray
        The increment of the field required to obtain the value at t + dt.
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Open file for writing velocity field; this uses parallel I/O
file = FileIONetCDF('navier_stokes.nc', OpenMode.Overwrite, communicator=MPI.COMM_WORLD)
# Register the field collection of the FFT object
file.register_field_collection(fft.real_field_collection)

# This holds a timestamp used for reporting the frame rate
last_time = None
for n in range(nb_steps):
    # Output to screen
    if n % screen_interval == 0:
        # Compute velocity field in real space
        u_cqks.p = uarr_cqks
        fft.ifft(u_cqks, u_cxyz)
        if rank == 0:
            if last_time is not None:
                frames_per_second = screen_interval / (time.time() - last_time)
                sys.stdout.write(
                    f'Step {n:>5}/{nb_steps:<5} - {np.min(u_cxyz.p):>9.3} / {np.mean(u_cxyz.p):>9.3} / {np.max(u_cxyz.p):>9.3} - {frames_per_second:10.5} frames/s\n')
            else:
                sys.stdout.write(
                    f'Step {n:>5}/{nb_steps:<5} - {np.min(u_cxyz.p):>9.3} / {np.mean(u_cxyz.p):>9.3} / {np.max(u_cxyz.p):>9.3}\n')
            sys.stdout.flush()
        last_time = time.time()

    # Integrate velocity field
    uarr_cqks += rk4(dudt, 0, uarr_cqks, timestep)

    # Output to file
    if n % dump_interval == 0:
        # Compute velocity field in real space
        u_cqks.p = uarr_cqks
        fft.ifft(u_cqks, u_cxyz)
        # Append frame to file and write all registered fields
        file.append_frame().write()

# Close file
file.close()
