import numpy as np

from NuMPI.Tools import Reduction

from muFFT import FFT


class NavierStokes:
    def __init__(self, nb_grid_pts, physical_size=(1, 1, 1), viscosity=0.001, dealias=True, engine='mpi',
                 communicator=None):
        self._nb_grid_pts = nb_grid_pts
        self._physical_size = physical_size
        self._viscosity = viscosity
        self._dealias = dealias
        self._engine = engine
        self._communicator = communicator

        self._init_fft()

    def _init_fft(self):
        # Initialize helper variables
        self._parnp = Reduction(self._communicator)
        self._grid_spacing = np.array(self._physical_size) / np.array(self._nb_grid_pts)

        # Create FFT engine
        self._fft = FFT(self._nb_grid_pts, engine=self._engine, communicator=self._communicator)

        # Pre-compute wavevectors
        self._wavevector_cqks = (2 * np.pi * self._fft.fftfreq.T / self._grid_spacing).T
        self._zero_wavevector_qks = (self._wavevector_cqks.T == np.zeros(3, dtype=int)).T.all(axis=0)
        self._zero_wavevectorx_qks = self._wavevector_cqks[0] == 0
        self._wavevector_sq_qks = np.sum(self._wavevector_cqks ** 2, axis=0)
        self._wavevector0_sq_qks = self._wavevector_sq_qks.copy()
        self._wavevector0_sq_qks[self._zero_wavevector_qks] = 1.0  # to avoid divide by zero
        self._inv_wavevector_cqks = self._wavevector_cqks / self._wavevector0_sq_qks  # k / |k|^2

        # Dealiasing field and frozen wavevectors
        self._dealias_wavevector_c = 2 * np.pi / (3 * self._grid_spacing)
        self._dealias_qks = np.all((np.abs(self._wavevector_cqks).T < self._dealias_wavevector_c).T, axis=0)

    @property
    def fft(self):
        return self._fft

    def dudt(self, t, uarr_cqks):
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
        u_cqks = self._fft.fourier_space_field('u_cqks', 3)
        u_cxyz = self._fft.real_space_field('u_cxyz', 3)
        curlu_cqks = self._fft.fourier_space_field('curlu_cqks', 3)
        curlu_cxyz = self._fft.real_space_field('curlu_cxyz', 3)
        ucurlu_cqks = self._fft.fourier_space_field('ucurlu_cqks', 3)
        ucurlu_cxyz = self._fft.real_space_field('ucurlu_cxyz', 3)

        # Copy numpy array to field
        u_cqks.p = uarr_cqks

        # Compute u x (nabla x u) = u x (curl u)
        curlu_cqks.p = np.cross(self._wavevector_cqks * 1j, u_cqks.p, axis=0)
        self._fft.ifft(curlu_cqks, curlu_cxyz)
        self._fft.ifft(u_cqks, u_cxyz)
        ucurlu_cxyz.p = np.cross(u_cxyz.p, curlu_cxyz.p, axis=0)
        self._fft.fft(ucurlu_cxyz, ucurlu_cqks)
        # Multiply result with dealiasing field to eliminate Gibbs ringing
        if self._dealias:
            ucurlu_cqks.p *= self._fft.normalisation * self._dealias_qks
        else:
            ucurlu_cqks.p *= self._fft.normalisation

        # Navier-Stokes equation
        return ucurlu_cqks.p \
            - self._viscosity * self._wavevector_sq_qks * uarr_cqks \
            - self._wavevector_cqks * np.sum(self._inv_wavevector_cqks * ucurlu_cqks.p, axis=0)

    def power(self, u_cqks, mask=None):
        p = 2 * np.sum(np.real(u_cqks * np.conj(u_cqks)), axis=0)
        p[self._zero_wavevectorx_qks] -= np.sum(
            np.real(u_cqks[:, self._zero_wavevectorx_qks] * np.conj(u_cqks[:, self._zero_wavevectorx_qks])), axis=0)
        if mask is None:
            return self._parnp.sum(p) / self._fft.normalisation
        else:
            return self._parnp.sum(p[mask]) / self._fft.normalisation

    def freeze_long_wavelength_amplitudes(self, u_cqks, target_power):
        p = self._power(u_cqks, self._freeze_mask)
        u_cqks[:, self._zero_wavevector_qks] = 0
        u_cqks[:, self._freeze_mask] *= np.sqrt(target_power / p)

    def to_incompressible(self, u_cqks):
        return u_cqks - np.sum(u_cqks * self._wavevector_cqks, axis=0) * self._inv_wavevector_cqks
