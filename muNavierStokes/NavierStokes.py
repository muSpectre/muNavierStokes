"""
Pseudo-spectral solver for the incompressible Navier-Stokes equations.

The solver works in the rotational form of the incompressible Navier-Stokes
equation and integrates the Fourier representation of the velocity field in
time. All fast Fourier transforms and the underlying (optionally MPI-parallel)
data layout are provided by µGrid's :class:`muGrid.FFTEngine`.

Notes on the µGrid field interface
----------------------------------
* Fields are obtained from the engine with ``real_space_field(name, ncomp)``
  and ``fourier_space_field(name, ncomp)``. Repeated calls with the same name
  return the *same* field (and hence reuse the same memory).
* The numpy view of a field is exposed through the ``.p`` (pixel layout)
  property. This view is writable in place (``field.p[...] = value``) but the
  property itself has no setter, so plain assignment (``field.p = value``) is
  not possible.

Performance
-----------
The fields used by the right-hand side and the Runge-Kutta integrator are
allocated once and reused (no per-step allocation or repeated ``.p`` view
construction). Constant Fourier symbols (``i k`` and the viscous symbol
``nu |k|^2``) are precomputed. The integrator (:meth:`rk4_step`) and the
linear-combination parts of the right-hand side use µGrid's BLAS-like field
operations (``copy``/``scal``/``axpy``), which are fused, in place, and run on
host or device. The cross products and the per-pixel pressure projection are
the only remaining element-wise array temporaries.

GPU support
-----------
µGrid field buffers (``field.p``) are numpy arrays on the CPU and CuPy arrays
on the GPU. All precomputed coefficient arrays are moved to the device of the
field buffers (``self._xp``), and array *methods* / the array module
(``self._xp.cross``) are used instead of the numpy free functions. Pass
``device="cuda"`` (or ``"cuda:N"``) to run on the GPU.
"""

import numpy as np

from muGrid import Communicator, FFTEngine, linalg


class NavierStokes:
    def __init__(self, nb_grid_pts, physical_size=(1, 1, 1), viscosity=0.001, dealias=True,
                 communicator=None, device=None):
        self._nb_grid_pts = nb_grid_pts
        self._physical_size = physical_size
        self._viscosity = viscosity
        self._dealias = dealias
        self._communicator = communicator
        self._device = device

        self._init_fft()

    def _init_fft(self):
        # Parallel reduction helper and grid spacing. muGrid's communicator
        # provides GPU-aware, full-array global reductions (sum/min/max/mean)
        # via its `.reduction` adapter, replacing NuMPI's numpy-only Reduction.
        self._parnp = Communicator(self._communicator).reduction
        self._grid_spacing = np.array(self._physical_size) / np.array(self._nb_grid_pts)

        # FFT engine (µGrid selects PocketFFT on the CPU, cuFFT/rocFFT on the GPU)
        self._fft = FFTEngine(self._nb_grid_pts, communicator=self._communicator, device=self._device)

        # Array module (numpy on the CPU, cupy on the GPU) of the field buffers.
        # All coefficient arrays below are moved to this device so that the
        # element-wise operations in `dudt` never mix host and device memory.
        self._xp = self._array_module()

        # Angular wavevectors k = 2 pi * fftfreq / dx, shape (3, *fourier_subdomain).
        # fftfreq is always a host (numpy) array, so build on the host first.
        wavevector_cqks = (2 * np.pi * self._fft.fftfreq.T / self._grid_spacing).T
        wavevector_sq_qks = np.sum(wavevector_cqks ** 2, axis=0)  # |k|^2
        zero_wavevector_qks = wavevector_sq_qks == 0  # the k = 0 mode

        # The half-complex (r2c) transform stores only kx in [0, N/2]. Modes
        # with 0 < kx < N/2 stand in for a +/- pair and are counted twice in the
        # energy sum, while the kx = 0 plane and (for even Nx) the kx = Nyquist
        # plane are their own Hermitian conjugates and must be counted once.
        absfreqx_qks = np.abs(self._fft.fftfreq[0])
        kx_counted_once_qks = (absfreqx_qks == 0) | (absfreqx_qks == 0.5)

        # k / |k|^2 for the pressure (Leray) projection. The k = 0 entry is set
        # to zero; the projection leaves the mean flow untouched anyway.
        nonzero_sq_qks = np.where(zero_wavevector_qks, 1.0, wavevector_sq_qks)
        inv_wavevector_cqks = wavevector_cqks / nonzero_sq_qks

        # 2/3-rule dealiasing mask: True for the retained (lower 2/3) modes
        cutoff = 2 * np.pi / (3 * self._grid_spacing)
        dealias_qks = np.all((np.abs(wavevector_cqks).T < cutoff).T, axis=0)

        # Move the arrays used at run time onto the compute device (a no-op on
        # the CPU, where `self._xp is numpy`).
        self._wavevector_cqks = self._xp.asarray(wavevector_cqks)
        self._wavevector_sq_qks = self._xp.asarray(wavevector_sq_qks)
        self._inv_wavevector_cqks = self._xp.asarray(inv_wavevector_cqks)
        self._dealias_qks = self._xp.asarray(dealias_qks)
        self._kx_counted_once_qks = self._xp.asarray(kx_counted_once_qks)
        self._zero_wavevector_qks = zero_wavevector_qks  # host only (unused at run time)

        # Precomputed constant Fourier symbols (saves an array multiply + alloc
        # per `dudt` call): i k for the curl, and nu |k|^2 for the viscous term.
        self._i_wavevector_cqks = 1j * self._wavevector_cqks

        # Reusable scratch fields for the right-hand side, allocated once.
        self._u_cqks = self._fft.fourier_space_field('u_cqks', 3)
        self._u_cxyz = self._fft.real_space_field('u_cxyz', 3)
        self._curlu_cqks = self._fft.fourier_space_field('curlu_cqks', 3)
        self._curlu_cxyz = self._fft.real_space_field('curlu_cxyz', 3)
        self._ucurlu_cqks = self._fft.fourier_space_field('ucurlu_cqks', 3)
        self._ucurlu_cxyz = self._fft.real_space_field('ucurlu_cxyz', 3)

        # Reusable scratch fields for the Runge-Kutta integrator.
        self._rk_k = self._fft.fourier_space_field('_rk_k', 3)
        self._rk_tmp = self._fft.fourier_space_field('_rk_tmp', 3)
        self._rk_accum = self._fft.fourier_space_field('_rk_accum', 3)

        # Single-component real symbols on the Fourier collection, used as
        # per-pixel multipliers by `linalg.scal` (broadcast over the 3 velocity
        # components): the dealiasing mask and the viscous symbol nu |k|^2.
        fc = self._fft.fourier_space_collection
        self._dealias_field = fc.real_field('_dealias_symbol', 1)
        self._dealias_field.p[...] = self._dealias_qks
        self._visc_field = fc.real_field('_visc_symbol', 1)
        self._visc_field.p[...] = self._viscosity * self._wavevector_sq_qks

        # Three-component coefficient fields for µGrid's fused per-pixel
        # kernels (`linalg.cross` and `linalg.leray_project`), which operate on
        # fields rather than numpy/cupy arrays. Filled once and reused: the
        # curl symbol i k, the wavevector k, and k / |k|^2 for the projection.
        self._ik_field = fc.complex_field('_ik_symbol', 3)
        self._ik_field.p[...] = self._i_wavevector_cqks
        self._k_field = fc.real_field('_k_symbol', 3)
        self._k_field.p[...] = self._wavevector_cqks
        self._invk_field = fc.real_field('_invk_symbol', 3)
        self._invk_field.p[...] = self._inv_wavevector_cqks

    def _array_module(self):
        """Return the array module (numpy or cupy) of this engine's fields."""
        probe = self._fft.fourier_space_field('_array_module_probe', 1)
        if probe.is_on_gpu:
            import cupy
            return cupy
        return np

    @property
    def fft(self):
        """The underlying :class:`muGrid.FFTEngine`."""
        return self._fft

    @property
    def parnp(self):
        """Parallel reduction helper (MPI-aware ``min``/``max``/``mean``/``sum``)."""
        return self._parnp

    @property
    def array_module(self):
        """The array module (numpy or cupy) of the solver's fields."""
        return self._xp

    @property
    def wavevector_sq(self):
        """Squared wavevector magnitude ``|k|^2`` (Fourier-space array)."""
        return self._wavevector_sq_qks

    def dudt_into(self, t, y, out):
        """
        Compute the right-hand side of the (Fourier-space) Navier-Stokes
        equation for the velocity field ``y`` and store it in ``out``.

        Both ``y`` and ``out`` are 3-component Fourier-space µGrid fields. ``y``
        is not modified. This is the performance-oriented entry point used by
        :meth:`rk4_step`; the linear-combination and scaling steps use µGrid's
        in-place BLAS-like field operations.
        """
        uc, ur = self._u_cqks, self._u_cxyz
        cc, cr = self._curlu_cqks, self._curlu_cxyz
        nc, nr = self._ucurlu_cqks, self._ucurlu_cxyz

        # Vorticity in Fourier space: omega = i k x u  (fused cross product)
        linalg.cross(self._ik_field, y, cc)
        # Working copy of the velocity for the nonlinear product
        uc.p[...] = y.p
        # 2/3-rule dealiasing: band-limit velocity and vorticity to the lower
        # 2/3 of the spectrum *before* forming the product in real space.
        if self._dealias:
            linalg.scal(self._dealias_field, uc)
            linalg.scal(self._dealias_field, cc)
        self._fft.ifft(uc, ur)
        self._fft.ifft(cc, cr)
        linalg.cross(ur, cr, nr)  # u x omega in real space (fused cross)
        self._fft.fft(nr, nc)
        linalg.scal(self._fft.normalisation, nc)
        # Discard the aliased high-wavenumber band so no spurious energy is
        # injected into the cut-off modes (they then only decay viscously).
        if self._dealias:
            linalg.scal(self._dealias_field, nc)

        # out = N - nu |k|^2 u - k (k . N) / |k|^2  (projected onto div-free)
        linalg.copy(nc, out)  # out = N
        # viscous term: out -= nu |k|^2 u  (reuse uc, free after the ifft above)
        linalg.copy(y, uc)
        linalg.scal(self._visc_field, uc)
        linalg.axpy(-1.0, uc, out)
        # pressure projection (fused per-pixel contraction + rank-1 update):
        # out -= k (k . N) / |k|^2
        linalg.leray_project(self._k_field, self._invk_field, nc, out)

    def rk4_step(self, y, t, dt):
        """
        Advance the Fourier-space velocity field ``y`` by one classical
        fourth-order Runge-Kutta step, in place.

        ``y`` is a 3-component Fourier-space µGrid field. The stage combinations
        use µGrid's fused, in-place ``copy``/``axpy`` field operations, so the
        step allocates no per-stage temporaries.
        """
        k, tmp, accum = self._rk_k, self._rk_tmp, self._rk_accum

        linalg.copy(y, accum)                 # accum = y
        self.dudt_into(t, y, k)               # k = k1 = f(t, y)
        linalg.axpy(dt / 6, k, accum)

        linalg.copy(y, tmp)
        linalg.axpy(dt / 2, k, tmp)           # tmp = y + dt/2 k1
        self.dudt_into(t + dt / 2, tmp, k)    # k = k2
        linalg.axpy(dt / 3, k, accum)

        linalg.copy(y, tmp)
        linalg.axpy(dt / 2, k, tmp)           # tmp = y + dt/2 k2
        self.dudt_into(t + dt / 2, tmp, k)    # k = k3
        linalg.axpy(dt / 3, k, accum)

        linalg.copy(y, tmp)
        linalg.axpy(dt, k, tmp)               # tmp = y + dt k3
        self.dudt_into(t + dt, tmp, k)        # k = k4
        linalg.axpy(dt / 6, k, accum)

        linalg.copy(accum, y)                 # y <- y + dt/6 (k1 + 2 k2 + 2 k3 + k4)

    def dudt(self, t, uarr_cqks):
        """
        Array-based right-hand side of the Navier-Stokes equation (rotational
        form). Convenience wrapper around :meth:`dudt_into` that accepts and
        returns a plain numpy/cupy array; the in-loop integrator should use
        :meth:`rk4_step` instead.

        Parameters
        ----------
        t : float
            The current time.
        uarr_cqks : array_like
            The current Fourier-representation of the velocity field.

        Returns
        -------
        array_like
            The time derivative of the Fourier-representation of the velocity field.
        """
        self._rk_tmp.p[...] = uarr_cqks
        self.dudt_into(t, self._rk_tmp, self._rk_k)
        return self._rk_k.p.copy()

    def power(self, u_cqks, mask=None):
        """
        Total (or masked) spectral energy of the velocity field via Parseval's
        theorem. The factor of two accounts for the half-complex (r2c) storage;
        the kx = 0 and kx = Nyquist planes are self-conjugate and counted once.
        """
        p = 2 * (u_cqks * u_cqks.conj()).real.sum(axis=0)
        once = self._kx_counted_once_qks
        p[once] -= (u_cqks[:, once] * u_cqks[:, once].conj()).real.sum(axis=0)
        if mask is None:
            return self._parnp.sum(p) / self._fft.normalisation
        else:
            return self._parnp.sum(p[mask]) / self._fft.normalisation

    def to_incompressible(self, u_cqks):
        """Project a Fourier-space velocity field onto the divergence-free subspace."""
        return u_cqks - (u_cqks * self._wavevector_cqks).sum(axis=0) * self._inv_wavevector_cqks
