"""Pseudo-spectral direct numerical simulation of the incompressible Navier-Stokes equations on top of muGrid."""

from .NavierStokes import NavierStokes
from .RungeKutta import rk4

__version__ = "0.1.0"

__all__ = ["NavierStokes", "rk4"]
