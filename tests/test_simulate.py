"""
Tests for the simulation driver ``simulate.py``: the initial-condition helpers
and an end-to-end run that exercises the time loop and NetCDF output.
"""
import numpy as np
import pytest

import simulate
from muNavierStokes import NavierStokes


def make_solver(nb=(16, 16, 16), dealias=True):
    return NavierStokes(nb, simulate.PHYSICAL_SIZE, 1 / 1600, dealias=dealias)


def test_taylor_green_ic_is_divergence_free():
    ns = make_solver()
    uarr = simulate.taylor_green(ns, amplitude=1.0)
    div = np.sum(ns._wavevector_cqks * uarr, axis=0)
    assert np.max(np.abs(div)) < 1e-10


def test_taylor_green_ic_scales_with_amplitude():
    ns = make_solver()
    u1 = simulate.taylor_green(ns, amplitude=1.0)
    u3 = simulate.taylor_green(ns, amplitude=3.0)
    # Linear in amplitude, so the difference is at machine precision (a relative
    # tolerance would trip over the near-zero coefficients away from the modes).
    assert np.max(np.abs(u3 - 3.0 * u1)) < 1e-12


def test_turbulence_ic_is_divergence_free_with_nonempty_forcing():
    ns = make_solver()
    uarr, freeze_mask, frozen = simulate.turbulence(ns, amplitude=1.0, seed=42)
    div = np.sum(ns._wavevector_cqks * uarr, axis=0)
    assert np.max(np.abs(div)) < 1e-10
    assert freeze_mask.sum() > 0
    # the frozen amplitudes are exactly the masked modes of the field
    np.testing.assert_array_equal(frozen, uarr[:, freeze_mask])
    # the mean flow (k = 0) is never frozen
    assert not np.any(freeze_mask[ns.wavevector_sq == 0])


def test_turbulence_ic_is_reproducible_with_seed():
    a = simulate.turbulence(make_solver(), amplitude=1.0, seed=7)[0]
    b = simulate.turbulence(make_solver(), amplitude=1.0, seed=7)[0]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("ic", ["taylor-green", "turbulence"])
def test_end_to_end_writes_only_velocity(ic, tmp_path, monkeypatch):
    """Run the full driver for a few steps (both initial conditions, so the
    forcing path is exercised too) and check the NetCDF file contains only the
    velocity field with the expected number of frames."""
    import muGrid
    if not getattr(muGrid, "has_netcdf", False):
        pytest.skip("muGrid built without NetCDF support")
    netCDF4 = pytest.importorskip("netCDF4")

    out = tmp_path / "out.nc"
    argv = ["simulate.py", "-i", ic, "-n", "16", "16", "16",
            "-N", "5", "--dump-interval", "2", "--screen-interval", "2",
            "--seed", "0", "-o", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    simulate.main()

    with netCDF4.Dataset(out) as ds:
        assert list(ds.variables.keys()) == ["velocity"]
        # dumps happen at steps 0, 2, 4 -> 3 frames
        assert ds.variables["velocity"].shape[0] == 3
