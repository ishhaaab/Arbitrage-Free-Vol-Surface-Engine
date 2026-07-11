from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.svi.data import slice_to_point
from arbfree_vol.models.surface import VolSurface, ExpirySlice


from scipy.optimize import least_squares
import numpy as np


def calibrate(points: list[tuple[float,float]]) -> SVIParams:
    """Fit raw SVI parameters (a, b, rho, m, sigma) to (k, w) points."""
    if len(points) < 5:
        raise ValueError("need at least 5 points to fit SVI")

    def residuals(p):
        """Return model minus market total variance at each (k, w)."""
        return [svi_total_variance(k, *p) - w for k, w in points]

    x0 = [min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]

    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    result = least_squares(residuals, x0, bounds=bounds)
    a, b, rho, m, sigma = result.x

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def calibrate_slice(surface: VolSurface, s: ExpirySlice) -> SVIParams:
    """Convenience: calls ``slice_to_point`` then ``calibrate``."""
    return calibrate(slice_to_point(surface, s))
