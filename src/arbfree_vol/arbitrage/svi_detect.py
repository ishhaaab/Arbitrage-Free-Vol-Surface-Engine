from arbfree_vol.svi.model import SVIParams, svi_g, svi_total_variance
from arbfree_vol.arbitrage.report import ArbitrageViolation, ArbitrageReport, ViolationType
from math import sqrt
from numpy import linspace
from scipy.optimize import minimize_scalar


def min_total_variance(params: SVIParams)-> float:
    """Returns the minimum total variance ie the min value of the SVI curve
    found at dw(k)/dk=0 where w(k)= a + b(rho(k-m) + sqrt((k-m)**2+sigma**2))
    that gives us w min= a + b* sigma* sqrt(1-rho**2)"""

    return params.a + params.b* params.sigma* sqrt(1-params.rho**2)


def _check_min_variance(params: SVIParams,
                        violations: list[ArbitrageViolation]) -> None:
    """Checks if given SVI's minimum total variance is non-negative.

    If w_min = a + b * sigma * sqrt(1 - rho^2) < -tolerance
    there exists a k where total variance is negative — a no-arb violation.
    """
    total_variance_min= min_total_variance(params)

    tolerance= 1e-4
    if total_variance_min < - tolerance :
        violations.append(ArbitrageViolation(
            kind= ViolationType.NEGATIVE_VARIANCE,
            detail=f"SVI min total variance is negative: w_min={total_variance_min:.4f}",
            magnitude= -total_variance_min,
            offending=(),
        ))


def _check_butterfly(params: SVIParams,
                     violations: list[ArbitrageViolation],
                     k_min: float = -3.0,
                    k_max: float = 3.0) -> None:
    """Check that the SVI curve admits no butterfly arbitrage.

    Uses Gatheral's g(k): the density condition g(k) >= 0 for all k
    in [k_min, k_max].  A coarse grid scan locates the approximate
    minimum, then bounded minimization refines it locally.  This
    avoids being trapped by the sharp spike at k=m.
    """
    a, b, rho, m, sigma= params.a, params.b, params.rho, params.m, params.sigma
    
    def single_g(k):
        return svi_g(k, a, b, rho, m, sigma)
        
    # coarse grid scan using 121 points to find the approximate minimum
    N_GRID = 121
    k_grid = linspace(k_min, k_max, N_GRID)
    g_min = float('inf')
    min_idx = 0
    for i in range(N_GRID):
        g_val = single_g(k_grid[i])
        if g_val < g_min:
            g_min = g_val
            min_idx = i

    # refine with local bounded minimization around the best grid point
    bracket_lo = k_grid[max(0, min_idx - 3)]
    bracket_hi = k_grid[min(N_GRID - 1, min_idx + 3)]
    result = minimize_scalar(fun=single_g, method='bounded', bounds=(bracket_lo, bracket_hi))
    value, location = result.fun, result.x # type: ignore 

    tolerance= 1e-4
    if value < -tolerance:
        violations.append(ArbitrageViolation(
            kind=ViolationType.BUTTERFLY,
            detail=f"SVI butterfly arbitrage: g={value:.6f} < 0 at k={location:.4f} (negative risk-neutral density)",
            magnitude= -value,
            offending=(),
        ))


def _check_calendar(slices: list[tuple[float, SVIParams]],
                    violations: list[ArbitrageViolation],
                    k_grid= linspace(-1.5, 1.5, 121))-> None:
    """Check calendar arbitrage across adjacent SVI slices.

    Total variance must be non decreasing with maturity for each
    log moneyness k.  For each adjacent pair (T_i, T_{i+1}), this evaluates
    w_earlier(k) and w_later(k) on a k-grid and flags any contiguous band
    of k where w_earlier(k) > w_later(k) beyond tolerance.
    """

    by_T= sorted(slices, key= lambda x:x[0])
    tolerance= 1e-4    

    for i in range(len(by_T)-1):

        a1, b1, rho1, m1, sigma1= by_T[i][1].a, by_T[i][1].b, by_T[i][1].rho, by_T[i][1].m, by_T[i][1].sigma
        a2, b2, rho2, m2, sigma2= by_T[i+1][1].a, by_T[i+1][1].b, by_T[i+1][1].rho, by_T[i+1][1].m, by_T[i+1][1].sigma

        # contiguous-run state machine:
        # in_run tracks whether we are currently inside a violating band.
        # run_start remembers the grid index where the current band began.
        # max_gap tracks the worst violation inside the current band.
        in_run= False
        run_start= 0
        max_gap= 0.0

        for j in range(len(k_grid)):
            k= k_grid[j]
            w_earlier= svi_total_variance(k, a1, b1, rho1, m1, sigma1)
            w_later= svi_total_variance(k, a2, b2, rho2, m2, sigma2)
            gap= w_earlier - w_later

            # positive gap means the earlier slice has MORE total variance
            # at this k, a calendar arbitrage violation at this point.
            if gap > tolerance:
                if not in_run:
                    # start a new violation band at this index
                    run_start= j
                    max_gap= gap
                    in_run= True
                else:
                    # extend the current band, track the worst point
                    max_gap= max(max_gap, gap)

            else:
                # no violation at this grid point.
                # if we were inside a run, it just ended at j-1.
                if in_run:
                    violations.append(ArbitrageViolation(
                        kind= ViolationType.CALENDAR,
                        detail= f"calendar arbitrage: T={by_T[i][0]} -> T={by_T[i+1][0]}, "
                                f"k=[{k_grid[run_start]:.4f}, {k_grid[j-1]:.4f}], "
                                f"worst gap={max_gap:.6f}",
                        magnitude= max_gap,
                        offending=(),
                    ))
                    in_run= False

        # if a run was still open when the grid ended, close it now
        if in_run:
            violations.append(ArbitrageViolation(
                kind= ViolationType.CALENDAR,
                detail= f"calendar arbitrage: T={by_T[i][0]} -> T={by_T[i+1][0]}, "
                        f"k=[{k_grid[run_start]:.4f}, {k_grid[-1]:.4f}], "
                        f"worst gap={max_gap:.6f}",
                magnitude= max_gap,
                offending=(),
            ))


def detect_svi(params: SVIParams) -> ArbitrageReport:
    """Runs per slice no arbitrage checks on a single SVI slice.

    Checks:
      - w_min >= 0 (no negative total variance)
      - g(k) >= 0 (no butterfly / negative risk-neutral density)

    Does NOT check calendar consistency as that needs multiple slices.
    """
    violations: list[ArbitrageViolation] = []
    _check_min_variance(params, violations)
    _check_butterfly(params, violations)
    return ArbitrageReport(violations=violations)


def detect_svi_surface(slices: list[tuple[float, SVIParams]],
                       k_grid= linspace(-1.5, 1.5, 121)) -> ArbitrageReport:
    """Runs all SVI no arbitrage checks across a full surface.

    Checks:
      - per-slice: w_min >= 0, g(k) >= 0  (min variance, butterfly)
      - cross-slice: non-decreasing total variance with T (calendar)

    Returns one combined report with violations from all checks.
    """
    violations: list[ArbitrageViolation] = []
    for _, params in slices:
        _check_min_variance(params, violations)
        _check_butterfly(params, violations)
    _check_calendar(slices, violations, k_grid)
    return ArbitrageReport(violations=violations)
