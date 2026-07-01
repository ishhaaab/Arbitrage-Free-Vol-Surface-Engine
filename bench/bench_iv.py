"""Benchmark the implied-vol hot path.

Run:  PYTHONPATH=src python bench/bench_iv.py
"""

import time
from datetime import date

from arbfree_vol.models.option import (
    BlackScholesInput,
    ImpliedVolInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.black_scholes import price
from arbfree_vol.pricing.implied_vol import implied_vol


def _make_price(strike: float, sigma: float) -> float:
    contract = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=strike,
        expiry_date=date(2027, 1, 1),
    )
    model = BlackScholesInput(
        contract=contract, spot=100.0, expiry_time=1.0,
        risk_free=0.05, div_yield=0.0, volatility=sigma,
    )
    return price(model)


def main(n: int = 5000) -> None:
    strikes = [80.0 + (i % 41) for i in range(n)]          # 80..120
    sigmas = [0.1 + 0.3 * ((i % 20) / 20) for i in range(n)]  # 0.1..0.4
    targets = [_make_price(k, s) for k, s in zip(strikes, sigmas)]

    inputs = [
        ImpliedVolInput(
            contract=OptionContract(
                symbol="X", option_type=OptionType.CALL, strike=k,
                expiry_date=date(2027, 1, 1),
            ),
            spot=100.0, expiry_time=1.0, risk_free=0.05, div_yield=0.0,
            market_price=p,
        )
        for k, p in zip(strikes, targets)
    ]

    t0 = time.perf_counter()
    for iv_in in inputs:
        implied_vol(iv_in)
    total = time.perf_counter() - t0

    print(
        f"{n} implied-vol solves: {total:.3f}s  ->  "
        f"{1e6 * total / n:.1f} us/solve  ({n / total:,.0f} solves/s)"
    )


if __name__ == "__main__":
    main()
