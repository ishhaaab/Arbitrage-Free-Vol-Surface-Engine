"""Yield term structure — flat and interpolated curves.

Mirrors QuantLib's ``YieldTermStructure`` at a research-library scale:
``zero_rate(T)`` with linear interpolation on zero rates and flat
extrapolation outside the pillar range.  Discount factors are
``exp(-r(T)*T)``.

The curve is intentionally small — no bootstrapping, no convexity
adjustment.  For production curves plug in QuantLib or
``rateslib`` and adapt via :meth:`YieldTermStructure.from_callable`.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pillar:
    maturity: float  # T in years, >0
    zero_rate: float  # continuously compounded


class YieldTermStructure:
    """Continuously-compounded zero curve.

    Parameters
    ----------
    pillars:
        Sorted ``(T, r)`` pairs.  ``r`` is the continuously-compounded
        zero rate valid at ``T``.  Interpolation is linear on ``r``.
    day_count:
        Convention label stored for provenance (e.g. ``"ACT/365F"``);
        not used in interpolation.

    Construction helpers
    --------------------
    * :meth:`flat` — single-rate curve (preserves the old ``^IRX`` path)
    * :meth:`from_pillars` — from explicit pillars
    * :meth:`from_callable` — wrap any ``r(T)`` function (e.g. QuantLib)
    """

    def __init__(
        self,
        pillars: list[tuple[float, float]] | None = None,
        *,
        day_count: str = "ACT/365F",
        _pillars_sorted: list[Pillar] | None = None,
    ) -> None:
        if _pillars_sorted is not None:
            self._pillars = _pillars_sorted
        elif pillars is None or len(pillars) == 0:
            self._pillars = [Pillar(1.0, 0.05)]
        else:
            pts = [Pillar(float(t), float(r)) for t, r in pillars]
            pts.sort(key=lambda p: p.maturity)
            self._pillars = pts
        self.day_count = day_count
        self._ts = [p.maturity for p in self._pillars]
        self._rs = [p.zero_rate for p in self._pillars]

    @classmethod
    def flat(cls, rate: float, *, day_count: str = "ACT/365F") -> YieldTermStructure:
        return cls([(1.0, float(rate))], day_count=day_count)

    @classmethod
    def from_pillars(
        cls,
        pillars: list[tuple[float, float]],
        *,
        day_count: str = "ACT/365F",
    ) -> YieldTermStructure:
        return cls(pillars, day_count=day_count)

    @classmethod
    def from_callable(
        cls,
        func,  # r(T) -> float
        *,
        day_count: str = "ACT/365F",
        pillars_for_inspect: list[tuple[float, float]] | None = None,
    ) -> YieldTermStructure:
        """Wrap an arbitrary ``r(T)`` callable.

        Stores a few sample pillars for ``pillars`` inspection; pricing
        always calls ``func`` directly.
        """

        class _CallableCurve(YieldTermStructure):
            def zero_rate(self, T: float) -> float:  # type: ignore[override]
                if T <= 0:
                    return float(func(1e-8))
                return float(func(float(T)))

            @property
            def pillars(self) -> list[Pillar]:
                if pillars_for_inspect is not None:
                    return [Pillar(float(t), float(r)) for t, r in pillars_for_inspect]
                return self._pillars

        inst = _CallableCurve.__new__(_CallableCurve)
        # init base state for inspection
        if pillars_for_inspect is not None:
            pts = [Pillar(float(t), float(r)) for t, r in pillars_for_inspect]
            pts.sort(key=lambda p: p.maturity)
            inst._pillars = pts
        else:
            inst._pillars = [Pillar(0.25, float(func(0.25))), Pillar(10.0, float(func(10.0)))]
        inst.day_count = day_count
        inst._ts = [p.maturity for p in inst._pillars]
        inst._rs = [p.zero_rate for p in inst._pillars]
        inst._func = func  # type: ignore[attr-defined]
        return inst  # type: ignore[return-value]

    @property
    def pillars(self) -> list[Pillar]:
        return list(self._pillars)

    def zero_rate(self, T: float) -> float:
        """Continuously-compounded zero rate at ``T``.

        Linear on ``r``, flat extrapolation outside the pillar range.
        """
        if T <= 0:
            return float(self._rs[0])
        if len(self._pillars) == 1:
            return float(self._rs[0])
        if T <= self._ts[0]:
            return float(self._rs[0])
        if T >= self._ts[-1]:
            return float(self._rs[-1])
        idx = bisect.bisect_left(self._ts, T)
        t0, r0 = self._ts[idx - 1], self._rs[idx - 1]
        t1, r1 = self._ts[idx], self._rs[idx]
        w = (T - t0) / (t1 - t0) if t1 != t0 else 0.0
        return float(r0 + w * (r1 - r0))

    def discount(self, T: float) -> float:
        if T <= 0:
            return 1.0
        return math.exp(-self.zero_rate(T) * T)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward between ``t1`` and ``t2`` from discounts."""
        if t2 <= t1 or t1 < 0:
            raise ValueError("need 0 <= t1 < t2")
        d1 = self.discount(t1)
        d2 = self.discount(t2)
        return math.log(d1 / d2) / (t2 - t1)

    def __repr__(self) -> str:  # pragma: no cover
        ps = ", ".join(f"({p.maturity:.4g},{p.zero_rate:.4g})" for p in self._pillars[:4])
        more = "…" if len(self._pillars) > 4 else ""
        return f"YieldTermStructure[{ps}{more}]({self.day_count})"
