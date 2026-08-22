"""YAML-backed defaults for the ``arbfree`` CLI.

``config.yaml`` (or ``arbfree.yaml``) in the working directory supplies
defaults; CLI flags override them. ``pyyaml`` is imported lazily — only
when a config file is actually present — and is a declared dependency,
so a config file can always be honored.

An explicitly passed config path must be readable and take effect:
a missing file, a missing ``pyyaml`` install, or invalid YAML raises
instead of silently running with defaults. The implicit working-directory
discovery keeps the lenient warn-and-continue behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULTS: dict = {
    "day_count": "ACT/365F",
    "calendar": None,
    "risk_free": 0.05,
    "div_yield": 0.0,
    "use_fred_curve": False,
    "output": None,
    "cleaning": {"min_T": 7.0 / 365.0, "max_spread_ratio": 0.5, "max_log_moneyness": 1.5},
}


@dataclass(slots=True)
class Config:
    day_count: str = "ACT/365F"
    calendar: str | None = None
    risk_free: float = 0.05
    div_yield: float = 0.0
    use_fred_curve: bool = False
    output: str | None = None
    cleaning: dict = field(default_factory=lambda: dict(_DEFAULTS["cleaning"]))

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        d = dict(_DEFAULTS)
        # shallow merge + cleaning nested merge
        for k, v in (data or {}).items():
            if k == "cleaning" and isinstance(v, dict):
                merged = dict(d["cleaning"])
                merged.update(v)
                d[k] = merged
            else:
                d[k] = v
        return cls(
            day_count=str(d.get("day_count", d.get("dayCount", "ACT/365F"))),
            calendar=d.get("calendar"),
            risk_free=float(d.get("risk_free", d.get("riskFree", 0.05))),
            div_yield=float(d.get("div_yield", d.get("divYield", 0.0))),
            use_fred_curve=bool(d.get("use_fred_curve", False)),
            output=d.get("output"),
            cleaning=dict(d.get("cleaning", {})),
        )

    def as_dict(self) -> dict:
        return {
            "day_count": self.day_count,
            "calendar": self.calendar,
            "risk_free": self.risk_free,
            "div_yield": self.div_yield,
            "use_fred_curve": self.use_fred_curve,
            "output": self.output,
            "cleaning": dict(self.cleaning),
        }


def load_config(path: Path | None = None) -> Config:
    """Load ``Config`` from *path* or the default search.

    An explicit *path* raises on any problem (missing file, missing
    ``pyyaml``, invalid YAML, non-mapping document) so callers like
    ``arbfree --config ...`` fail loudly. The implicit search falls back
    to defaults with a warning.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                f"reading config file {p} requires pyyaml — install pyyaml"
            ) from exc
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise ValueError(f"config file {p} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"config file {p} is not a YAML mapping")
        return Config.from_dict(data)

    for p in (Path("config.yaml"), Path("arbfree.yaml")):
        if p.exists():
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("config file %s found but pyyaml not installed — ignoring", p)
                return Config.from_dict({})
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    return Config.from_dict(data)
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to parse config %s: %s", p, exc)
                return Config.from_dict({})
    return Config.from_dict({})


__all__ = ["Config", "load_config"]
