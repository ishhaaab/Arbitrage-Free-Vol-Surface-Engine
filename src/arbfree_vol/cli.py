"""CLI entry point — ``arbfree``.

Subcommands::

    arbfree repair chain.csv --spot 450 [--use-ssvi|--use-sabr] [--plot]
    arbfree detect chain.csv --spot 450 [--forward|--no-forward]
    arbfree price --spot 100 --strike 100 --expiry 0.25 --vol 0.2
    arbfree fetch --symbol SPY [--use-fred-curve] [--day-count ACT/365F]

DayCount/Calendar and YieldTermStructure/FRED options are threaded
through to ingestion: ``--day-count`` selects the year-fraction,
``--calendar USNYSE`` rolls expiries, ``--use-fred-curve`` pulls a
Treasury+SOFR curve with flat fallback.

Config file (``config.yaml``) is optional: flags override file values.
Requires ``pyyaml`` only when a config file is present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from arbfree_vol.config import load_config
from arbfree_vol.rates import YieldTermStructure, build_fred_curve
from arbfree_vol.time import Calendar, DayCount

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


# ── config helpers ──────────────────────────────────────────────────

def _load_config_dict(path: Path | None) -> dict:
    """Load config file into a plain dict (via :mod:`arbfree_vol.config`)."""
    try:
        cfg = load_config(path)
        return cfg.as_dict()
    except Exception as exc:  # pragma: no cover
        logger.warning("config load failed: %s", exc)
        return {}


def _resolve_day_count(value: str | None, cfg: dict) -> str:
    if value is not None:
        return value
    return str(cfg.get("day_count", cfg.get("dayCount", "ACT/365F")))


def _resolve_calendar(value: str | None, cfg: dict) -> str | None:
    if value is not None:
        if value.lower() in ("none", "off", "no"):
            return None
        return value
    v = cfg.get("calendar")
    if v is None or str(v).lower() in ("none", "off", "no"):
        return None
    return str(v)


def _as_day_count(value: str) -> DayCount:
    return DayCount(value)


def _as_calendar(name: str | None) -> Calendar | None:
    if name is None:
        return None
    return Calendar(name)


# ── helpers shared by repair/detect ─────────────────────────────────

def _parse_as_of(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _resolve_risk_free(args: argparse.Namespace, cfg: dict, day_count: str) -> float | YieldTermStructure:
    use_fred = bool(getattr(args, "use_fred_curve", False) or cfg.get("use_fred_curve"))
    fred_alias = bool(getattr(args, "fred_curve", False))
    if use_fred or fred_alias:
        as_of = _parse_as_of(getattr(args, "as_of", None))
        offline = bool(getattr(args, "offline", False))
        curve = build_fred_curve(as_of=as_of, day_count=day_count, offline=offline)
        print(f"[rates] FRED curve: {curve}")
        return curve
    raw = getattr(args, "risk_free", None)
    if raw is not None:
        return float(raw)
    return float(cfg.get("risk_free", cfg.get("riskFree", 0.05)))


def _resolve_div_yield(args: argparse.Namespace, cfg: dict) -> float:
    raw = getattr(args, "div_yield", None)
    if raw is not None:
        return float(raw)
    return float(cfg.get("div_yield", cfg.get("divYield", 0.0)))


def _write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[out] wrote {p}")


# ── subcommand: repair ──────────────────────────────────────────────

def _repair_validate_exclusive(args: argparse.Namespace) -> int | None:
    if getattr(args, "use_ssvi", False) and getattr(args, "use_sabr", False):
        print("error: --use-ssvi and --use-sabr are mutually exclusive", file=sys.stderr)
        return 2
    return None


def _repair_build_payload(report, metrics) -> dict:  # type: ignore[no-untyped-def]
    return {
        "metrics": {
            "n_rejected": metrics.n_rejected,
            "n_total_quotes": metrics.n_total_quotes,
            "n_slices_input": metrics.n_slices_input,
            "n_slices_fitted": metrics.n_slices_fitted,
            "n_violations_before": metrics.n_violations_before,
            "n_violations_after": metrics.n_violations_after,
            "rejection_rate": metrics.rejection_rate,
        },
        "fallback_slices": report.fallback_slices,
        "failed_slices": report.failed_slices,
        "repair_infeasible": report.repair_infeasible,
        "remaining_violations": [
            {"kind": v.kind.value, "detail": v.detail, "magnitude": v.magnitude}
            for v in report.remaining_violations.violations
        ],
        "fitted_slices": [
            {
                "expiry": fs.expiry_time,
                "forward": fs.forward_price,
                "params": fs.params.model_dump() if hasattr(fs.params, "model_dump") else str(fs.params),
            }
            for fs in report.fitted_slices
        ],
    }


def _cmd_repair(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.ingestion.loader import load_chain_csv
    from arbfree_vol.repair.engine import repair

    csv_path = Path(args.chain)
    if not csv_path.exists():
        print(f"error: chain file not found: {csv_path}", file=sys.stderr)
        return 2

    err = _repair_validate_exclusive(args)
    if err is not None:
        return err

    day_count_str = _resolve_day_count(getattr(args, "day_count", None), cfg)
    calendar_name = _resolve_calendar(getattr(args, "calendar", None), cfg)

    try:
        dc = _as_day_count(day_count_str)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    calendar = _as_calendar(calendar_name)
    risk_free_arg = _resolve_risk_free(args, cfg, day_count_str)
    div_yield = _resolve_div_yield(args, cfg)
    as_of_date = _parse_as_of(getattr(args, "as_of", None))

    print(f"[ingest] loading {csv_path} (spot={args.spot} day_count={dc.convention} calendar={calendar_name or 'none'})")
    try:
        surface, rejected = load_chain_csv(
            csv_path,
            spot=float(args.spot),
            risk_free=risk_free_arg,
            div_yield=div_yield,
            as_of=as_of_date,
            day_count=dc,
            calendar=calendar,
        )
    except Exception as exc:
        print(f"error loading chain: {exc}", file=sys.stderr)
        return 1

    kept = sum(len(sl.quotes) for sl in surface.slices)
    print(f"[clean] kept {kept} quotes in {len(surface.slices)} slices; rejected {len(rejected)}")
    if rejected and getattr(args, "verbose", False):
        for rec in rejected[:20]:
            print(f"  reject {rec.rule.value}: {rec.detail}")

    report = repair(surface, use_ssvi=bool(args.use_ssvi), use_sabr=bool(args.use_sabr))
    m = report.metrics
    print(
        f"[repair] rejected={m.n_rejected} total={m.n_total_quotes} slices {m.n_slices_fitted}/{m.n_slices_input} "
        f"violations {m.n_violations_before}->{m.n_violations_after} "
        f"fallback={report.fallback_slices} failed={report.failed_slices} infeasible={report.repair_infeasible}"
    )
    if report.remaining_violations.violations and getattr(args, "verbose", False):
        for v in report.remaining_violations.violations[:20]:
            print(f"  remaining {v.kind.value}: {v.detail}")

    out = getattr(args, "output", None) or cfg.get("output")
    if out:
        _write_json(out, _repair_build_payload(report, m))

    if getattr(args, "plot", False):
        try:
            plot_dir = Path(getattr(args, "plot_dir", None) or ".")
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[plot] (surface plots would write to {plot_dir}) — use demo scripts for full 7-plot output")
        except Exception as exc:
            print(f"[plot] skipped: {exc}", file=sys.stderr)

    return 0


# ── subcommand: detect ──────────────────────────────────────────────

def _cmd_detect(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.arbitrage.quote_detect import detect, detect_with_forward
    from arbfree_vol.ingestion.loader import load_chain_csv

    csv_path = Path(args.chain)
    if not csv_path.exists():
        print(f"error: chain file not found: {csv_path}", file=sys.stderr)
        return 2

    day_count_str = _resolve_day_count(getattr(args, "day_count", None), cfg)
    calendar_name = _resolve_calendar(getattr(args, "calendar", None), cfg)
    try:
        dc = _as_day_count(day_count_str)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    calendar = _as_calendar(calendar_name)
    as_of_date = _parse_as_of(getattr(args, "as_of", None))

    try:
        surface, _ = load_chain_csv(
            csv_path,
            spot=float(args.spot),
            risk_free=float(getattr(args, "risk_free", None) if getattr(args, "risk_free", None) is not None else cfg.get("risk_free", 0.05)),
            div_yield=float(getattr(args, "div_yield", None) if getattr(args, "div_yield", None) is not None else cfg.get("div_yield", 0.0)),
            as_of=as_of_date,
            clean=False,
            day_count=dc,
            calendar=calendar,
        )
    except Exception as exc:
        print(f"error loading chain: {exc}", file=sys.stderr)
        return 1

    use_forward = not bool(getattr(args, "no_forward", False))
    report = detect_with_forward(surface) if use_forward else detect(surface)
    if not report.violations:
        print("no violations detected")
        return 0

    from collections import Counter

    counts = Counter(v.kind.value for v in report.violations)
    print(f"violations: {len(report.violations)}  by kind: {dict(counts)}  (forward={'on' if use_forward else 'off'})")
    for v in report.violations:
        print(f"  [{v.kind.value}] {v.detail}  mag={v.magnitude:.4g}")

    out = getattr(args, "output", None) or cfg.get("output")
    if out:
        payload = [{"kind": v.kind.value, "detail": v.detail, "magnitude": v.magnitude} for v in report.violations]
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[out] wrote {out}")

    return 0


# ── subcommand: price ───────────────────────────────────────────────

def _price_resolve_T(args: argparse.Namespace, cfg: dict) -> tuple[float | None, str | None]:
    if getattr(args, "expiry_date", None):
        dc = _as_day_count(_resolve_day_count(getattr(args, "day_count", None), cfg))
        as_of_str = getattr(args, "as_of", None)
        as_of = date.fromisoformat(as_of_str) if as_of_str else date.today()
        exp = date.fromisoformat(args.expiry_date)
        T = dc.year_fraction(as_of, exp)
        if T <= 0:
            return None, f"expiry {exp} is not after as_of {as_of}"
        return T, None
    if getattr(args, "expiry", None) is not None:
        return float(args.expiry), None
    return None, "provide --expiry (years) or --expiry-date YYYY-MM-DD"


def _cmd_price(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.models.option import ImpliedVolInput, OptionContract, OptionType
    from arbfree_vol.pricing.black_scholes import price_floats
    from arbfree_vol.pricing.implied_vol import implied_vol

    T, err = _price_resolve_T(args, cfg)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    assert T is not None

    spot = float(args.spot)
    strike = float(args.strike)
    is_call = not bool(getattr(args, "put", False))
    r = float(getattr(args, "risk_free", None) if getattr(args, "risk_free", None) is not None else cfg.get("risk_free", 0.05))
    q = float(getattr(args, "div_yield", None) if getattr(args, "div_yield", None) is not None else cfg.get("div_yield", 0.0))

    if getattr(args, "implied_vol", None) is not None:
        vol = float(args.implied_vol)
        px = price_floats(spot, strike, T, r, q, vol, is_call)
        print(f"price={px:.6f}  (S={spot} K={strike} T={T:.6f} r={r} q={q} vol={vol} {'call' if is_call else 'put'})")
        return 0

    if getattr(args, "price", None) is not None:
        contract = OptionContract(
            symbol="CLI",
            option_type=OptionType.CALL if is_call else OptionType.PUT,
            strike=strike,
            expiry_date=date.today(),
        )
        iv_in = ImpliedVolInput(
            contract=contract,
            spot=spot,
            expiry_time=T,
            risk_free=r,
            div_yield=q,
            market_price=float(args.price),
        )
        iv = implied_vol(iv_in)
        if iv is None:
            print("implied vol: no root in bracket [1e-6, 5.0]", file=sys.stderr)
            return 1
        print(f"iv={iv:.6f}  (price={args.price} S={spot} K={strike} T={T:.6f} r={r} q={q} {'call' if is_call else 'put'})")
        return 0

    print("error: provide --vol (to price) or --price (to invert IV)", file=sys.stderr)
    return 2


# ── subcommand: fetch (live) ────────────────────────────────────────

def _cmd_fetch(args: argparse.Namespace, cfg: dict) -> int:
    try:
        from arbfree_vol.ingestion.yahoo import fetch_chain
        from arbfree_vol.repair.engine import repair
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    day_count = _resolve_day_count(getattr(args, "day_count", None), cfg)
    calendar_name = _resolve_calendar(getattr(args, "calendar", None), cfg)
    use_fred = bool(getattr(args, "use_fred_curve", False) or cfg.get("use_fred_curve"))

    print(f"[fetch] {args.symbol} max_expiries={args.max_expiries} day_count={day_count} calendar={calendar_name or 'none'} fred={use_fred}")
    try:
        surface, rejected, drops = fetch_chain(
            args.symbol,
            max_expiries=int(args.max_expiries),
            day_count=day_count,
            calendar=calendar_name,
            use_fred_curve=use_fred,
        )
    except Exception as exc:
        print(f"error fetching chain: {exc}", file=sys.stderr)
        return 1

    print(f"[fetch] spot={surface.spot} r={surface.risk_free:.4%} q={surface.div_yield:.4%} slices={len(surface.slices)} rejected={len(rejected)} quality_drops={len(drops)}")
    for sl in surface.slices:
        rf = sl.risk_free if sl.risk_free is not None else surface.risk_free
        print(f"  T={sl.expiry_time:.4f} r(T)={rf:.4%} n={len(sl.quotes)}")

    if getattr(args, "repair", False):
        report = repair(surface, use_ssvi=bool(getattr(args, "use_ssvi", False)), use_sabr=bool(getattr(args, "use_sabr", False)))
        m = report.metrics
        print(f"[repair] {m.n_violations_before}->{m.n_violations_after} fallback={report.fallback_slices} failed={report.failed_slices} infeasible={report.repair_infeasible}")
        if getattr(args, "output", None):
            Path(str(args.output)).write_text(
                json.dumps(
                    {
                        "spot": surface.spot,
                        "risk_free": surface.risk_free,
                        "div_yield": surface.div_yield,
                        "metrics": {
                            "n_rejected": m.n_rejected,
                            "n_total_quotes": m.n_total_quotes,
                            "n_slices_input": m.n_slices_input,
                            "n_slices_fitted": m.n_slices_fitted,
                            "n_violations_before": m.n_violations_before,
                            "n_violations_after": m.n_violations_after,
                        },
                        "fallback_slices": report.fallback_slices,
                        "failed_slices": report.failed_slices,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            print(f"[out] wrote {args.output}")

    return 0


# ── parser builders (kept small to keep CCN low) ───────────────────

def _add_repair_parser(sub) -> None:  # type: ignore[no-untyped-def]
    r = sub.add_parser("repair", help="clean + fit a chain CSV to an arb-free surface")
    r.add_argument("chain", help="path to chain CSV (strike,expiry,option_type,price[,bid,ask])")
    r.add_argument("--spot", required=True, type=float, help="underlying spot price")
    r.add_argument("--risk-free", type=float, default=None, dest="risk_free", help="flat risk-free rate (ignored when --use-fred-curve)")
    r.add_argument("--div-yield", type=float, default=None, dest="div_yield", help="dividend yield")
    r.add_argument("--as-of", type=str, default=None, help="valuation date YYYY-MM-DD (default: today)")
    r.add_argument("--day-count", type=str, default=None, help="ACT/365F (default), ACT/360, 30/360")
    r.add_argument("--calendar", type=str, default=None, help="USNYSE to roll expiries to business days, or none (default)")
    r.add_argument("--use-fred-curve", action="store_true", dest="use_fred_curve", help="pull Treasury+SOFR zero curve from FRED (offline falls back to flat)")
    r.add_argument("--fred-curve", action="store_true", help="alias for --use-fred-curve")
    r.add_argument("--offline", action="store_true", help="force offline (no FRED network)")
    r.add_argument("--use-ssvi", action="store_true", help="fit eSSVI (H&M hard constraints)")
    r.add_argument("--use-sabr", action="store_true", help="fit SABR term structure")
    r.add_argument("--output", "-o", type=str, default=None, help="write JSON report to file")
    r.add_argument("--plot", action="store_true", help="write diagnostic plots (requires matplotlib)")
    r.add_argument("--plot-dir", type=str, default=None, help="plot output dir (default: .)")


def _add_detect_parser(sub) -> None:  # type: ignore[no-untyped-def]
    d = sub.add_parser("detect", help="detect arbitrage violations in a chain CSV")
    d.add_argument("chain", help="path to chain CSV")
    d.add_argument("--spot", required=True, type=float)
    d.add_argument("--risk-free", type=float, default=None, dest="risk_free")
    d.add_argument("--div-yield", type=float, default=None, dest="div_yield")
    d.add_argument("--as-of", type=str, default=None)
    d.add_argument("--day-count", type=str, default=None)
    d.add_argument("--calendar", type=str, default=None)
    d.add_argument("--no-forward", action="store_true", help="disable forward-curve pre-pass (use raw r/q parity)")
    d.add_argument("--output", "-o", type=str, default=None)


def _add_price_parser(sub) -> None:  # type: ignore[no-untyped-def]
    pr = sub.add_parser("price", help="Black-Scholes price or implied vol")
    pr.add_argument("--spot", required=True, type=float)
    pr.add_argument("--strike", required=True, type=float)
    pr.add_argument("--expiry", type=float, default=None, help="time to expiry in years")
    pr.add_argument("--expiry-date", type=str, default=None, help="expiry date YYYY-MM-DD (uses --day-count and --as-of)")
    pr.add_argument("--as-of", type=str, default=None)
    pr.add_argument("--day-count", type=str, default=None)
    pr.add_argument("--vol", type=float, default=None, dest="implied_vol", help="vol to price at")
    pr.add_argument("--price", type=float, default=None, help="market price to invert to IV")
    pr.add_argument("--risk-free", type=float, default=None, dest="risk_free")
    pr.add_argument("--div-yield", type=float, default=None, dest="div_yield")
    pr.add_argument("--put", action="store_true", help="price as put (default: call)")


def _add_fetch_parser(sub) -> None:  # type: ignore[no-untyped-def]
    f = sub.add_parser("fetch", help="fetch a live chain via yfinance and optionally repair it")
    f.add_argument("--symbol", type=str, default="SPY")
    f.add_argument("--max-expiries", type=int, default=5)
    f.add_argument("--day-count", type=str, default=None)
    f.add_argument("--calendar", type=str, default=None)
    f.add_argument("--use-fred-curve", action="store_true", dest="use_fred_curve")
    f.add_argument("--repair", action="store_true", help="run repair pipeline after fetch")
    f.add_argument("--use-ssvi", action="store_true")
    f.add_argument("--use-sabr", action="store_true")
    f.add_argument("--output", "-o", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arbfree", description="Arbitrage-free vol surface toolkit")
    p.add_argument("--config", type=str, default=None, help="path to config YAML (default: config.yaml if present)")
    p.add_argument("--verbose", "-v", action="store_true", help="verbose output")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd", required=False)
    _add_repair_parser(sub)
    _add_detect_parser(sub)
    _add_price_parser(sub)
    _add_fetch_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(f"arbfree {VERSION}")
        return 0
    if args.cmd is None:
        parser.print_help()
        return 0
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO)

    cfg = _load_config_dict(getattr(args, "config", None))

    if args.cmd == "repair":
        return _cmd_repair(args, cfg)
    if args.cmd == "detect":
        return _cmd_detect(args, cfg)
    if args.cmd == "price":
        return _cmd_price(args, cfg)
    if args.cmd == "fetch":
        return _cmd_fetch(args, cfg)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
