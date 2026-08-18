"""CLI entry point — ``arbfree``.

Subcommands (Milestone 1, now wired to the real pipeline)::

    arbfree repair chain.csv --spot 450 [--use-ssvi|--use-sabr] [--plot]
    arbfree detect chain.csv --spot 450 [--forward|--no-forward]
    arbfree price --spot 100 --strike 100 --expiry 0.25 --vol 0.2
    arbfree fetch --symbol SPY [--use-fred-curve] [--day-count ACT/365F]

``repair`` and ``detect`` thread the new ``DayCount``/``Calendar`` and
``YieldTermStructure`` (FRED) options through to the ingestion layer:
``--day-count`` selects the year-fraction (default ``ACT/365F`` =
``days/365.0``, fixtures byte-identical), ``--calendar USNYSE`` rolls
expiries to the next business day, ``--use-fred-curve`` pulls a
Treasury+SOFR zero curve from FRED with flat fallback.

Config file (``config.yaml``) is optional: CLI flags override file
values, which override built-in defaults.  Requires ``pyyaml`` only
when a config file is actually used — no hard dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
import logging
from datetime import date
from pathlib import Path

from arbfree_vol.time import DayCount, Calendar
from arbfree_vol.rates import YieldTermStructure, build_fred_curve

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# ── config helper ─────────────────────────────────────────────────────

def _load_config_file(path: Path | None) -> dict:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(Path("config.yaml"))
        candidates.append(Path("arbfree.yaml"))
    for p in candidates:
        if p.exists():
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError:
                logger.warning("config file %s found but pyyaml not installed — ignoring", p)
                return {}
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                logger.warning("failed to parse config %s: %s", p, exc)
    return {}


def _resolve_day_count(value: str | None, cfg: dict) -> str:
    if value is not None:
        return value
    return str(cfg.get("day_count", cfg.get("dayCount", "ACT/365F")))


def _resolve_calendar(value: str | None, cfg: dict) -> str | None:
    if value is not None:
        # explicit --calendar none disables
        if value.lower() in ("none", "off", "no"):
            return None
        return value
    v = cfg.get("calendar")
    if v is None or str(v).lower() in ("none", "off", "no"):
        return None
    return str(v)


# ── subcommand: repair ────────────────────────────────────────────────

def _cmd_repair(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.ingestion.loader import load_chain_csv
    from arbfree_vol.repair.engine import repair

    csv_path = Path(args.chain)
    if not csv_path.exists():
        print(f"error: chain file not found: {csv_path}", file=sys.stderr)
        return 2

    day_count = _resolve_day_count(args.day_count, cfg)
    calendar_name = _resolve_calendar(args.calendar, cfg)
    calendar = Calendar(calendar_name) if calendar_name else None

    # risk-free: flat float, or FRED curve
    use_fred = bool(args.use_fred_curve or cfg.get("use_fred_curve"))
    risk_free_arg: float | YieldTermStructure
    if use_fred or args.fred_curve:
        as_of = None
        if args.as_of:
            as_of = date.fromisoformat(args.as_of)
        curve = build_fred_curve(as_of=as_of, day_count=day_count, offline=bool(args.offline))
        risk_free_arg = curve
        print(f"[rates] FRED curve: {curve}")
    else:
        rf = args.risk_free if args.risk_free is not None else float(cfg.get("risk_free", cfg.get("riskFree", 0.05)))
        risk_free_arg = float(rf)

    div_yield = float(args.div_yield if args.div_yield is not None else cfg.get("div_yield", cfg.get("divYield", 0.0)))
    as_of_date = date.fromisoformat(args.as_of) if args.as_of else None

    try:
        dc = DayCount(day_count)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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

    print(f"[clean] kept {sum(len(sl.quotes) for sl in surface.slices)} quotes in {len(surface.slices)} slices; rejected {len(rejected)}")
    if rejected and args.verbose:
        for r in rejected[:20]:
            print(f"  reject {r.rule.value}: {r.detail}")

    if args.use_ssvi and args.use_sabr:
        print("error: --use-ssvi and --use-sabr are mutually exclusive", file=sys.stderr)
        return 2

    report = repair(surface, use_ssvi=bool(args.use_ssvi), use_sabr=bool(args.use_sabr))
    m = report.metrics
    print(f"[repair] rejected={m.n_rejected} total={m.n_total_quotes} slices {m.n_slices_fitted}/{m.n_slices_input} "
          f"violations {m.n_violations_before}->{m.n_violations_after} "
          f"fallback={report.fallback_slices} failed={report.failed_slices} infeasible={report.repair_infeasible}")

    if report.remaining_violations.violations and args.verbose:
        for v in report.remaining_violations.violations[:20]:
            print(f"  remaining {v.kind.value}: {v.detail}")

    # write JSON report if requested
    out = args.output or cfg.get("output")
    if out:
        out_path = Path(out)
        payload = {
            "metrics": {
                "n_rejected": m.n_rejected,
                "n_total_quotes": m.n_total_quotes,
                "n_slices_input": m.n_slices_input,
                "n_slices_fitted": m.n_slices_fitted,
                "n_violations_before": m.n_violations_before,
                "n_violations_after": m.n_violations_after,
                "rejection_rate": m.rejection_rate,
            },
            "fallback_slices": report.fallback_slices,
            "failed_slices": report.failed_slices,
            "repair_infeasible": report.repair_infeasible,
            "remaining_violations": [
                {"kind": v.kind.value, "detail": v.detail, "magnitude": v.magnitude}
                for v in report.remaining_violations.violations
            ],
            "fitted_slices": [
                {"expiry": fs.expiry_time, "forward": fs.forward_price, "params": fs.params.model_dump() if hasattr(fs.params, "model_dump") else str(fs.params)}
                for fs in report.fitted_slices
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[out] wrote {out_path}")

    if args.plot:
        try:
            plot_dir = Path(args.plot_dir or ".")
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[plot] (surface plots would write to {plot_dir}) — use demo scripts for full 7-plot output")
        except Exception as exc:
            print(f"[plot] skipped: {exc}", file=sys.stderr)

    return 0 if not report.repair_infeasible else 0  # infeasible is reportable, not an error exit


# ── subcommand: detect ────────────────────────────────────────────────

def _cmd_detect(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.ingestion.loader import load_chain_csv
    from arbfree_vol.arbitrage.quote_detect import detect, detect_with_forward

    csv_path = Path(args.chain)
    if not csv_path.exists():
        print(f"error: chain file not found: {csv_path}", file=sys.stderr)
        return 2

    day_count = _resolve_day_count(args.day_count, cfg)
    calendar_name = _resolve_calendar(args.calendar, cfg)
    calendar = Calendar(calendar_name) if calendar_name else None
    try:
        dc = DayCount(day_count)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    as_of_date = date.fromisoformat(args.as_of) if args.as_of else None
    # for detect we keep raw quotes (clean=False) so violations are visible
    try:
        surface, _ = load_chain_csv(
            csv_path,
            spot=float(args.spot),
            risk_free=float(args.risk_free if args.risk_free is not None else cfg.get("risk_free", 0.05)),
            div_yield=float(args.div_yield if args.div_yield is not None else cfg.get("div_yield", 0.0)),
            as_of=as_of_date,
            clean=False,
            day_count=dc,
            calendar=calendar,
        )
    except Exception as exc:
        print(f"error loading chain: {exc}", file=sys.stderr)
        return 1

    use_forward = not args.no_forward
    report = detect_with_forward(surface) if use_forward else detect(surface)
    if not report.violations:
        print("no violations detected")
        return 0
    # group by kind
    from collections import Counter
    counts = Counter(v.kind.value for v in report.violations)
    print(f"violations: {len(report.violations)}  by kind: {dict(counts)}  (forward={'on' if use_forward else 'off'})")
    for v in report.violations:
        print(f"  [{v.kind.value}] {v.detail}  mag={v.magnitude:.4g}")
    out = args.output or cfg.get("output")
    if out:
        Path(out).write_text(json.dumps(
            [{"kind": v.kind.value, "detail": v.detail, "magnitude": v.magnitude} for v in report.violations],
            indent=2,
        ), encoding="utf-8")
        print(f"[out] wrote {out}")

    return 0


# ── subcommand: price ─────────────────────────────────────────────────

def _cmd_price(args: argparse.Namespace, cfg: dict) -> int:
    from arbfree_vol.pricing.black_scholes import price_floats
    from arbfree_vol.pricing.implied_vol import implied_vol
    from arbfree_vol.models.option import OptionType, OptionContract, ImpliedVolInput
    from datetime import date as date_cls

    # expiry can be T (years) or date string when day_count+as_of provided
    T: float
    if args.expiry_date:
        dc = DayCount(_resolve_day_count(args.day_count, cfg))
        as_of = date_cls.fromisoformat(args.as_of) if args.as_of else date_cls.today()
        exp = date_cls.fromisoformat(args.expiry_date)
        T = dc.year_fraction(as_of, exp)
        if T <= 0:
            print(f"error: expiry {exp} is not after as_of {as_of}", file=sys.stderr)
            return 2
    elif args.expiry is not None:
        T = float(args.expiry)
    else:
        print("error: provide --expiry (years) or --expiry-date YYYY-MM-DD", file=sys.stderr)
        return 2

    spot = float(args.spot)
    strike = float(args.strike)
    is_call = not args.put
    r = float(args.risk_free if args.risk_free is not None else cfg.get("risk_free", 0.05))
    q = float(args.div_yield if args.div_yield is not None else cfg.get("div_yield", 0.0))

    if args.implied_vol is not None:
        # price from vol
        vol = float(args.implied_vol)
        px = price_floats(spot, strike, T, r, q, vol, is_call)
        print(f"price={px:.6f}  (S={spot} K={strike} T={T:.6f} r={r} q={q} vol={vol} {'call' if is_call else 'put'})")
        return 0

    if args.price is not None:
        # implied vol from price
        # ImpliedVolInput requires contract with expiry_date — synthesize
        exp_date = date_cls.today()
        contract = OptionContract(
            symbol="CLI",
            option_type=OptionType.CALL if is_call else OptionType.PUT,
            strike=strike,
            expiry_date=exp_date,
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

    # default: need --vol or --price
    print("error: provide --vol (to price) or --price (to invert IV)", file=sys.stderr)
    return 2


# ── subcommand: fetch (live) ──────────────────────────────────────────

def _cmd_fetch(args: argparse.Namespace, cfg: dict) -> int:
    # live fetch via yfinance, then optional repair
    try:
        from arbfree_vol.ingestion.yahoo import fetch_chain
        from arbfree_vol.repair.engine import repair
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    day_count = _resolve_day_count(args.day_count, cfg)
    calendar_name = _resolve_calendar(args.calendar, cfg)
    use_fred = bool(args.use_fred_curve or cfg.get("use_fred_curve"))

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

    if args.repair:
        report = repair(surface, use_ssvi=bool(args.use_ssvi), use_sabr=bool(args.use_sabr))
        m = report.metrics
        print(f"[repair] {m.n_violations_before}->{m.n_violations_after} fallback={report.fallback_slices} failed={report.failed_slices} infeasible={report.repair_infeasible}")
        if args.output:
            Path(args.output).write_text(json.dumps({
                "spot": surface.spot,
                "risk_free": surface.risk_free,
                "div_yield": surface.div_yield,
                "metrics": m.__dict__ if hasattr(m, "__dict__") else str(m),
                "fallback_slices": report.fallback_slices,
                "failed_slices": report.failed_slices,
            }, indent=2, default=str), encoding="utf-8")
            print(f"[out] wrote {args.output}")

    return 0


# ── parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arbfree", description="Arbitrage-free vol surface toolkit")
    p.add_argument("--config", type=str, default=None, help="path to config YAML (default: config.yaml if present)")
    p.add_argument("--verbose", "-v", action="store_true", help="verbose output")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd", required=False)

    # repair
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

    # detect
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

    # price
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

    # fetch (live)
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

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    cfg = _load_config_file(getattr(args, "config", None))

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
