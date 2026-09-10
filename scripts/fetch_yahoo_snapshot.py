"""Fetch a new Yahoo option snapshot for later offline study."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument("--risk-free", type=float, required=True)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--max-expiries", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError as error:
        raise SystemExit('Install the collector with: pip install -e ".[yahoo]"') from error

    ticker = yf.Ticker(args.symbol)
    history = ticker.history(period="5d")
    if history.empty:
        raise SystemExit(f"No spot history returned for {args.symbol}")
    spot = float(history["Close"].dropna().iloc[-1])
    as_of = date.today()
    rows: list[dict[str, object]] = []
    for expiry in ticker.options[: args.max_expiries]:
        chain = ticker.option_chain(expiry)
        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            for record in frame.to_dict("records"):
                bid = record.get("bid")
                ask = record.get("ask")
                if bid is None or ask is None:
                    continue
                rows.append(
                    {
                        "expiry": expiry,
                        "strike": record["strike"],
                        "option_type": option_type,
                        "price": (float(bid) + float(ask)) / 2,
                        "bid": bid,
                        "ask": ask,
                    }
                )
    if not rows:
        raise SystemExit(f"No option quotes returned for {args.symbol}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "symbol": args.symbol,
        "source": "Yahoo Finance via yfinance",
        "as_of": as_of.isoformat(),
        "spot": spot,
        "risk_free": args.risk_free,
        "dividend_yield": args.dividend_yield,
        "day_count": "ACT/365F",
        "notes": "Rate and dividend yield supplied by the collector user.",
    }
    metadata_path = args.output.with_name(f"{args.output.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} quotes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
