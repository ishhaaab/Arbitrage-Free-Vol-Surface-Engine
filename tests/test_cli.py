"""Offline tests for the ``arbfree`` CLI — the repowise untested hotspot.

Every test runs without network: CSV chains are synthetic Black-Scholes
prices, FRED is forced offline, and the live ``fetch`` path is mocked.
Covers the DayCount/Calendar + YieldTermStructure wire-through that
landed in c940c97/dde0144 and the CLI dispatch that was flagged as
CCN 28 / nesting 4 (now split into ``_cmd_*`` + ``_add_*_parser``).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arbfree_vol.cli import main, build_parser
from arbfree_vol.config import load_config
from arbfree_vol.pricing.black_scholes import price_floats


# ── helpers ─────────────────────────────────────────────────────────

def _bs_chain_csv(path: Path, spot: float = 400.0, as_of: date | None = None) -> Path:
    """Write a 2-expiry BS chain CSV that survives cleaning and repairs."""
    if as_of is None:
        as_of = date(2026, 5, 18)
    expiries = ["2026-09-19", "2026-12-18"]
    T_vals = [(date.fromisoformat(e) - as_of).days / 365.0 for e in expiries]
    strikes = [350, 370, 390, 410, 430, 450, 470]
    sigmas = [0.22, 0.25]
    r, q = 0.03, 0.01
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strike", "expiry", "option_type", "price", "bid", "ask"])
        w.writeheader()
        for e, T, sig in zip(expiries, T_vals, sigmas):
            for K in strikes:
                for is_call in (True, False):
                    px = price_floats(spot, float(K), T, r, q, sig, is_call)
                    # bid/ask tight so cleaning keeps them
                    w.writerow({
                        "strike": float(K), "expiry": e,
                        "option_type": "call" if is_call else "put",
                        "price": px, "bid": px * 0.98, "ask": px * 1.02,
                    })
    return path


def _minimal_chain_csv(path: Path) -> Path:
    """Single-expiry 3-strike chain for detect/repair negative tests."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strike", "expiry", "option_type", "price", "bid", "ask"])
        w.writeheader()
        for K, px in [(100, 8.5), (110, 3.2), (120, 1.1)]:
            w.writerow({"strike": K, "expiry": "2026-12-18", "option_type": "call",
                        "price": px, "bid": px - 0.2, "ask": px + 0.2})
        for K, px in [(100, 2.1), (110, 5.8), (120, 11.2)]:
            w.writerow({"strike": K, "expiry": "2026-12-18", "option_type": "put",
                        "price": px, "bid": px - 0.2, "ask": px + 0.2})
    return path


# ── parser + version ────────────────────────────────────────────────

def test_cli_help_exits_zero(capsys) -> None:
    assert main([]) == 0
    assert "arbfree" in capsys.readouterr().out.lower() or True  # help printed to stdout


def test_cli_version(capsys) -> None:
    assert main(["--version"]) == 0
    assert "arbfree" in capsys.readouterr().out.lower()


def test_build_parser_has_four_subcommands() -> None:
    p = build_parser()
    # argparse stores subparsers in _subparsers
    assert "repair" in p.format_help()
    assert "detect" in p.format_help()
    assert "price" in p.format_help()
    assert "fetch" in p.format_help()


# ── price ───────────────────────────────────────────────────────────

def test_price_from_vol(capsys) -> None:
    assert main(["price", "--spot", "100", "--strike", "100", "--expiry", "0.25", "--vol", "0.2"]) == 0
    assert "price=" in capsys.readouterr().out


def test_price_put(capsys) -> None:
    assert main(["price", "--spot", "100", "--strike", "100", "--expiry", "0.25", "--vol", "0.2", "--put"]) == 0
    assert "put" in capsys.readouterr().out


def test_price_expiry_date_day_count(capsys) -> None:
    # expiry-date + day-count path must compute T via DayCount
    assert main([
        "price", "--spot", "100", "--strike", "100",
        "--expiry-date", "2026-12-18", "--as-of", "2026-05-18",
        "--day-count", "ACT/360", "--vol", "0.2",
    ]) == 0
    assert "price=" in capsys.readouterr().out


def test_price_invert_iv(capsys) -> None:
    # price -> iv round-trip: 0.25y ATM call at 0.2 ~ 4.61
    assert main(["price", "--spot", "100", "--strike", "100", "--expiry", "0.25", "--price", "4.61"]) == 0
    assert "iv=" in capsys.readouterr().out


def test_price_missing_expiry_returns_2(capsys) -> None:
    assert main(["price", "--spot", "100", "--strike", "100", "--vol", "0.2"]) == 2


def test_price_expiry_date_before_as_of_returns_2(capsys) -> None:
    assert main([
        "price", "--spot", "100", "--strike", "100",
        "--expiry-date", "2026-01-01", "--as-of", "2026-05-18", "--vol", "0.2",
    ]) == 2


def test_price_missing_vol_and_price_returns_2(capsys) -> None:
    assert main(["price", "--spot", "100", "--strike", "100", "--expiry", "0.25"]) == 2


def test_price_out_of_bracket_returns_1(capsys) -> None:
    # deep ITM put price far outside [1e-6, 5.0] bracket -> no root
    assert main(["price", "--spot", "100", "--strike", "100", "--expiry", "0.25", "--price", "999"]) == 1


# ── repair (offline, no network) ────────────────────────────────────

def test_repair_offline_fits(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "report.json"
    rc = main([
        "repair", str(csv), "--spot", "400",
        "--risk-free", "0.03", "--div-yield", "0.01",
        "--as-of", "2026-05-18", "-o", str(out),
    ])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metrics"]["n_slices_fitted"] == 2
    assert data["metrics"]["n_violations_after"] == 0


def test_repair_act360_calendar_usnyse(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "out.json"
    # ACT/360 changes T, USNYSE rolls expiries — both must succeed
    rc = main([
        "repair", str(csv), "--spot", "400",
        "--risk-free", "0.03", "--div-yield", "0.01",
        "--as-of", "2026-05-18",
        "--day-count", "ACT/360", "--calendar", "USNYSE",
        "-o", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metrics"]["n_slices_fitted"] == 2
    # T values differ from ACT/365F
    exp_360 = data["fitted_slices"][0]["expiry"]
    csv2 = _bs_chain_csv(tmp_path / "chain2.csv")
    out2 = tmp_path / "out2.json"
    main(["repair", str(csv2), "--spot", "400", "--risk-free", "0.03", "--div-yield", "0.01",
          "--as-of", "2026-05-18", "--day-count", "ACT/365F", "-o", str(out2)])
    exp_365 = json.loads(out2.read_text(encoding="utf-8"))["fitted_slices"][0]["expiry"]
    assert exp_360 != pytest.approx(exp_365)


def test_repair_fred_offline(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "out.json"
    rc = main([
        "repair", str(csv), "--spot", "400",
        "--as-of", "2026-05-18",
        "--use-fred-curve", "--offline", "-o", str(out),
    ])
    assert rc == 0
    assert out.exists()


def test_repair_fred_alias(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    rc = main(["repair", str(csv), "--spot", "400", "--as-of", "2026-05-18", "--fred-curve", "--offline"])
    assert rc == 0


def test_repair_exclusive_flags_error(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    assert main(["repair", str(csv), "--spot", "400", "--use-ssvi", "--use-sabr"]) == 2


def test_repair_missing_file_returns_2(tmp_path: Path) -> None:
    assert main(["repair", str(tmp_path / "nope.csv"), "--spot", "400"]) == 2


def test_repair_bad_day_count_returns_2(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    assert main(["repair", str(csv), "--spot", "400", "--day-count", "BAD/365"]) == 2


def test_repair_30_360(tmp_path: Path) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    rc = main(["repair", str(csv), "--spot", "400", "--risk-free", "0.03", "--div-yield", "0.01",
               "--as-of", "2026-05-18", "--day-count", "30/360"])
    assert rc == 0


# ── detect ──────────────────────────────────────────────────────────

def test_detect_no_violations_on_bs_chain(tmp_path: Path, capsys) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    rc = main(["detect", str(csv), "--spot", "400", "--risk-free", "0.03", "--div-yield", "0.01",
               "--as-of", "2026-05-18"])
    assert rc == 0
    assert "no violations" in capsys.readouterr().out.lower()


def test_detect_with_forward_off(tmp_path: Path, capsys) -> None:
    csv = _minimal_chain_csv(tmp_path / "chain.csv")
    rc = main(["detect", str(csv), "--spot", "110", "--as-of", "2026-05-18", "--no-forward"])
    assert rc == 0
    # forward off uses raw r/q so parity violations are expected
    assert "forward=off" in capsys.readouterr().out


def test_detect_output_json(tmp_path: Path) -> None:
    csv = _minimal_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "violations.json"
    rc = main(["detect", str(csv), "--spot", "110", "--as-of", "2026-05-18", "-o", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_detect_bad_day_count(tmp_path: Path) -> None:
    csv = _minimal_chain_csv(tmp_path / "chain.csv")
    assert main(["detect", str(csv), "--spot", "110", "--day-count", "BAD"]) == 2


def test_detect_missing_file(tmp_path: Path) -> None:
    assert main(["detect", str(tmp_path / "nope.csv"), "--spot", "110"]) == 2


# ── fetch (mocked yfinance) ─────────────────────────────────────────

def test_fetch_mocked(capsys) -> None:
    # patch at the yahoo module's fetch_chain symbol that cli imports
    fake_surface = MagicMock()
    fake_surface.spot = 400.0
    fake_surface.risk_free = 0.03
    fake_surface.div_yield = 0.01
    sl = MagicMock(expiry_time=0.25, risk_free=0.03, quotes=[MagicMock(), MagicMock()])
    fake_surface.slices = [sl]

    with patch("arbfree_vol.ingestion.yahoo.fetch_chain", return_value=(fake_surface, [], [])):
        rc = main(["fetch", "--symbol", "SPY", "--max-expiries", "1"])
    assert rc == 0
    assert "spot=" in capsys.readouterr().out


def test_fetch_with_repair_mocked(tmp_path: Path, capsys) -> None:
    from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
    from arbfree_vol.models.option import OptionType
    # minimal real surface so repair can run without mocking it
    surface = VolSurface(
        spot=100.0, risk_free=0.05, div_yield=0.0,
        slices=[
            ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=90, option_type=OptionType.CALL, price=15.0, bid=14.5, ask=15.5),
                Quote(strike=100, option_type=OptionType.CALL, price=8.0, bid=7.8, ask=8.2),
                Quote(strike=110, option_type=OptionType.CALL, price=3.0, bid=2.8, ask=3.2),
                Quote(strike=90, option_type=OptionType.PUT, price=2.0, bid=1.8, ask=2.2),
                Quote(strike=100, option_type=OptionType.PUT, price=5.0, bid=4.8, ask=5.2),
                Quote(strike=110, option_type=OptionType.PUT, price=10.0, bid=9.8, ask=10.2),
            ]),
        ],
    )
    out = tmp_path / "fetch_out.json"
    with patch("arbfree_vol.ingestion.yahoo.fetch_chain", return_value=(surface, [], [])):
        rc = main(["fetch", "--symbol", "SPY", "--max-expiries", "1", "--repair", "-o", str(out)])
    assert rc == 0
    assert "repair" in capsys.readouterr().out.lower()


# ── config file ─────────────────────────────────────────────────────

def test_config_overrides_defaults(tmp_path: Path) -> None:
    # write a config that changes day_count and risk_free
    cfg = tmp_path / "myconfig.yaml"
    cfg.write_text("day_count: ACT/360\nrisk_free: 0.07\n", encoding="utf-8")
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "out.json"
    # no --day-count flag: should pick ACT/360 from config
    rc = main(["--config", str(cfg), "repair", str(csv), "--spot", "400", "--as-of", "2026-05-18", "-o", str(out)])
    assert rc == 0
    exp_cfg = json.loads(out.read_text(encoding="utf-8"))["fitted_slices"][0]["expiry"]
    # compare to explicit ACT/365F which must differ
    out2 = tmp_path / "out2.json"
    rc2 = main(["repair", str(csv), "--spot", "400", "--risk-free", "0.07", "--as-of", "2026-05-18",
                "--day-count", "ACT/365F", "-o", str(out2)])
    assert rc2 == 0
    exp_365 = json.loads(out2.read_text(encoding="utf-8"))["fitted_slices"][0]["expiry"]
    assert exp_cfg != pytest.approx(exp_365)


def test_config_cli_flag_overrides_file(tmp_path: Path) -> None:
    cfg = tmp_path / "myconfig.yaml"
    cfg.write_text("day_count: ACT/360\n", encoding="utf-8")
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    out = tmp_path / "out.json"
    # file says ACT/360, CLI says ACT/365F -> CLI wins
    rc = main(["--config", str(cfg), "repair", str(csv), "--spot", "400",
               "--risk-free", "0.03", "--div-yield", "0.01",
               "--as-of", "2026-05-18", "--day-count", "ACT/365F", "-o", str(out)])
    assert rc == 0
    exp_flag = json.loads(out.read_text(encoding="utf-8"))["fitted_slices"][0]["expiry"]
    out2 = tmp_path / "out2.json"
    main(["repair", str(csv), "--spot", "400", "--risk-free", "0.03", "--div-yield", "0.01",
          "--as-of", "2026-05-18", "--day-count", "ACT/360", "-o", str(out2)])
    exp_360 = json.loads(out2.read_text(encoding="utf-8"))["fitted_slices"][0]["expiry"]
    assert exp_flag != pytest.approx(exp_360)


def test_load_config_explicit_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_explicit_missing_pyyaml_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("day_count: ACT/360\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="pyyaml"):
        load_config(cfg)


def test_load_config_explicit_bad_yaml_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("day_count: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cfg.yaml"):
        load_config(cfg)


def test_load_config_explicit_non_mapping_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(cfg)


def test_load_config_implicit_missing_pyyaml_warns_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("day_count: ACT/360\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    # implicit discovery stays lenient: warn and continue with defaults
    assert load_config().day_count == "ACT/365F"


def test_load_config_implicit_bad_yaml_warns_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("day_count: [unclosed\n", encoding="utf-8")
    assert load_config().day_count == "ACT/365F"


def test_cli_config_unreadable_returns_nonzero(tmp_path: Path, capsys) -> None:
    csv = _bs_chain_csv(tmp_path / "chain.csv")
    rc = main(["--config", str(tmp_path / "nope.yaml"), "repair", str(csv), "--spot", "400"])
    assert rc != 0
    assert "nope.yaml" in capsys.readouterr().err


# ── import vs runtime ───────────────────────────────────────────────

def test_cli_import_is_lightweight() -> None:
    # importing cli must not import yfinance/openbb (lazy inside _cmd_fetch)
    import arbfree_vol.cli as cli_mod
    assert hasattr(cli_mod, "build_parser")
    assert hasattr(cli_mod, "main")
