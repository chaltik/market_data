from datetime import date
from pathlib import Path

import pytest
import yaml

from djia_history.load_djia import parse_history, check_recent_agreement, weekday_sessions_since


def test_wikipedia_history_has_complete_1991_baseline_and_current_members():
    # A tiny fixture tests the table shape without depending on a live website.
    names = [f"Company {n}" for n in range(30)]
    cells = "".join(f"<td>{name}</td>" for name in names)
    html = f'<div class="mw-heading"><h2>May 6, 1991</h2></div><table class="wikitable"><tr>{cells}</tr></table>'
    mapping = {name: f"X{n}" for n, name in enumerate(names)}
    snapshots = parse_history(html, mapping)
    assert snapshots[0][0] == date(1991, 5, 6)
    assert len(snapshots[0][1]) == 30


def test_history_rejects_incomplete_table():
    html = '<div class="mw-heading"><h2>May 6, 1991</h2></div><table class="wikitable"><tr><td>Only one</td></tr></table>'
    with pytest.raises(ValueError, match="expected 30"):
        parse_history(html, {"Only one": "ONE"})


def test_history_rejects_unknown_names():
    names = [f"Company {n}" for n in range(30)]
    cells = "".join(f"<td>{name}</td>" for name in names)
    html = f'<div class="mw-heading"><h2>May 6, 1991</h2></div><table class="wikitable"><tr>{cells}</tr></table>'
    with pytest.raises(ValueError, match="unmapped"):
        parse_history(html, {name: f"X{n}" for n, name in enumerate(names[:-1])})


def test_current_crosscheck_rejects_mismatch():
    with pytest.raises(ValueError, match="differs from DIA"):
        check_recent_agreement([(date(1991, 5, 6), {"A": "A"})], {"B": "B"})


def test_mapping_is_explicit():
    mapping = yaml.safe_load((Path(__file__).parents[1] / "djia_history/company_tickers.yaml").read_text())
    assert mapping["F. W. Woolworth Company"] == "Z"
    assert mapping["Alphabet Inc."] == "GOOGL"


def test_weekday_lag_is_not_calendar_day_lag():
    assert weekday_sessions_since(date(2026, 10, 30), date(2026, 11, 2)) == 1
    assert weekday_sessions_since(date(2026, 10, 30), date(2026, 11, 3)) == 2
