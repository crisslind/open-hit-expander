"""Chemistry-free checks of the prefilter helpers."""
from open_hit_expander.prefilter import DEFAULT_WINDOWS, _window_reasons


def test_window_reasons_only_report_violations():
    deltas = {"mw": 30.0, "clogp": 2.0, "tpsa": -50.0, "hbd": 0, "rotatable_bonds": 1}
    reasons = _window_reasons(deltas, DEFAULT_WINDOWS)
    assert any("Δclogp" in r and ">" in r for r in reasons)
    assert any("Δtpsa" in r and "<" in r for r in reasons)
    assert not any("Δmw" in r or "Δhbd" in r or "Δrotatable" in r for r in reasons)


def test_open_ended_windows_and_missing_values_are_ignored():
    assert _window_reasons({"mw": 500.0}, {"mw": (None, None)}) == []
    assert _window_reasons({"mw": float("nan")}, {"mw": (-1, 1)}) == []
    assert _window_reasons({}, {"mw": (-1, 1)}) == []
    assert _window_reasons({"mw": -2.0}, {"mw": (-1, None)}) == ["Δmw -2.00 < -1"]
