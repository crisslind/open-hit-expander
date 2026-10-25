"""End-to-end prefilter checks on a tiny parent. Requires RDKit."""
import pytest

pytest.importorskip("rdkit")

from open_hit_expander import HitExpanderConfig, run_hit_expander
from open_hit_expander.prefilter import PrefilterConfig, apply_prefilter

PARENT = "CC(=O)Nc1ccc(O)cc1"  # paracetamol: four aromatic C-H positions


def _run():
    config = HitExpanderConfig(
        additions=("Me", "F", "Cl", "Br"), reject_alert_catalogs=(), min_morgan_tanimoto=0.0)
    return run_hit_expander(PARENT, config)


def test_prefilter_does_not_modify_its_input():
    run = _run()
    before = len(run["passed"])
    apply_prefilter(run, PrefilterConfig(windows={"mw": (-10, 20)}))
    assert len(run["passed"]) == before


def test_relative_window_drops_heavy_analogues_and_logs_why():
    out = apply_prefilter(_run(), PrefilterConfig(windows={"mw": (-10, 40)}))
    assert not out["passed"]["analog_smiles"].str.contains("Br").any()   # +79 Da
    assert out["passed"]["analog_smiles"].str.contains("Cl").any()       # +34 Da stays
    dropped = out["prefilter_log"].query("stage == 'windows' and action == 'drop'")
    assert len(dropped) > 0 and dropped["reason"].str.contains("Δmw").all()


def test_forbidden_and_required_smarts():
    out = apply_prefilter(_run(), PrefilterConfig(windows={}, forbid_smarts=("[F]",)))
    assert not out["passed"]["analog_smiles"].str.contains("F").any()
    none_left = apply_prefilter(_run(), PrefilterConfig(windows={}, require_smarts=("[N+](=O)[O-]",)))
    assert none_left["passed"].empty


def test_core_given_as_parent_atom_indices_is_retained():
    # atoms 0-3 are the acetamide: CH3, C, O, N
    out = apply_prefilter(_run(), PrefilterConfig(windows={}, core_atoms={"mol_1": [0, 1, 2, 3]}))
    assert len(out["passed"]) == len(_run()["passed"])


def test_cap_per_parent_and_funnel():
    out = apply_prefilter(_run(), PrefilterConfig(windows={}, max_candidates_per_parent=2))
    assert len(out["passed"]) == 2
    assert out["prefilter_funnel"]["remaining"].iloc[-1] == 2
    assert (out["prefilter_log"]["stage"] == "selection").any()
