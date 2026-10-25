"""Resume, cross-run cache and MSA-reuse tests using a fake `boltz` executable.

No GPU, Boltz or RDKit chemistry is exercised; these cover the bookkeeping.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from open_hit_expander.boltz2 import (
    make_boltz2_job, plan_boltz2_replicates, run_boltz2_replicates,
)
from open_hit_expander.jobcache import cache_key, ligand_key, prediction_settings

FAKE = str(Path(__file__).with_name("fake_boltz.py"))
SEQ = "MKTAYIAKQRQISFVKSHFSRQ"
LIGANDS = {"mol_1_parent": "CCO", "mol_1_analogue_0001": "CCN",
           "mol_1_analogue_0002": "CCC", "mol_1_analogue_0003": "CCF"}


def make_state(root: Path, use_server=True):
    run = root / "out" / "boltz2_cofold" / "20260101_000000_000000"
    jobs, results = run / "jobs", run / "results"
    jobs.mkdir(parents=True)
    results.mkdir()
    rows = []
    for job_id, smi in LIGANDS.items():
        (jobs / f"{job_id}.yaml").write_text(
            yaml.safe_dump(make_boltz2_job(SEQ, smi), sort_keys=False))
        rows.append({"job_id": job_id, "source_id": "mol_1", "ligand_smiles": smi,
                     "transformation": "parent" if "parent" in job_id else "x",
                     "is_parent": "parent" in job_id, "ligand_key": ligand_key(smi)})
    command = [FAKE, "predict", str(jobs), "--out_dir", str(results), "--diffusion_samples", "3",
               "--use_potentials"] + (["--use_msa_server"] if use_server else [])
    return {"root": run, "job_directory": jobs, "output_directory": results,
            "manifest": pd.DataFrame(rows), "command": command, "protein_sequence": SEQ}


def calls(log: Path):
    return [line.split("\t") for line in log.read_text().splitlines()] if log.exists() else []


def with_env(**env):
    class _Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in env}
            os.environ.update({k: str(v) for k, v in env.items()})
        def __exit__(self, *exc):
            for k, v in self.old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return _Ctx()


def test_resume_runs_only_unfinished_jobs(tmp_path):
    state, log = make_state(tmp_path, use_server=False), tmp_path / "calls.tsv"
    with with_env(FAKE_BOLTZ_LOG=log, FAKE_BOLTZ_FAIL_AFTER=2):
        _, status = run_boltz2_replicates(state, n_replicates=1, cache_dir=None, log=lambda *_: None)
    assert status.loc[0, "status"] == "partial" and status.loc[0, "n_success"] == 2
    first = calls(log)
    assert len(first) == 2

    with with_env(FAKE_BOLTZ_LOG=log):
        records, status = run_boltz2_replicates(state, n_replicates=1, cache_dir=None, log=lambda *_: None)
    assert status.loc[0, "status"] == "ok" and status.loc[0, "n_success"] == 4
    second = calls(log)[len(first):]
    assert len(second) == 2, "only the two unfinished jobs should be recomputed"
    assert {c[1] for c in first}.isdisjoint({c[1] for c in second})
    assert records["predicted_pic50"].notna().all()

    # A third call finds everything finished and launches nothing.
    before = len(calls(log))
    with with_env(FAKE_BOLTZ_LOG=log):
        run_boltz2_replicates(state, n_replicates=1, cache_dir=None, log=lambda *_: None)
    assert len(calls(log)) == before


def test_cross_run_cache_restores_with_new_job_ids(tmp_path):
    log = tmp_path / "calls.tsv"
    state = make_state(tmp_path, use_server=False)
    with with_env(FAKE_BOLTZ_LOG=log):
        run_boltz2_replicates(state, n_replicates=1, cache_dir=tmp_path / "cache", log=lambda *_: None)
    assert len(calls(log)) == 4

    # New run in which the same molecules carry DIFFERENT job ids (re-enumeration).
    other = tmp_path / "second"
    state2 = make_state(other, use_server=False)
    renamed = {k: k.replace("analogue_000", "analogue_010") for k in LIGANDS}
    for old, new in renamed.items():
        (state2["job_directory"] / f"{old}.yaml").rename(state2["job_directory"] / f"{new}.yaml")
    state2["manifest"]["job_id"] = state2["manifest"]["job_id"].map(renamed)
    with with_env(FAKE_BOLTZ_LOG=log):
        records, status = run_boltz2_replicates(state2, n_replicates=1, cache_dir=tmp_path / "cache",
                                                log=lambda *_: None)
    assert len(calls(log)) == 4, "no new cofolding should have run"
    assert status.loc[0, "n_cache"] == 4 and status.loc[0, "n_computed"] == 0
    assert records["predicted_pic50"].notna().all()
    assert set(records["job_id"]) == set(renamed.values())


def test_plan_reports_pending_without_running(tmp_path):
    state, log = make_state(tmp_path, use_server=False), tmp_path / "calls.tsv"
    plan = plan_boltz2_replicates(state, n_replicates=2, cache_dir=tmp_path / "cache")
    assert list(plan["to_compute"]) == [4, 4] and not log.exists()


def test_msa_is_generated_once_and_reused_by_every_replicate(tmp_path):
    state, log = make_state(tmp_path, use_server=True), tmp_path / "calls.tsv"
    with with_env(FAKE_BOLTZ_LOG=log):
        run_boltz2_replicates(state, n_replicates=2, cache_dir=tmp_path / "cache", log=lambda *_: None)
    modes = [c[2] for c in calls(log)]
    assert modes.count("server") == 1, "exactly one server call (the one-off MSA bootstrap)"
    assert modes.count("local") == 8, "all 2 x 4 real jobs ran against the cached MSA"
    msas = list((tmp_path / "cache" / "msa_cache" / "msa").glob("*.csv"))
    assert len(msas) == 1
    for rep in (tmp_path / "out" / "boltz2_cofold" / "20260101_000000_000000" / "replicates").glob("*/jobs/*.yaml"):
        assert yaml.safe_load(rep.read_text())["sequences"][0]["protein"]["msa"] == str(msas[0].resolve())


def test_cache_key_ignores_hardware_flags_but_not_model_settings():
    base = [FAKE, "predict", "/a", "--out_dir", "/b", "--diffusion_samples", "3", "--accelerator", "cpu"]
    moved = [FAKE, "predict", "/x", "--out_dir", "/y", "--diffusion_samples", "3", "--accelerator", "gpu",
             "--no_kernels"]
    more = [FAKE, "predict", "/a", "--out_dir", "/b", "--diffusion_samples", "5"]
    assert prediction_settings(base) == prediction_settings(moved)
    assert cache_key(SEQ, "CCO", 1, base) == cache_key(SEQ, "CCO", 1, moved)
    assert cache_key(SEQ, "CCO", 1, base) != cache_key(SEQ, "CCO", 2, base)
    assert cache_key(SEQ, "CCO", 1, base) != cache_key(SEQ, "CCO", 1, more)
