#!/usr/bin/env python3
"""Minimal stand-in for `boltz predict` used by the resume/cache tests.

Mimics the on-disk layout the package parses:
  <out>/boltz_results_<input dir name>/predictions/<job>/{confidence_*.json,<job>_model_0.cif,affinity_<job>.json}
Env: FAKE_BOLTZ_LOG (append processed job ids), FAKE_BOLTZ_FAIL_AFTER (crash after k jobs).
"""
import hashlib, json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == "predict", args
input_dir = Path(args[1])
out = Path(args[args.index("--out_dir") + 1])
seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
use_server = "--use_msa_server" in args
fail_after = int(os.environ.get("FAKE_BOLTZ_FAIL_AFTER", "-1"))
log = os.environ.get("FAKE_BOLTZ_LOG")
base = out / f"boltz_results_{input_dir.name}"

done = 0
for yaml_file in sorted(input_dir.glob("*.yaml")):
    if 0 <= fail_after <= done:
        sys.exit(3)
    job = yaml_file.stem
    text = yaml_file.read_text()
    if use_server and "msa:" not in text:
        msa = base / "msa"
        msa.mkdir(parents=True, exist_ok=True)
        (msa / f"{job}_0.csv").write_text("key,sequence\n-1,MKT\n")
    if log:
        with open(log, "a") as handle:
            handle.write(f"{seed}\t{job}\t{'server' if use_server else 'local'}\n")
    if "affinity" in text:
        pred = base / "predictions" / job
        pred.mkdir(parents=True, exist_ok=True)
        h = int(hashlib.sha256(f"{seed}{job}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
        (pred / f"confidence_{job}_model_0.json").write_text(json.dumps(
            {"confidence_score": 0.5 + h / 2, "ligand_iptm": 0.6, "iptm": 0.7, "ptm": 0.8}))
        (pred / f"{job}_model_0.cif").write_text("data_fake\n")
        (pred / f"affinity_{job}.json").write_text(json.dumps(
            {"affinity_pred_value": -1.0 - 2 * h, "affinity_probability_binary": 0.9}))
    done += 1
