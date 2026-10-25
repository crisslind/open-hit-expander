"""Compute the target MSA once per protein sequence and reuse it everywhere.

Without this, a batch prepared without ``MSA_PATH`` carries ``--use_msa_server``
and every replicate (a fresh ``boltz predict`` process) asks the server again.

Strategy: run ONE structure-only job with ``--use_msa_server`` into a scratch
directory, keep the MSA file Boltz writes, and store it under
``<cache_root>/msa/<sequence key>.<ext>``. Later replicates and later runs point
their YAML files at that file and drop ``--use_msa_server``.

The location Boltz writes MSAs to (``.../msa/<job>_0.csv`` or ``.a3m``) was
written from its documented behaviour and is searched defensively; if no file is
found, :func:`ensure_target_msa` returns None and the caller falls back to the
server. Verify once on your machine (see the log it prints).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from .jobcache import protein_sequence_from_job, sequence_key
from .stability import _replace_command

MSA_SUFFIXES = (".a3m", ".csv")


def find_cached_msa(cache_root: str | Path, sequence: str) -> Path | None:
    key = sequence_key(sequence)
    for suffix in MSA_SUFFIXES:
        candidate = Path(cache_root) / "msa" / f"{key}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _harvest_msa(search_root: Path, job_id: str) -> Path | None:
    """Find the MSA file Boltz wrote for ``job_id`` anywhere under ``search_root``."""
    found = []
    for folder in search_root.rglob("msa"):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix in MSA_SUFFIXES and path.name.startswith(job_id) and path.stat().st_size > 0:
                found.append(path)
    return sorted(found)[0] if found else None


def _pick_job_yaml(state) -> Path:
    job_dir = Path(state["job_directory"])
    manifest = state.get("manifest")
    if manifest is not None and "is_parent" in manifest:
        parents = manifest.loc[manifest["is_parent"].astype(str).str.lower().isin(["true", "1"]), "job_id"]
        for job_id in parents:
            candidate = job_dir / f"{job_id}.yaml"
            if candidate.is_file():
                return candidate
    yamls = sorted(job_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No job YAML files in {job_dir}")
    return yamls[0]


def ensure_target_msa(state, cache_root: str | Path, *, seed: int = 1001, log=print) -> Path | None:
    """Return a cached MSA for the batch's protein, generating it once if needed.

    Returns None when there is nothing to reuse (the batch already uses a local
    MSA) or when generation did not yield a file; callers then keep the old
    behaviour (server MSA).
    """
    command = [str(c) for c in state["command"]]
    if "--use_msa_server" not in command:
        return None  # a local MSA is already in use
    template = _pick_job_yaml(state)
    sequence = protein_sequence_from_job(template)
    cached = find_cached_msa(cache_root, sequence)
    if cached is not None:
        log(f"MSA: reusing cached {cached}")
        return cached

    key = sequence_key(sequence)
    scratch = Path(cache_root) / "_msa_bootstrap" / key
    shutil.rmtree(scratch, ignore_errors=True)
    jobs, output = scratch / "jobs", scratch / "results"
    jobs.mkdir(parents=True)
    document = yaml.safe_load(template.read_text())
    document.pop("properties", None)  # structure only; we just want the MSA
    (jobs / template.name).write_text(yaml.safe_dump(document, sort_keys=False))

    bootstrap = _replace_command(command, state["job_directory"], state["output_directory"], jobs, output, seed)
    if "--diffusion_samples" in bootstrap:
        bootstrap[bootstrap.index("--diffusion_samples") + 1] = "1"
    log(f"MSA: generating once via the MSA server using {template.stem} (cached afterwards)")
    with (scratch / "boltz.log").open("w") as handle:
        completed = subprocess.run(bootstrap, stdout=handle, stderr=subprocess.STDOUT, check=False)

    harvested = _harvest_msa(output, template.stem)
    if harvested is None:
        log(
            f"MSA: no MSA file found under {output} (exit code {completed.returncode}); "
            f"falling back to per-batch server MSAs. See {scratch / 'boltz.log'}."
        )
        return None
    destination = Path(cache_root) / "msa" / f"{key}{harvested.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(harvested, destination)
    log(f"MSA: cached {destination}")
    return destination
