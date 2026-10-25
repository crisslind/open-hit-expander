from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, math, re, shutil, subprocess
import pandas as pd
import yaml
from rdkit import Chem

from .jobcache import (
    CofoldCache, cache_key, job_msa_path, ligand_key, prediction_directory,
    protein_sequence_from_job, set_job_msa, file_digest,
)
def _clean_protein_sequence(sequence):
    cleaned = re.sub(r"\s+", "", sequence).upper()
    if not cleaned:
        raise ValueError("PROTEIN_SEQUENCE is empty")
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", cleaned):
        raise ValueError("PROTEIN_SEQUENCE contains invalid amino-acid characters")
    return cleaned


def _resolve_boltz_executable(executable):
    if not executable:
        raise ValueError(
            "BOLTZ_EXECUTABLE is empty. Activate the boltz2 Conda environment, "
            "run `which boltz`, and paste the returned path into the configuration."
        )
    candidate = Path(executable).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Boltz executable not found: {executable}")
    return Path(resolved).resolve()


def _boltz_ligand_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles}")
    # Approximate the atom counting used by the Boltz affinity input check.
    return Chem.RemoveHs(Chem.AddHs(mol)).GetNumAtoms()


def build_boltz2_manifest(hit_run, max_analogues=None):
    if hit_run is None:
        raise RuntimeError("Run Hit Expander before preparing Boltz-2 jobs")

    records = []
    for parent in hit_run["parents"]:
        smiles = parent["source_smiles"]
        records.append({
            "job_id": f"{parent['source_id']}_parent",
            "source_id": parent["source_id"],
            "ligand_smiles": smiles,
            "transformation": "parent",
            "atom_indices": "",
            "is_parent": True,
            "morgan_tanimoto": 1.0,
            "boltz_atom_count": _boltz_ligand_atom_count(smiles),
            "ligand_key": ligand_key(smiles),
            "prefilter_flags": "",
        })

    analogues = hit_run["passed"].copy()
    if max_analogues is not None:
        analogues = analogues.head(int(max_analogues))

    for number, (_, row) in enumerate(analogues.iterrows(), start=1):
        smiles = row["analog_smiles"]
        records.append({
            "job_id": f"{row['source_id']}_analogue_{number:04d}",
            "source_id": row["source_id"],
            "ligand_smiles": smiles,
            "transformation": row["transformations"],
            "atom_indices": row["atom_indices"],
            "is_parent": False,
            "morgan_tanimoto": row["morgan_tanimoto"],
            "boltz_atom_count": _boltz_ligand_atom_count(smiles),
            "ligand_key": ligand_key(smiles),
            "prefilter_flags": row.get("prefilter_flags", "") if hasattr(row, "get") else "",
        })

    manifest = pd.DataFrame(records)
    if manifest["job_id"].duplicated().any():
        raise ValueError("Duplicate Boltz job IDs were generated")
    if (manifest["boltz_atom_count"] > 128).any():
        oversized = manifest.loc[manifest["boltz_atom_count"] > 128, ["job_id", "boltz_atom_count"]]
        raise ValueError(
            "A ligand exceeds the Boltz affinity limit of 128 atoms:\n"
            + oversized.to_string(index=False)
            + "\nDrop these (see open_hit_expander.prefilter) or lower max_analogues."
        )
    if (manifest["boltz_atom_count"] > 56).any():
        print("Warning: a ligand exceeds the recommended 56-atom affinity range")
    return manifest


def make_boltz2_job(protein_sequence, ligand_smiles, msa_path=None):
    protein = {"id": "A", "sequence": protein_sequence}
    if msa_path is not None:
        protein["msa"] = str(Path(msa_path).expanduser().resolve())
    return {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": {"id": "B", "smiles": ligand_smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }


def prepare_boltz2_batch(
    hit_run,
    protein_sequence,
    boltz_executable,
    msa_path=None,
    max_analogues=None,
    accelerator="cpu",
    devices=1,
    diffusion_samples=3,
    affinity_mw_correction=False,
    output_directory="hit_expander_output",
    screening_preset="balanced",
    screening_presets=None,
    no_kernels=False,
    num_workers=0,
    preprocessing_threads=1,
):
    sequence = _clean_protein_sequence(protein_sequence)
    executable = _resolve_boltz_executable(boltz_executable)

    if msa_path is not None:
        msa_path = Path(msa_path).expanduser().resolve()
        if not msa_path.is_file():
            raise FileNotFoundError(f"MSA file not found: {msa_path}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(output_directory) / "boltz2_cofold" / run_tag
    job_directory = root / "jobs"
    output_directory = root / "results"
    job_directory.mkdir(parents=True, exist_ok=False)
    output_directory.mkdir(parents=True, exist_ok=False)

    manifest = build_boltz2_manifest(hit_run, max_analogues=max_analogues)
    for record in manifest.to_dict(orient="records"):
        document = make_boltz2_job(
            sequence, record["ligand_smiles"], msa_path=msa_path
        )
        path = job_directory / f"{record['job_id']}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))

    command = [
        str(executable),
        "predict",
        str(job_directory.resolve()),
        "--out_dir",
        str(output_directory.resolve()),
        "--output_format",
        "mmcif",
        "--accelerator",
        accelerator,
        "--devices",
        str(int(devices)),
        "--diffusion_samples",
        str(int(diffusion_samples)),
        "--use_potentials",
    ]
    if msa_path is None:
        command.append("--use_msa_server")
    if affinity_mw_correction:
        command.append("--affinity_mw_correction")
    if no_kernels:
        command.append("--no_kernels")

    return {
        "root": root,
        "job_directory": job_directory,
        "output_directory": output_directory,
        "manifest": manifest,
        "command": command,
        "screening_preset": screening_preset,
        "screening_settings": {"diffusion_samples": diffusion_samples},
        "protein_sequence": sequence,
        "msa_path": str(msa_path) if msa_path is not None else None,
    }
def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def affinity_value_to_pic50(value):
    value = _finite_float(value)
    return None if value is None else 6.0 - value


def affinity_value_to_ic50_uM(value):
    value = _finite_float(value)
    return None if value is None else 10.0 ** value


def parse_boltz2_prediction(job_id, output_directory):
    # Boltz writes predictions beneath boltz_results_<input-job-directory>.
    # Support both this layout and the older direct predictions directory.
    output_directory = Path(output_directory)
    candidates = [output_directory / "predictions" / job_id]
    candidates.extend(p / job_id for p in sorted(output_directory.glob("boltz_results_*/predictions")))
    prediction_directory = next((p for p in candidates if p.is_dir()), candidates[0])
    result = {
        "job_id": job_id,
        "cofold_status": "not_found",
        "structure_path": None,
        "confidence_score": None,
        "ptm": None,
        "iptm": None,
        "ligand_iptm": None,
        "protein_iptm": None,
        "complex_plddt": None,
        "complex_iplddt": None,
        "complex_pde": None,
        "complex_ipde": None,
        "affinity_pred_value": None,
        "affinity_probability_binary": None,
        "affinity_pred_value1": None,
        "affinity_probability_binary1": None,
        "affinity_pred_value2": None,
        "affinity_probability_binary2": None,
        "predicted_pic50": None,
        "predicted_ic50_uM": None,
        "predicted_ic50_nM": None,
    }

    if not prediction_directory.is_dir():
        result["cofold_status"] = "missing_prediction_directory"
        return result

    confidence_candidates = []
    for path in prediction_directory.glob("confidence_*.json"):
        try:
            data = _read_json(path)
            score = _finite_float(data.get("confidence_score"))
            confidence_candidates.append((
                score if score is not None else float("-inf"), path, data
            ))
        except Exception:
            continue

    if not confidence_candidates:
        result["cofold_status"] = "missing_confidence"
        return result

    _, confidence_path, confidence = max(
        confidence_candidates, key=lambda item: item[0]
    )
    model_name = confidence_path.stem.removeprefix("confidence_")
    structure_path = next((
        path
        for path in [
            prediction_directory / f"{model_name}.cif",
            prediction_directory / f"{model_name}.pdb",
        ]
        if path.is_file()
    ), None)

    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        result[key] = _finite_float(confidence.get(key))
    result["structure_path"] = (
        str(structure_path.resolve()) if structure_path is not None else None
    )
    if structure_path is None:
        result["cofold_status"] = "missing_structure"
        return result

    affinity_paths = sorted(prediction_directory.glob("affinity_*.json"))
    if not affinity_paths:
        result["cofold_status"] = "missing_affinity"
        return result

    affinity = _read_json(affinity_paths[0])
    for key in [
        "affinity_pred_value", "affinity_probability_binary",
        "affinity_pred_value1", "affinity_probability_binary1",
        "affinity_pred_value2", "affinity_probability_binary2",
    ]:
        result[key] = _finite_float(affinity.get(key))

    value = result["affinity_pred_value"]
    result["predicted_pic50"] = affinity_value_to_pic50(value)
    result["predicted_ic50_uM"] = affinity_value_to_ic50_uM(value)
    if result["predicted_ic50_uM"] is not None:
        result["predicted_ic50_nM"] = result["predicted_ic50_uM"] * 1000.0
    result["cofold_status"] = "success"
    return result

def run_boltz2_batch(state, preprocessing_threads=1):
    command = state.get("command")
    if not command:
        raise RuntimeError("Prepare a Boltz-2 batch first")
    log_path = Path(state["root"]) / "boltz2_run.log"
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"Boltz failed ({code}); see {log_path}")
    return log_path


def collect_boltz2_results(state):
    manifest = state["manifest"].copy()
    parsed = pd.DataFrame([parse_boltz2_prediction(job_id, state["output_directory"])
                           for job_id in manifest["job_id"]])
    results = manifest.merge(parsed, on="job_id", how="left")
    parents = (results.loc[results["is_parent"].astype(str).str.lower().eq("true") &
                          results["cofold_status"].eq("success")]
               .drop_duplicates("source_id").set_index("source_id"))
    for field, new in [("predicted_pic50","parent_predicted_pic50"),
                       ("ligand_iptm","parent_ligand_iptm"),
                       ("confidence_score","parent_confidence")]:
        results[new] = results["source_id"].map(parents[field].to_dict()) if field in parents else float("nan")
    results["delta_pic50_vs_parent"] = results["predicted_pic50"] - results["parent_predicted_pic50"]
    results["predicted_potency_fold_vs_parent"] = 10 ** results["delta_pic50_vs_parent"]
    results["delta_ligand_iptm_vs_parent"] = results["ligand_iptm"] - results["parent_ligand_iptm"]
    results["delta_confidence_vs_parent"] = results["confidence_score"] - results["parent_confidence"]
    ranked = (results.loc[~results["is_parent"].astype(str).str.lower().eq("true") &
                          results["cofold_status"].eq("success")]
              .sort_values(["predicted_pic50","ligand_iptm","confidence_score"],ascending=False)
              .reset_index(drop=True))
    ranked.insert(0,"boltz2_rank",range(1,len(ranked)+1))
    root = Path(state["root"])
    manifest.to_csv(root / "hit_expander_boltz2_manifest.csv",index=False)
    results.to_csv(root / "hit_expander_boltz2_all_results.csv",index=False)
    ranked.to_csv(root / "hit_expander_boltz2_ranked_analogues.csv",index=False)
    state.update(results=results,ranked=ranked)
    return results,ranked

SCREENING_PRESETS = {"balanced": {"diffusion_samples": 3}, "quick": {"diffusion_samples": 1}, "confirm": {"diffusion_samples": 5}}


# Integrated cofolding replicates: this is the ONLY inference stage.
def _is_true(series):
    return series.astype(str).str.lower().isin(["true", "1"])


def _strip_flag(command, flag):
    return [c for c in command if c != flag]


def _replicate_dir(batches, index, seed):
    return batches / f"replicate_{index:02d}_seed_{seed}"


def _resolve_cache_roots(state, cache_dir):
    """Return (cofold_cache_root | None, msa_cache_root). ``"auto"`` puts both
    next to the run folders: <output_directory>/cofold_cache and /msa_cache."""
    if cache_dir is None:
        return None, Path(state["root"]).resolve().parent.parent / "msa_cache"
    if cache_dir == "auto":
        base = Path(state["root"]).resolve().parent.parent
        return base / "cofold_cache", base / "msa_cache"
    base = Path(cache_dir).expanduser().resolve()
    return base / "cofold_cache", base / "msa_cache"


def _msa_tag(job_yaml, msa_root):
    """"server" for MSAs we generated from the server; a file digest for a user MSA."""
    msa = job_msa_path(job_yaml)
    if not msa:
        return "server"
    msa = Path(msa)
    try:
        msa.resolve().relative_to(Path(msa_root).resolve())
        return "server"  # our own cached server MSA
    except ValueError:
        return file_digest(msa) if msa.is_file() else "server"


def plan_boltz2_replicates(state, n_replicates=1, seeds=None, cache_dir="auto"):
    """Dry run: how much cofolding is still needed? Launches nothing.

    Returns one row per (replicate, seed) with the number of jobs that are
    already finished on disk, restorable from the cache, and still pending.
    """
    seeds = list(range(1001, 1001 + n_replicates)) if seeds is None else list(seeds)
    manifest = state["manifest"]
    batches = Path(state["root"]).resolve() / "replicates"
    cache_root, msa_root = _resolve_cache_roots(state, cache_dir)
    cache = CofoldCache(cache_root) if cache_root is not None else None
    jobs_dir = Path(state["job_directory"])
    sequence = state.get("protein_sequence")
    rows = []
    for index, seed in enumerate(seeds, 1):
        output = _replicate_dir(batches, index, seed) / "results"
        done = cached = pending = 0
        for record in manifest.to_dict(orient="records"):
            job_id = str(record["job_id"])
            if parse_boltz2_prediction(job_id, output)["cofold_status"] == "success":
                done += 1
                continue
            hit = False
            if cache is not None and (jobs_dir / f"{job_id}.yaml").is_file():
                seq = sequence or protein_sequence_from_job(jobs_dir / f"{job_id}.yaml")
                key = cache_key(seq, record["ligand_smiles"], seed, state["command"],
                                _msa_tag(jobs_dir / f"{job_id}.yaml", msa_root))
                hit = cache.has(key)
            cached += hit
            pending += not hit
        rows.append({"replicate": index, "seed": seed, "jobs": len(manifest),
                     "done_on_disk": done, "in_cache": cached, "to_compute": pending})
    return pd.DataFrame(rows)


def run_boltz2_replicates(state, n_replicates=1, seeds=None, overwrite=False,
                          *, cache_dir="auto", reuse_msa=True, log=print):
    """Run the prepared YAML batch once per seed; resumable per job.

    Each replicate lives in ``replicates/replicate_NN_seed_S`` as before. Work is
    tracked per job (a job is finished when its structure, confidence and
    affinity files parse), so after a crash or interrupt, calling this again
    runs only the unfinished jobs. Earlier versions raised ``FileExistsError``
    and required ``overwrite=True``, which deleted the finished jobs too.

    cache_dir : "auto" (default) keeps a cross-run prediction cache and an MSA
        cache beside the run folders; a path puts both there; None disables the
        prediction cache. A cached prediction is reused when protein sequence,
        ligand SMILES, seed and prediction settings all match.
    reuse_msa : compute the target MSA once and reuse it for every replicate
        when the batch was prepared with ``--use_msa_server``.
    overwrite : delete and redo the replicate folders (explicit opt-in only).
    """
    import time
    if not isinstance(n_replicates, int) or n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    seeds = list(range(1001, 1001 + n_replicates)) if seeds is None else list(seeds)
    if len(seeds) != n_replicates or len(set(seeds)) != n_replicates:
        raise ValueError("Supply one distinct seed per replicate")
    original_jobs = Path(state["job_directory"]).resolve()
    if not original_jobs.is_dir():
        raise FileNotFoundError(original_jobs)
    manifest = state["manifest"]
    if manifest is None or manifest.empty:
        raise ValueError("Empty manifest")
    from .stability import _replace_command
    root = Path(state["root"]).resolve()
    batches = root / "replicates"
    batches.mkdir(parents=True, exist_ok=True)

    cache_root, msa_root = _resolve_cache_roots(state, cache_dir)
    cache = CofoldCache(cache_root) if cache_root is not None else None

    base_command = [str(c) for c in state["command"]]
    msa_file = None
    if reuse_msa and "--use_msa_server" in base_command:
        from .msa import ensure_target_msa
        msa_file = ensure_target_msa(state, msa_root, seed=seeds[0], log=log)
        if msa_file is not None:
            base_command = _strip_flag(base_command, "--use_msa_server")

    sequence = state.get("protein_sequence") or protein_sequence_from_job(
        next(iter(sorted(original_jobs.glob("*.yaml")))))
    records = manifest.to_dict(orient="records")
    all_results, statuses = [], []

    for index, seed in enumerate(seeds, 1):
        run_dir = _replicate_dir(batches, index, seed)
        jobs, output = run_dir / "jobs", run_dir / "results"
        if overwrite and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not jobs.is_dir():
            shutil.copytree(original_jobs, jobs)
            if msa_file is not None:
                for yaml_file in jobs.glob("*.yaml"):
                    set_job_msa(yaml_file, msa_file)
        output.mkdir(parents=True, exist_ok=True)

        # 1. Triage: finished on disk / restorable from cache / still to compute.
        source, pending = {}, []
        for record in records:
            job_id = str(record["job_id"])
            if parse_boltz2_prediction(job_id, output)["cofold_status"] == "success":
                source[job_id] = "existing"
                continue
            if cache is not None:
                key = cache_key(sequence, record["ligand_smiles"], seed, base_command,
                                _msa_tag(jobs / f"{job_id}.yaml", msa_root))
                stale = prediction_directory(output, job_id)
                if stale.exists():
                    shutil.rmtree(stale)  # incomplete leftovers of an earlier attempt
                if cache.restore(key, prediction_directory(output, job_id), job_id):
                    source[job_id] = "cache"
                    continue
            pending.append(job_id)

        # 2. Compute only what is missing.
        started = time.monotonic()
        exit_code, error = 0, ""
        try:
            if pending:
                staging = run_dir / "_pending"
                shutil.rmtree(staging, ignore_errors=True)
                (staging / "jobs").mkdir(parents=True)
                for job_id in pending:
                    shutil.copy2(jobs / f"{job_id}.yaml", staging / "jobs" / f"{job_id}.yaml")
                    stale = prediction_directory(output, job_id)
                    if stale.exists():
                        shutil.rmtree(stale)
                # Input folder must be named "jobs" so results land in
                # results/boltz_results_jobs/, the layout older runs and the
                # notebook viewer expect.
                command = _replace_command(base_command, original_jobs, state["output_directory"],
                                           staging / "jobs", output, seed)
                (run_dir / "command.json").write_text(json.dumps(command, indent=2))
                log(f"Replicate {index} (seed {seed}): computing {len(pending)} of {len(records)} jobs"
                    f" ({sum(v == 'existing' for v in source.values())} finished, "
                    f"{sum(v == 'cache' for v in source.values())} from cache)")
                with (run_dir / "boltz.log").open("a") as log_file:
                    completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
                exit_code = completed.returncode
        finally:
            # 3. Harvest whatever finished (also after a crash or Ctrl-C).
            for job_id in pending:
                if parse_boltz2_prediction(job_id, output)["cofold_status"] == "success":
                    source[job_id] = "computed"
                    if cache is not None:
                        record = next(r for r in records if str(r["job_id"]) == job_id)
                        key = cache_key(sequence, record["ligand_smiles"], seed, base_command,
                                        _msa_tag(jobs / f"{job_id}.yaml", msa_root))
                        cache.store(key, prediction_directory(output, job_id), job_id)
            shutil.rmtree(run_dir / "_pending", ignore_errors=True)
            pd.DataFrame(
                [{"job_id": str(r["job_id"]), "status": source.get(str(r["job_id"]), "pending")}
                 for r in records]
            ).to_csv(run_dir / "job_status.csv", index=False)

        n_done = len(source)
        if exit_code and n_done < len(records):
            error = f"Boltz exit {exit_code}; see {run_dir / 'boltz.log'}"
        run_state = dict(state, root=run_dir, job_directory=jobs, output_directory=output,
                         manifest=manifest.copy())
        if n_done:
            df, _ = collect_boltz2_results(run_state)
            df["replicate"] = index
            df["seed"] = seed
            df.to_csv(run_dir / "replicate_results.csv", index=False)
            all_results.append(df)
        status = "failed" if n_done == 0 else "ok" if n_done == len(records) else "partial"
        statuses.append(dict(
            replicate=index, seed=seed, status=status, error=error,
            n_jobs=len(records), n_success=n_done,
            n_existing=sum(v == "existing" for v in source.values()),
            n_cache=sum(v == "cache" for v in source.values()),
            n_computed=sum(v == "computed" for v in source.values()),
            seconds=round(time.monotonic() - started, 1),
        ))

    records_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    state["replicate_results"] = records_df
    state["replicate_status"] = pd.DataFrame(statuses)
    records_df.to_csv(root / "replicate_predictions.csv", index=False)
    state["replicate_status"].to_csv(root / "replicate_status.csv", index=False)
    if records_df.empty:
        raise RuntimeError("No successful Boltz predictions. Inspect per-replicate boltz.log files")
    if (state["replicate_status"]["status"] == "partial").any():
        log("Some jobs are still unfinished; call run_boltz2_replicates again to resume.")
    return records_df, state["replicate_status"]


def collect_boltz2_replicate_results(state):
    """Aggregate existing predictions; no inference. Return viewer-compatible results."""
    df = state.get('replicate_results')
    if df is None:
        df = pd.read_csv(Path(state['root'])/'replicate_predictions.csv')
    df = df.copy()
    df['predicted_pic50'] = pd.to_numeric(df['predicted_pic50'], errors='coerce')
    df['is_parent'] = df['is_parent'].astype(str).str.lower().isin(['true', '1'])
    parents = df[df['is_parent']].dropna(subset=['predicted_pic50'])
    # Paired differences are defined only if the parent from that seed succeeded.
    if parents.groupby(['replicate','source_id']).size().gt(1).any():
        raise ValueError('Duplicate parent predictions for a replicate/source')
    parent_lookup = parents.set_index(['replicate','source_id'])['predicted_pic50']
    df['parent_replicate_pic50'] = [parent_lookup.get((r.replicate, r.source_id), float('nan')) for r in df.itertuples()]
    df['delta_pic50_paired'] = df['predicted_pic50'] - df['parent_replicate_pic50']
    df['improved'] = (df['delta_pic50_paired'] > 0).where(df['delta_pic50_paired'].notna())
    records = []
    for job_id, group in df.groupby('job_id', sort=False):
        valid = group.dropna(subset=['predicted_pic50'])
        if valid.empty:
            representative = group.iloc[0].copy()
        else:
            median = valid['predicted_pic50'].median()
            representative = valid.loc[(valid['predicted_pic50']-median).abs().idxmin()].copy()
        p = valid['predicted_pic50']
        d = group['delta_pic50_paired'].dropna()
        representative['mean_pic50'] = p.mean()
        representative['median_pic50'] = p.median()
        representative['sd_pic50'] = p.std(ddof=1) if len(p)>1 else float('nan')
        representative['min_pic50'] = p.min()
        representative['max_pic50'] = p.max()
        representative['n_success'] = len(p)
        representative['median_delta_pic50'] = d.median() if len(d) else float('nan')
        representative['mean_delta_pic50'] = d.mean() if len(d) else float('nan')
        representative['sd_delta_pic50'] = d.std(ddof=1) if len(d)>1 else float('nan')
        representative['fraction_predicted_improvement'] = (d>0).mean() if len(d) else float('nan')
        representative['n_paired'] = len(d)
        # Display median as primary potency; preserve actual representative pose potency separately.
        # Keep the highest-scoring *replicate* alongside the median-nearest pose.
        # A Boltz job may generate several structural samples but only one
        # aggregated affinity prediction; do not count those as replicates.
        pose_candidates = valid.loc[valid['structure_path'].notna()].copy() if 'structure_path' in valid else valid.iloc[0:0].copy()
        if not pose_candidates.empty:
            pose_candidates = pose_candidates.sort_values(
                ['predicted_pic50', 'replicate'], ascending=[False, True], kind='stable'
            )
            best = pose_candidates.iloc[0]
            representative['best_structure_path'] = best['structure_path']
            representative['best_pic50'] = float(best['predicted_pic50'])
            representative['best_replicate'] = best.get('replicate')
            representative['best_seed'] = best.get('seed')
        else:
            representative['best_structure_path'] = None
            representative['best_pic50'] = float('nan')
            representative['best_replicate'] = None
            representative['best_seed'] = None
        representative['representative_pic50'] = representative['predicted_pic50']
        representative['representative_replicate'] = representative.get('replicate')
        representative['representative_seed'] = representative.get('seed')
        representative['predicted_pic50'] = p.median()
        records.append(representative)
    results = pd.DataFrame(records).drop(columns=['parent_replicate_pic50','improved','delta_pic50_paired'], errors='ignore')
    parent_medians = results[results.is_parent].set_index('source_id')['median_pic50']
    results['parent_predicted_pic50'] = results['source_id'].map(parent_medians)
    results['delta_pic50_vs_parent'] = results['predicted_pic50'] - results['parent_predicted_pic50']
    ranked = results.loc[~results.is_parent].sort_values('predicted_pic50', ascending=False).reset_index(drop=True)
    ranked.insert(0,'boltz2_rank',range(1,len(ranked)+1))
    root = Path(state['root'])
    results.to_csv(root/'hit_expander_boltz2_all_results.csv', index=False)
    ranked.to_csv(root/'hit_expander_boltz2_ranked_analogues.csv', index=False)
    df.to_csv(root/'replicate_predictions_with_deltas.csv', index=False)
    state.update(results=results, ranked=ranked, replicate_results=df)
    return results, ranked


def recover_boltz2_replicates(state, seeds=None):
    """Reparse existing Boltz inference outputs; never launch inference.

    Updates the replicate CSV caches, state and aggregate results. Safe when
    an older collector mistakenly marked predictions as missing.
    """
    root = Path(state["root"]).resolve()
    batches = root / "replicates"
    manifest = state["manifest"].copy()
    if not batches.is_dir():
        raise FileNotFoundError(batches)
    replicate_dirs = sorted(batches.glob("replicate_*_seed_*"))
    if seeds is not None:
        expected = {int(seed) for seed in seeds}
        replicate_dirs = [p for p in replicate_dirs if int(p.name.rsplit("_", 1)[-1]) in expected]
    if not replicate_dirs:
        raise FileNotFoundError("No matching completed replicate directories found")
    all_results, statuses = [], []
    for replicate_dir in replicate_dirs:
        name_parts = replicate_dir.name.split("_")
        replicate, seed = int(name_parts[1]), int(name_parts[-1])
        output = replicate_dir / "results"
        parsed = pd.DataFrame([
            parse_boltz2_prediction(str(job_id), output)
            for job_id in manifest["job_id"]
        ])
        data = manifest.merge(parsed, on="job_id", how="left")
        data["replicate"] = replicate
        data["seed"] = seed
        data.to_csv(replicate_dir / "replicate_results.csv", index=False)
        all_results.append(data)
        n_success = int(data["cofold_status"].eq("success").sum())
        statuses.append({"replicate": replicate, "seed": seed,
                         "status": "recovered" if n_success else "missing_predictions",
                         "n_success": n_success, "n_jobs": len(data)})
    records = pd.concat(all_results, ignore_index=True)
    state["replicate_results"] = records
    state["replicate_status"] = pd.DataFrame(statuses)
    records.to_csv(root / "replicate_predictions.csv", index=False)
    state["replicate_status"].to_csv(root / "replicate_status.csv", index=False)
    if not records["cofold_status"].eq("success").any():
        raise RuntimeError("No usable predictions recovered; inspect replicate logs")
    return collect_boltz2_replicate_results(state)


def load_boltz2_run(timestamp, output_directory="hit_expander_output", *, reparse=True):
    """Load an existing timestamped run; NO Boltz inference is performed.

    By default, reparse prediction JSON and poses to repair stale/corrupt CSV caches.
    The returned results contain one representative structure per ligand, from
    the replicate whose potency is closest to that ligand's median pIC50.
    """
    import re
    timestamp = str(timestamp).strip()
    if not re.fullmatch(r"\d{8}_\d{6}_\d{6}", timestamp):
        raise ValueError("Timestamp must look like 20261013_111348_145796")
    run_root = (Path(output_directory) / "boltz2_cofold" / timestamp).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Saved run not found: {run_root}")
    replicate_dirs = sorted((run_root / "replicates").glob("replicate_*_seed_*"))
    if not replicate_dirs:
        raise FileNotFoundError(f"No replicate directories in {run_root}")
    manifest_file = next((d / "hit_expander_boltz2_manifest.csv"
                          for d in replicate_dirs if (d / "hit_expander_boltz2_manifest.csv").is_file()), None)
    if manifest_file is None:
        manifest_file = run_root / "hit_expander_boltz2_manifest.csv"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"No saved hit expander job manifest found under {run_root}")
    manifest = pd.read_csv(manifest_file)
    if manifest.empty or not {"job_id", "source_id", "is_parent"}.issubset(manifest):
        raise ValueError(f"Incomplete or invalid saved manifest: {manifest_file}")
    state = {"root": run_root, "manifest": manifest,
             "job_directory": run_root / "jobs", "output_directory": run_root / "results"}
    if reparse or not (run_root / "replicate_predictions.csv").exists():
        results, ranked = recover_boltz2_replicates(state)
    else:
        results, ranked = collect_boltz2_replicate_results(state)
    return state, results, ranked
