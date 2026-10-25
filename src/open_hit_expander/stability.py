"""Independent Boltz-2 replicate runs and within-model stability summaries.

Uses an existing prepared BOLTZ2_STATE (manifest, YAML job directory and CLI
command), so it does not regenerate analogues or require Boltz in the notebook
Python environment. Each replicate writes to an isolated directory.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _replace_command(command, original_jobs, original_output, new_jobs, new_output, seed):
    """Repoint the original CLI command while preserving screening settings."""
    cmd = [str(x) for x in command]
    if len(cmd) < 3 or cmd[1] != "predict":
        raise ValueError("Expected a prepared `boltz predict ...` command")
    old_jobs, old_output = Path(original_jobs).resolve(), Path(original_output).resolve()
    replaced_jobs = replaced_output = False
    for i, item in enumerate(cmd):
        if item.startswith("-"):
            continue
        try:
            path = Path(item).resolve()
        except (OSError, ValueError):
            continue
        if path == old_jobs:
            cmd[i] = str(new_jobs)
            replaced_jobs = True
        elif path == old_output:
            cmd[i] = str(new_output)
            replaced_output = True
    # The standard Boltz CLI uses `boltz predict <job_directory> --out_dir <path>`.
    if not replaced_jobs:
        cmd[2] = str(new_jobs)
    if not replaced_output:
        if "--out_dir" in cmd:
            cmd[cmd.index("--out_dir") + 1] = str(new_output)
        else:
            cmd += ["--out_dir", str(new_output)]
    if "--seed" in cmd:
        cmd[cmd.index("--seed") + 1] = str(seed)
    else:
        cmd += ["--seed", str(seed)]
    return cmd


def _results_from_run(state):
    """Delegate parsing and pIC50 conversion to the project's own code."""
    from open_hit_expander import collect_boltz2_results
    results, _ = collect_boltz2_results(state)
    return results


def run_stability_analysis(
    state: dict,
    n_replicates: int = 5,
    output_dir: str | Path = "hit_expander_output/stability",
    seeds: list[int] | None = None,
    preprocessing_threads: int | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerun the entire prepared batch with distinct seeds.

    Parameters
    ----------
    state : dict
        Prepared BOLTZ2_STATE, including `command`, `job_directory`,
        `output_directory`, `manifest`. Do not pass only a results CSV.
    n_replicates : int
        Includes the parent in every replicate batch; initial screening is not
        counted as a replicate because its seed/settings may be different.
    seeds : list[int] | None
        One unique seed for each replicate; defaults to 1001, 1002, ...
    overwrite : bool
        Explicitly permit rerunning existing replicate output directories.

    Returns
    -------
    (replicate_results, replicate_status)
    """
    if n_replicates < 2:
        raise ValueError("n_replicates must be at least 2 to estimate SD")
    if not state.get("command") or not state.get("job_directory"):
        raise ValueError("Pass the prepared BOLTZ2_STATE, not just saved results")
    seeds = list(range(1001, 1001 + n_replicates)) if seeds is None else list(seeds)
    if len(seeds) != n_replicates or len(set(seeds)) != n_replicates:
        raise ValueError("Provide exactly one distinct seed per replicate")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    original_jobs = Path(state["job_directory"]).resolve()
    original_output = Path(state["output_directory"]).resolve()
    if not original_jobs.is_dir():
        raise FileNotFoundError(f"Prepared YAML directory not found: {original_jobs}")
    manifest = state["manifest"]
    if manifest is None or len(manifest) == 0:
        raise ValueError("The prepared manifest is empty")
    if "is_parent" not in manifest.columns:
        raise ValueError("Manifest lacks is_parent column; cannot calculate paired deltas")
    parents = manifest["is_parent"].astype(str).str.lower().isin(["true", "1"])
    if int(parents.sum()) != 1:
        raise ValueError("Stability analysis currently requires exactly one parent per batch")

    all_rows, status = [], []
    for rep, seed in enumerate(seeds, start=1):
        folder = root / f"replicate_{rep:02d}_seed_{seed}"
        jobs, pred_output = folder / "jobs", folder / "boltz_output"
        report_path = folder / "replicate_results.csv"
        if report_path.exists() and not overwrite:
            try:
                df = pd.read_csv(report_path)
                all_rows.append(df)
                status.append({"replicate": rep, "seed": seed, "status": "cached", "error": ""})
            except Exception as exc:
                status.append({"replicate": rep, "seed": seed, "status": "failed", "error": str(exc)})
            continue
        if folder.exists() and not overwrite:
            raise FileExistsError(f"Incomplete replicate exists: {folder}. Inspect or pass overwrite=True.")
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        shutil.copytree(original_jobs, jobs)
        cmd = _replace_command(state["command"], original_jobs, original_output, jobs, pred_output, seed)
        (folder / "command.json").write_text(json.dumps(cmd, indent=2))
        started = time.monotonic()
        try:
            with (folder / "boltz.log").open("w") as log:
                execution = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            if execution.returncode:
                raise RuntimeError(f"Boltz exited {execution.returncode}; check {folder/'boltz.log'}")
            replica_state = dict(state)
            replica_state.update(root=folder, job_directory=jobs, output_directory=pred_output,
                                 command=cmd, manifest=manifest.copy(), results=None, ranked=None)
            df = _results_from_run(replica_state)
            if df is None or df.empty:
                raise RuntimeError("No predictions parsed for replicate")
            df = df.copy()
            df["replicate"] = rep
            df["seed"] = seed
            df.to_csv(report_path, index=False)
            all_rows.append(df)
            status.append({"replicate": rep, "seed": seed, "status": "ok",
                           "error": "", "seconds": round(time.monotonic()-started, 1)})
        except Exception as exc:
            status.append({"replicate": rep, "seed": seed, "status": "failed", "error": str(exc),
                           "seconds": round(time.monotonic()-started, 1)})
    results = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    statuses = pd.DataFrame(status)
    results.to_csv(root / "replicate_predictions.csv", index=False)
    statuses.to_csv(root / "replicate_status.csv", index=False)
    return results, statuses


def summarize_stability(stability_directory: str | Path):
    """Summarize replicates and paired deltas from one-parent batches."""
    folder = Path(stability_directory)
    df = pd.read_csv(folder / "replicate_predictions.csv")
    if df.empty:
        raise ValueError("No successful replicate predictions")
    required = {"job_id", "replicate", "predicted_pic50", "is_parent"}
    absent = required - set(df.columns)
    if absent:
        raise KeyError(f"Missing output fields: {sorted(absent)}")
    df["predicted_pic50"] = pd.to_numeric(df["predicted_pic50"], errors="coerce")
    df["is_parent"] = df["is_parent"].astype(str).str.lower().isin(["true", "1"])
    parents = df[df.is_parent].dropna(subset=["predicted_pic50"])
    if parents.groupby("replicate")["job_id"].nunique().gt(1).any():
        raise ValueError("Multiple parents detected in a replicate")
    parent_values = parents.groupby("replicate")["predicted_pic50"].first()
    df["parent_pic50"] = df["replicate"].map(parent_values)
    df["delta_pic50_paired"] = df["predicted_pic50"] - df["parent_pic50"]
    df.loc[df["is_parent"], "delta_pic50_paired"] = 0.0
    df["predicted_improvement"] = (df["delta_pic50_paired"] > 0).where(
        df["delta_pic50_paired"].notna())
    valid = df.dropna(subset=["predicted_pic50"])
    grouped = valid.groupby("job_id", sort=False)
    summary = grouped["predicted_pic50"].agg(
        n_success="count", mean_pic50="mean", sd_pic50="std", median_pic50="median",
        min_pic50="min", max_pic50="max").reset_index()
    deltas = df.groupby("job_id", sort=False)["delta_pic50_paired"].agg(
        n_paired="count", mean_delta_pic50="mean", sd_delta_pic50="std")
    fractions = df.groupby("job_id", sort=False)["predicted_improvement"].mean()
    summary = summary.join(deltas, on="job_id").join(
        fractions.rename("fraction_predicted_improvement"), on="job_id")
    summary["is_parent"] = summary["job_id"].isin(set(parents["job_id"]))
    summary.to_csv(folder / "stability_summary.csv", index=False)
    df.to_csv(folder / "replicate_predictions_with_deltas.csv", index=False)
    return summary, df


# Okabe-Ito colours: distinguishable with the common forms of colour-vision deficiency.
_IMPROVED, _MIXED, _WORSE, _PARENT = "#0072B2", "#E69F00", "#D55E00", "#000000"


def _present(fig, name, show, save_dir):
    """Save (optional) and show or close a figure; keeps notebooks from double-drawing."""
    import matplotlib.pyplot as plt
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    if not show:
        plt.close(fig)


def _dot_range(ax, frame, value, order, colors, reference=None, reference_label=None):
    """One row per ligand: every replicate as a dot, min-max as a line, median as a tick."""
    for row, job in enumerate(order):
        values = frame.loc[frame.job_id.eq(job), value].dropna().to_numpy()
        if not len(values):
            continue
        color = colors[job]
        ax.plot([values.min(), values.max()], [row, row], color=color, alpha=0.45, linewidth=2.5, zorder=1)
        ax.scatter(values, np.full(len(values), row), s=26, color=color, alpha=0.85, zorder=2)
        ax.plot([np.median(values)] * 2, [row - 0.32, row + 0.32], color=color, linewidth=2.2, zorder=3)
    if reference is not None:
        ax.axvline(reference, color=_PARENT, linestyle="--", linewidth=1, zorder=0, label=reference_label)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)  # best at the top
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_stability(stability_directory: str | Path, max_ligands: int = 30, *, show: bool = True,
                   save_dir: str | Path | None = None):
    """Three figures: potency per replicate, paired change vs parent, and spread vs potency.

    Every replicate is drawn as a dot (with two or three replicates, box plots
    summarise nothing), with a tick at the median and a line from minimum to
    maximum. Pass ``save_dir`` to also write PNGs; ``show=False`` suppresses display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    summary, df = summarize_stability(stability_directory)
    df["job_id"] = df["job_id"].astype(str)
    summary["job_id"] = summary["job_id"].astype(str)
    n_rep = int(df["replicate"].nunique())
    ranked = summary.sort_values("median_pic50", ascending=False)
    parents = ranked.loc[ranked.is_parent, "job_id"].tolist()
    others = ranked.loc[~ranked.is_parent, "job_id"].head(max_ligands).tolist()
    order = sorted(parents + others, key=lambda j: -float(ranked.set_index("job_id").loc[j, "median_pic50"]))
    parent_median = float(ranked.loc[ranked.is_parent, "median_pic50"].iloc[0]) if parents else None
    height = max(3.2, 0.3 * len(order) + 1.6)
    figures = []

    # 1. Potency, one row per ligand.
    fig, ax = plt.subplots(figsize=(8.5, height))
    colors = {j: (_PARENT if j in parents else _IMPROVED) for j in order}
    _dot_range(ax, df, "predicted_pic50", order, colors, parent_median, "parent median")
    ax.set(xlabel="Predicted pIC50   (dot = one replicate, tick = median)",
           title=f"Boltz-2 potency across {n_rep} replicate(s)")
    if parent_median is not None:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _present(fig, "stability_potency", show, save_dir)
    figures.append(fig)

    # 2. Paired change versus parent, coloured by consistency across replicates.
    analogues = [j for j in order if j not in parents]
    if analogues:
        deltas = {j: df.loc[df.job_id.eq(j), "delta_pic50_paired"].dropna() for j in analogues}
        colors = {j: (_IMPROVED if len(d) and (d > 0).all() else _WORSE if len(d) and (d < 0).all() else _MIXED)
                  for j, d in deltas.items()}
        by_delta = sorted(analogues, key=lambda j: -(deltas[j].median() if len(deltas[j]) else -np.inf))
        fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.3 * len(by_delta) + 1.6)))
        _dot_range(ax, df, "delta_pic50_paired", by_delta, colors, 0.0)
        ax.set(xlabel="ΔpIC50 versus parent, paired within each replicate",
               title="Predicted change versus parent")
        ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=c, label=t) for c, t in
                           [(_IMPROVED, "higher in every replicate"), (_MIXED, "mixed"),
                            (_WORSE, "lower in every replicate")]],
                  frameon=False, loc="lower right")
        fig.tight_layout()
        _present(fig, "stability_delta_vs_parent", show, save_dir)
        figures.append(fig)

    # 3. Is the potency ranking noisy where it matters?
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    frame = summary.set_index("job_id").reindex(order)
    spread = (frame["max_pic50"] - frame["min_pic50"]).fillna(0)
    ax.scatter(frame["median_pic50"], spread,
               c=[(_PARENT if j in parents else _IMPROVED) for j in order], s=34, alpha=0.85)
    for job in list(parents) + order[:5]:
        ax.annotate(job, (frame.loc[job, "median_pic50"], spread[job]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="Median predicted pIC50", ylabel="Range across replicates (max − min)",
           title="Replicate spread versus potency")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if n_rep < 4:
        ax.text(0.99, 0.02, f"{n_rep} replicates: range shown, SD not reliable", transform=ax.transAxes,
                ha="right", fontsize=7, color="#555555")
    fig.tight_layout()
    _present(fig, "stability_spread", show, save_dir)
    figures.append(fig)

    if show:
        plt.show()
    return figures
