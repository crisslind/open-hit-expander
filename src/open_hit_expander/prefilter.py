"""Cheap, CPU-only triage of enumerated analogues BEFORE any cofolding.

``run_hit_expander`` already standardizes structures and applies absolute
property bounds and alert catalogs. This stage adds what is specific to hit
expansion and to spending scarce GPU time well:

* property windows relative to the parent (how far has the analogue drifted?)
* structural alerts judged against the parent, so liabilities the hit already
  carries do not remove every analogue, with an allow-list for intended groups
* must-keep / must-avoid substructures, including a core defined by parent atom
  indices (the same indices used for ``selected_atoms``)
* Boltz-2 size limits, applied gracefully instead of aborting the whole batch
* an optional diversity-aware cap so the batch is not 25 near-duplicates

Every decision is logged per compound. The input run is never modified: a new
run dict is returned with a smaller ``passed`` table, so it can be handed
straight to ``prepare_boltz2_batch``.

Tip: for alerts to be flagged here instead of dropped earlier, construct
``HitExpanderConfig(reject_alert_catalogs=())`` (or ``alert_policy="new_only"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
from rdkit import Chem, DataStructs

from .core import FP_GENERATOR, _alert_matches, _build_alert_catalogs, _properties

# Limits quoted by the existing manifest checks (heavy atoms): above SOFT the
# affinity head is outside its recommended range, above HARD Boltz refuses.
BOLTZ_SOFT_ATOMS = 56
BOLTZ_HARD_ATOMS = 128

DEFAULT_WINDOWS: dict[str, tuple[float | None, float | None]] = {
    # property: (min delta, max delta) relative to the parent. None = open.
    "mw": (-50.0, 100.0),
    "clogp": (-1.0, 1.5),
    "tpsa": (-30.0, 40.0),
    "hbd": (-1, 2),
    "rotatable_bonds": (-2, 3),
}


@dataclass(frozen=True)
class PrefilterConfig:
    # Property windows relative to the parent; {} disables. Illustrative defaults.
    windows: dict[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: dict(DEFAULT_WINDOWS))
    window_action: str = "drop"          # "drop" or "flag"

    # Alerts: only alerts the analogue has and the parent lacks are reported.
    alert_catalogs: tuple[str, ...] = ("PAINS", "BRENK")
    alert_allowlist: tuple[str, ...] = ()  # alert descriptions to ignore
    alert_action: str = "flag"           # "flag" (default) or "drop"

    # Substructure rules. ``core_atoms`` maps source_id -> parent atom indices
    # whose substructure every analogue must still contain; ``core_smarts``
    # does the same with an explicit SMARTS (a str applies to every parent).
    core_atoms: dict[str, Sequence[int]] | None = None
    core_smarts: str | dict[str, str] | None = None
    require_smarts: tuple[str, ...] = ()  # every pattern must match
    forbid_smarts: tuple[str, ...] = ()   # no pattern may match
    smarts_action: str = "drop"

    # Boltz-2 limits (heavy atoms). Hard limit always drops; soft limit flags.
    soft_atom_limit: int | None = BOLTZ_SOFT_ATOMS
    hard_atom_limit: int | None = BOLTZ_HARD_ATOMS

    # Optional cap per parent after filtering. "maxmin" spreads picks across
    # chemical space; "most_similar" keeps the closest analogues to the parent.
    max_candidates_per_parent: int | None = None
    selection: str = "maxmin"


# ---------------------------------------------------------------- pure helpers
def _window_reasons(deltas: dict, windows: dict) -> list[str]:
    """Which parent-relative windows does a compound leave? Pure function."""
    reasons = []
    for name, (low, high) in windows.items():
        value = deltas.get(name)
        if value is None or pd.isna(value):
            continue
        if low is not None and value < low:
            reasons.append(f"Δ{name} {value:+.2f} < {low:+g}")
        if high is not None and value > high:
            reasons.append(f"Δ{name} {value:+.2f} > {high:+g}")
    return reasons


def _maxmin_pick(fingerprints: list, count: int) -> list[int]:
    """Greedy MaxMin diversity pick (Tanimoto). Deterministic; starts at index 0."""
    if count >= len(fingerprints):
        return list(range(len(fingerprints)))
    picked = [0]
    nearest = [1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fingerprints[0], fingerprints)]
    nearest[0] = -1.0
    while len(picked) < count:
        best = max(range(len(fingerprints)), key=lambda i: (nearest[i], -i))
        picked.append(best)
        distances = [1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fingerprints[best], fingerprints)]
        nearest = [min(a, b) for a, b in zip(nearest, distances)]
        for chosen in picked:
            nearest[chosen] = -1.0
    return sorted(picked)


def _check_action(name: str, value: str) -> None:
    if value not in {"drop", "flag"}:
        raise ValueError(f"{name} must be 'drop' or 'flag', not {value!r}")


def _smarts(pattern: str) -> Chem.Mol:
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"Invalid SMARTS: {pattern!r}")
    return query


def _core_queries(parents: dict, config: PrefilterConfig) -> dict[str, Chem.Mol | None]:
    queries: dict[str, Chem.Mol | None] = {}
    for source_id, parent in parents.items():
        query = None
        if config.core_smarts is not None:
            pattern = (config.core_smarts if isinstance(config.core_smarts, str)
                       else config.core_smarts.get(source_id))
            query = _smarts(pattern) if pattern else None
        if query is None and config.core_atoms and source_id in config.core_atoms:
            atoms = [int(i) for i in config.core_atoms[source_id]]
            n_atoms = parent["mol"].GetNumAtoms()
            if not atoms or any(i < 0 or i >= n_atoms for i in atoms):
                raise ValueError(f"core_atoms for {source_id!r} must be indices in 0..{n_atoms - 1}")
            query = _smarts(Chem.MolFragmentToSmarts(parent["mol"], atomsToUse=atoms))
        queries[source_id] = query
    return queries


# ------------------------------------------------------------------- main API
def apply_prefilter(hit_run: dict, config: PrefilterConfig | None = None) -> dict:
    """Return a copy of ``hit_run`` whose ``passed`` table has been pre-filtered.

    Added to the returned dict: ``prefilter_log`` (one row per decision),
    ``prefilter_funnel`` (counts after each stage), ``prefilter_rejected`` and
    ``prefilter_config``. Added to ``passed``: ``prefilter_status``
    (pass / flag), ``prefilter_flags``, ``new_alerts`` and ``delta_*`` columns.
    """
    config = config or PrefilterConfig()
    for name in ("window_action", "alert_action", "smarts_action"):
        _check_action(name, getattr(config, name))
    if config.selection not in {"maxmin", "most_similar"}:
        raise ValueError("selection must be 'maxmin' or 'most_similar'")

    table = hit_run["passed"].copy().reset_index(drop=True)
    parents = {p["source_id"]: p for p in hit_run["parents"]}
    catalogs = _build_alert_catalogs(config.alert_catalogs)
    allow = set(config.alert_allowlist)
    parent_props = {sid: _properties(p["mol"]) for sid, p in parents.items()}
    known = set(next(iter(parent_props.values())))
    unknown = set(config.windows) - known
    if unknown:
        raise ValueError(f"Unknown window propert(ies) {sorted(unknown)}; choose from {sorted(known)}")
    parent_alerts = {sid: _alert_matches(p["mol"], catalogs) for sid, p in parents.items()}
    core = _core_queries(parents, config)
    required = [_smarts(s) for s in config.require_smarts]
    forbidden = [_smarts(s) for s in config.forbid_smarts]

    log: list[dict] = []
    status, flags_text, alerts_text, first_drop = [], [], [], []
    delta_columns: dict[str, list] = {f"delta_{k}": [] for k in config.windows}

    def note(row, stage, action, reason):
        log.append({"source_id": row["source_id"], "analog_smiles": row["analog_smiles"],
                    "stage": stage, "action": action, "reason": reason})

    for _, row in table.iterrows():
        sid, smiles = row["source_id"], row["analog_smiles"]
        mol = hit_run["mol_by_key"][(sid, smiles)]
        drops: list[tuple[str, str]] = []
        flags: list[str] = []

        def decide(stage, action, reason):
            note(row, stage, action, reason)
            (drops.append((stage, reason)) if action == "drop" else flags.append(f"{stage}: {reason}"))

        # Boltz-2 size limits.
        atoms = int(row["heavy_atoms"]) if "heavy_atoms" in row else mol.GetNumHeavyAtoms()
        if config.hard_atom_limit is not None and atoms > config.hard_atom_limit:
            decide("boltz_limits", "drop", f"{atoms} heavy atoms > hard limit {config.hard_atom_limit}")
        elif config.soft_atom_limit is not None and atoms > config.soft_atom_limit:
            decide("boltz_limits", "flag", f"{atoms} heavy atoms > recommended {config.soft_atom_limit}")

        # Parent-relative property windows.
        deltas = {k: float(row[k]) - float(parent_props[sid][k]) for k in config.windows}
        for k, v in deltas.items():
            delta_columns[f"delta_{k}"].append(round(v, 3))
        for reason in _window_reasons(deltas, config.windows):
            decide("windows", config.window_action, reason)

        # Core retention and substructure rules.
        if core[sid] is not None and not mol.HasSubstructMatch(core[sid]):
            decide("substructure", config.smarts_action, "core of the parent is not retained")
        for pattern, query in zip(config.require_smarts, required):
            if not mol.HasSubstructMatch(query):
                decide("substructure", config.smarts_action, f"required pattern absent: {pattern}")
        for pattern, query in zip(config.forbid_smarts, forbidden):
            if mol.HasSubstructMatch(query):
                decide("substructure", config.smarts_action, f"forbidden pattern present: {pattern}")

        # Alerts the analogue introduces relative to the parent.
        new_alerts = []
        for catalog, names in _alert_matches(mol, catalogs).items():
            baseline = set(parent_alerts[sid].get(catalog, []))
            new_alerts += [f"{catalog}: {n}" for n in names if n not in baseline and n not in allow]
        for alert in new_alerts:
            decide("alerts", config.alert_action, f"new alert {alert}")

        status.append("drop" if drops else "flag" if flags else "pass")
        flags_text.append("; ".join(flags))
        alerts_text.append("; ".join(new_alerts))
        first_drop.append(drops[0][0] if drops else "")

    table["prefilter_status"] = status
    table["prefilter_flags"] = flags_text
    table["new_alerts"] = alerts_text
    for name, values in delta_columns.items():
        table[name] = values
    table["_first_drop"] = first_drop

    kept = table.loc[table["prefilter_status"] != "drop"].copy()
    rejected = table.loc[table["prefilter_status"] == "drop"].copy()

    selected_out = 0
    if config.max_candidates_per_parent is not None and not kept.empty:
        chosen = []
        for sid, group in kept.groupby("source_id", sort=False):
            group = group.sort_values("morgan_tanimoto", ascending=False, kind="stable")
            if config.selection == "most_similar":
                chosen += list(group.index[: config.max_candidates_per_parent])
            else:
                fingerprints = [FP_GENERATOR.GetFingerprint(hit_run["mol_by_key"][(sid, s)])
                                for s in group["analog_smiles"]]
                chosen += [group.index[i] for i in _maxmin_pick(fingerprints, config.max_candidates_per_parent)]
        dropped_by_cap = kept.index.difference(chosen)
        selected_out = len(dropped_by_cap)
        for idx in dropped_by_cap:
            note(kept.loc[idx], "selection", "drop",
                 f"not among {config.max_candidates_per_parent} {config.selection} picks per parent")
        kept = kept.loc[sorted(chosen)]

    funnel = [{"stage": "input (passed core filters)", "remaining": len(table)}]
    remaining = len(table)
    for stage in ("boltz_limits", "windows", "substructure", "alerts"):
        remaining -= int((table["_first_drop"] == stage).sum())
        funnel.append({"stage": f"after {stage}", "remaining": remaining})
    if config.max_candidates_per_parent is not None:
        funnel.append({"stage": f"after selection ({config.selection})", "remaining": remaining - selected_out})

    result = dict(hit_run)
    result["passed"] = kept.drop(columns=["_first_drop"]).reset_index(drop=True)
    result["prefilter_rejected"] = rejected.drop(columns=["_first_drop"]).reset_index(drop=True)
    result["prefilter_log"] = pd.DataFrame(log, columns=["source_id", "analog_smiles", "stage", "action", "reason"])
    result["prefilter_funnel"] = pd.DataFrame(funnel)
    result["prefilter_config"] = config
    return result


def export_prefilter(run: dict, output_directory: str | Path = "hit_expander_output") -> dict[str, Path]:
    """Write the decision log, funnel and rejected analogues next to the other outputs."""
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "log": out / "prefilter_log.csv",
        "funnel": out / "prefilter_funnel.csv",
        "rejected": out / "prefilter_rejected.csv",
    }
    run["prefilter_log"].to_csv(paths["log"], index=False)
    run["prefilter_funnel"].to_csv(paths["funnel"], index=False)
    run["prefilter_rejected"].to_csv(paths["rejected"], index=False)
    return paths
