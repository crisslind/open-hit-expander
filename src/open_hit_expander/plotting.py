"""Notebook plots for PLIP fingerprints exported by open_hit_expander.interactions.

No PLIP rerun is required. Use plot_interaction_summary(...) for all plots.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Okabe-Ito based palette (colour-vision-deficiency safe), one hue per interaction type.
COLORS = {
    "Hydrogen bond": "#0072B2", "Hydrophobic": "#999999",
    "π–π stacking": "#009E73", "Salt bridge": "#F0E442",
    "π–cation": "#D55E00", "Halogen bond": "#56B4E9",
    "Water bridge": "#CC79A7", "Metal coordination": "#E69F00",
    "Other": "#CCCCCC",
}
TYPE_ALIASES = {
    "hydrogen_bond": "Hydrogen bond", "hbond": "Hydrogen bond", "hydrogen bond": "Hydrogen bond",
    "hydrophobic_interaction": "Hydrophobic", "hydrophobic": "Hydrophobic", "hydrophobic interaction": "Hydrophobic",
    "pi_stack": "π–π stacking", "pi_stacking": "π–π stacking", "pi-stacking": "π–π stacking", "π–π stacking": "π–π stacking",
    "salt_bridge": "Salt bridge", "saltbridge": "Salt bridge", "salt bridge": "Salt bridge",
    "pi_cation_interaction": "π–cation", "cation_pi": "π–cation", "π–cation": "π–cation",
    "halogen_bond": "Halogen bond", "halogen bond": "Halogen bond",
    "water_bridge": "Water bridge", "water bridge": "Water bridge",
    "metal_complex": "Metal coordination", "metal coordination": "Metal coordination",
}


def _present(fig, name, show, save_dir):
    """Save (optional), then show or close, so notebooks never draw a figure twice."""
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _read(obj, filename):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if obj is None:
        return None
    obj = Path(obj)
    return pd.read_csv(obj / filename if obj.is_dir() else obj)


def _column(df, *names, required=True):
    actual = {str(k).strip().lower(): k for k in df.columns}
    for name in names:
        if name.lower() in actual:
            return actual[name.lower()]
    if required:
        raise ValueError(f"Missing column {names}; available: {list(df.columns)}")
    return None


def _residue_key(residue):
    match = re.search(r"(-?\d+)([A-Za-z]?)$", str(residue))
    chain = str(residue).split(":", 1)[0] if ":" in str(residue) else ""
    return (chain, int(match.group(1)) if match else 10**9, str(residue))


def _prepare(details, status=None, results=None):
    d = _read(details, "interaction_details.csv")
    if d is None:
        raise ValueError("interaction_details.csv is required")
    st = _read(status, "interaction_status.csv")
    results = _read(results, "hit_expander_boltz2_all_results.csv")
    job = _column(d, "job_id", "ligand_id", "ligand")
    itype = _column(d, "interaction", "interaction_type", "type")
    residue = _column(d, "residue", required=False)
    if residue:
        d["_residue"] = d[residue].astype(str)
    else:
        name = _column(d, "restype", "residue_name", "resname")
        number = _column(d, "resnr", "residue_number", "resnum")
        chain = _column(d, "reschain", "residue_chain", "chain", required=False)
        d["_residue"] = ((d[chain].astype(str) + ":") if chain else "") + d[name].astype(str) + d[number].astype(str)
    d["_job"] = d[job].astype(str)
    d["_type"] = d[itype].fillna("Other").astype(str).map(lambda x: TYPE_ALIASES.get(x.strip().lower(), x.strip() if x.strip() in COLORS else "Other"))
    if st is not None:
        sj = _column(st, "job_id", "ligand_id", "ligand")
        s = _column(st, "status", required=False)
        if s:
            failed = st.loc[st[s].astype(str).str.lower() != "ok", sj].astype(str).tolist()
            if failed:
                print(f"Excluded {len(failed)} failed PLIP analyses: {', '.join(failed[:8])}")
            d = d.loc[~d._job.isin(failed)].copy()
        jobs = st.loc[st[s].astype(str).str.lower() == "ok", sj].astype(str).tolist() if s else st[sj].astype(str).tolist()
    else:
        jobs = d._job.drop_duplicates().tolist()
    jobs = list(dict.fromkeys(jobs))
    p = {}
    if results is not None:
        pj = _column(results, "job_id", "ligand_id", "ligand")
        pc = _column(results, "predicted_pic50", "pic50", required=False)
        if pc:
            p = dict(zip(results[pj].astype(str), pd.to_numeric(results[pc], errors="coerce")))
    return d, jobs, p


def _order(d, jobs, potencies, sort):
    if sort == "name":
        return sorted(jobs)
    if sort == "interaction_count":
        cnt = d.groupby("_job").size()
        return sorted(jobs, key=lambda j: (-int(cnt.get(j, 0)), j))
    if sort == "predicted_pic50":
        if not potencies:
            print("Potency data not provided: sorting ligands by name instead.")
            return sorted(jobs)
        return sorted(jobs, key=lambda j: (-float(potencies[j]) if j in potencies and pd.notna(potencies[j]) else float("inf"), j))
    raise ValueError("ligand_sort must be 'name', 'interaction_count', or 'predicted_pic50'")


def _residues(d, sort):
    unique = d._residue.drop_duplicates().tolist()
    if sort == "sequence":
        return sorted(unique, key=_residue_key)
    if sort == "frequency":
        freq = d.groupby("_residue")._job.nunique()
        return sorted(unique, key=lambda r: (-int(freq[r]), _residue_key(r)))
    raise ValueError("residue_sort must be 'sequence' or 'frequency'")


def plot_interaction_fingerprint(d, jobs, *, residue_sort="sequence", max_residues=None, show=True, save_dir=None):
    residues = _residues(d, residue_sort)
    if max_residues is not None:
        residues = residues[:max_residues]
    grouped = d.groupby(["_job", "_residue"])["_type"].agg(lambda x: sorted(set(x), key=lambda t: list(COLORS).index(t))).to_dict()
    fig, ax = plt.subplots(figsize=(max(9, len(residues)*.46+2), max(4, len(jobs)*.4+1.5)))
    for i, job in enumerate(jobs):
        for j, res in enumerate(residues):
            types = grouped.get((job,res), [])
            ax.add_patch(Rectangle((j,i),1,1,facecolor="white",edgecolor="#dddddd",linewidth=.6))
            for k, kind in enumerate(types):
                ax.add_patch(Rectangle((j,i+k/len(types)),1,1/len(types),facecolor=COLORS[kind],edgecolor="none"))
    ax.set_xlim(0,len(residues)); ax.set_ylim(len(jobs),0)
    ax.set_xticks(np.arange(len(residues))+.5, residues, rotation=90)
    ax.set_yticks(np.arange(len(jobs))+.5, jobs)
    ax.xaxis.tick_top(); ax.set_title(f"Protein–ligand interaction fingerprint ({len(jobs)} ligands, {len(residues)} residues)",pad=28)
    present = set(d._type)
    ax.legend(handles=[Patch(color=c,label=t) for t,c in COLORS.items() if t in present],
              loc="upper left",bbox_to_anchor=(1.01,1),title="Interaction")
    fig.tight_layout(); _present(fig, "interaction_fingerprint", show, save_dir)
    return fig


def plot_interaction_counts(d, jobs, show=True, save_dir=None):
    counts = pd.crosstab(d._job,d._type).reindex(jobs,fill_value=0)
    fig, ax = plt.subplots(figsize=(max(9,len(jobs)*.55),5))
    bottom = np.zeros(len(jobs))
    for kind,color in COLORS.items():
        if kind in counts:
            values = counts[kind].to_numpy()
            ax.bar(range(len(jobs)),values,bottom=bottom,color=color,label=kind)
            bottom += values
    ax.set_xticks(range(len(jobs)),jobs,rotation=90)
    ax.set_ylabel("Detected interaction records");ax.set_title("Interactions by ligand")
    ax.legend(bbox_to_anchor=(1.01,1),loc="upper left")
    fig.tight_layout();_present(fig, "interaction_counts", show, save_dir)
    return fig


def plot_residue_conservation(d, jobs, show=True, save_dir=None):
    freq = d.groupby("_residue")._job.nunique().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(max(9,len(freq)*.4),4.5))
    ax.bar(range(len(freq)),freq.to_numpy())
    ax.set_xticks(range(len(freq)),freq.index,rotation=90)
    ax.set_ylim(0,max(len(jobs),1));ax.set_ylabel("Ligands with interaction")
    ax.set_title("Residue conservation across analyzed ligands")
    fig.tight_layout();_present(fig, "residue_conservation", show, save_dir)
    return fig


def plot_similarity(d, jobs, show=True, save_dir=None):
    features={j:set() for j in jobs}
    for job,res,typ in d[["_job","_residue","_type"]].itertuples(index=False,name=None):
        if job in features:
            features[job].add((res,typ))
    sim=np.full((len(jobs),len(jobs)),np.nan)
    for i,a in enumerate(jobs):
        for j,b in enumerate(jobs):
            union=features[a] | features[b]
            if union:
                sim[i,j]=len(features[a]&features[b])/len(union)
            elif i==j:
                sim[i,j]=1.0
    fig,ax=plt.subplots(figsize=(max(6,len(jobs)*.47),max(5,len(jobs)*.47)))
    im=ax.imshow(sim,vmin=0,vmax=1)
    ax.set_xticks(range(len(jobs)),jobs,rotation=90)
    ax.set_yticks(range(len(jobs)),jobs)
    ax.set_title("Interaction fingerprint similarity (Tanimoto)")
    fig.colorbar(im,ax=ax,label="Tanimoto similarity")
    fig.tight_layout();_present(fig, "interaction_similarity", show, save_dir)
    return fig


def plot_interaction_summary(details="hit_expander_output/interaction_analysis", *, status=None,
                             results=None, ligand_sort="name", residue_sort="sequence",
                             max_residues=None, show=True, save_dir=None):
    """Display 4 Matplotlib figures in Jupyter. Return their Figure objects.

    show=False suppresses display; save_dir writes each figure as a 200-dpi PNG.

    details: DataFrame, CSV file, or interaction-analysis directory.
    status: optional DataFrame/CSV; autodetected when details is a directory.
    results: optional Boltz results DataFrame/CSV to enable pIC50 sorting.
    """
    if status is None and not isinstance(details,pd.DataFrame) and Path(details).is_dir():
        candidate=Path(details)/"interaction_status.csv"
        if candidate.exists():
            status=candidate
    d,jobs,potencies=_prepare(details,status,results)
    if not jobs:
        raise ValueError("No successful ligand analyses to plot")
    jobs=_order(d,jobs,potencies,ligand_sort)
    if d.empty:
        raise ValueError("There are no interactions to plot; inspect interaction_status.csv and PLIP results")
    return {
        "fingerprint":plot_interaction_fingerprint(d,jobs,residue_sort=residue_sort,max_residues=max_residues,show=show,save_dir=save_dir),
        "counts":plot_interaction_counts(d,jobs,show=show,save_dir=save_dir),
        "conservation":plot_residue_conservation(d,jobs,show=show,save_dir=save_dir),
        "similarity":plot_similarity(d,jobs,show=show,save_dir=save_dir),
    }
