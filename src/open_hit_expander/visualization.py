"""Browse locally saved Boltz complexes in Jupyter without rerunning inference."""
from pathlib import Path
import html
import json
import re
import pandas as pd
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display


def resolve_structure_path(value, structure_root=None):
    """Resolve a saved path or uniquely relocate its job/model under a new root."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if path.is_file() and path.suffix.lower() in {".cif", ".mmcif", ".pdb"}:
        return path.resolve()
    if structure_root is None:
        return None
    root = Path(structure_root).expanduser()
    matches = [p for p in root.rglob(path.name)
               if p.is_file() and p.parent.name == path.parent.name]
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {path.parent.name}/{path.name}; use a narrower structure_root")
    return matches[0].resolve() if matches else None


def available_complexes(results, structure_root=None):
    """Keep structures even when affinity/confidence is unavailable."""
    if not isinstance(results, pd.DataFrame):
        raise ValueError("Pass a results DataFrame or load an exported all-results CSV first")
    if not {"job_id", "structure_path"}.issubset(results.columns):
        raise ValueError("Results require job_id and structure_path columns")
    frame = results.copy()
    frame["resolved_structure_path"] = [resolve_structure_path(p, structure_root)
                                        for p in frame["structure_path"]]
    frame = frame.loc[frame["resolved_structure_path"].notna()]
    if "predicted_pic50" in frame:
        frame = frame.sort_values("predicted_pic50", ascending=False, na_position="last", kind="stable")
    return frame.reset_index(drop=True)


def structure_models(path):
    """Return selected model first, followed by other models from the same job."""
    path = Path(path)
    stem = re.sub(r"_model_\d+$", "", path.stem)
    others = sorted(p for p in path.parent.iterdir()
                    if p.suffix.lower() in {".cif", ".mmcif", ".pdb"}
                    and re.fullmatch(re.escape(stem) + r"_model_\d+", p.stem))
    return [path] + [p for p in others if p != path]


def render_complex(path, ligand_chain="B", protein_chain="A", focus_ligand=True,
                   show_pocket=True, pocket_radius=5.0, width=640, height=500):
    """Build a py3Dmol view with element-colored heteroatoms and tinted carbons."""
    import py3Dmol
    path = Path(path)
    if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("Viewer supports PDB and mmCIF files")
    view = py3Dmol.view(width=width, height=height)
    view.addModel(path.read_text(), "pdb" if path.suffix.lower() == ".pdb" else "cif")
    protein = {"chain": protein_chain}
    ligand = {"chain": ligand_chain}
    view.setBackgroundColor("white")
    view.setStyle(protein, {"cartoon": {"color": "lightsteelblue", "opacity": 0.8}})
    view.setStyle(ligand, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.22}})
    if show_pocket:
        pocket = {"chain": protein_chain, "byres": True,
                  "within": {"distance": float(pocket_radius), "sel": ligand}}
        view.addStyle(pocket, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.12}})
    view.zoomTo(ligand if focus_ligand else {})
    if focus_ligand:
        view.zoom(0.8)
    return view



def _read_structure(path):
    import gemmi
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"No coordinate models in {path}")
    return structure


def _ca_atoms(structure, protein_chain):
    atoms = {}
    for chain in structure[0]:
        if chain.name != protein_chain:
            continue
        for residue in chain:
            for atom in residue:
                if atom.name == "CA" and atom.element.name == "C":
                    key = (residue.seqid.num, residue.seqid.icode, residue.name)
                    atoms.setdefault(key, atom.pos.tolist())
    return atoms


def aligned_structure(path, reference_ca, protein_chain="A"):
    """Rigidly fit a complex to matching protein Cα atoms; transform all atoms."""
    import gemmi
    import numpy as np
    structure = _read_structure(path)
    ca = _ca_atoms(structure, protein_chain)
    common = sorted(set(ca) & set(reference_ca))
    if len(common) < 3:
        raise ValueError(f"{Path(path).name}: at least 3 matching protein Cα atoms are required for overlay")
    moving = np.array([ca[k] for k in common], dtype=float)
    fixed = np.array([reference_ca[k] for k in common], dtype=float)
    center_m, center_f = moving.mean(axis=0), fixed.mean(axis=0)
    if np.linalg.matrix_rank(moving - center_m) < 2 or np.linalg.matrix_rank(fixed - center_f) < 2:
        raise ValueError(f"{Path(path).name}: matching Cα atoms do not define an alignment")
    u, _, vt = np.linalg.svd((moving-center_m).T @ (fixed-center_f))
    correction = np.eye(3)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = u @ correction @ vt
    translation = center_f - center_m @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((moving @ rotation + translation-fixed)**2, axis=1))))
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    xyz = np.array(atom.pos.tolist()) @ rotation + translation
                    atom.pos = gemmi.Position(*map(float, xyz))
    return structure.make_mmcif_document().as_string(), rmsd, len(common)


def _best_replicate_structure(selected_path, job_id):
    """Locate highest-affinity replicate structure from existing output, without inference.

    The main results table holds a representative (median-nearest) structure.
    Per-replicate predictions are stored beside the run under replicate_predictions.csv.
    """
    path = Path(selected_path).resolve()
    run_root = next((parent for parent in path.parents
                     if (parent / "replicate_predictions.csv").is_file()), None)
    if run_root is None:
        return None, None
    try:
        table = pd.read_csv(run_root / "replicate_predictions.csv")
        # Use the same pIC50 convention as the results collector.
        if "predicted_pic50" not in table or "job_id" not in table:
            return None, None
        rows = table.loc[table["job_id"].astype(str) == str(job_id)].copy()
        rows["predicted_pic50"] = pd.to_numeric(rows["predicted_pic50"], errors="coerce")
        rows = rows.dropna(subset=["predicted_pic50"]).sort_values("predicted_pic50", ascending=False)
        for _, row in rows.iterrows():
            candidate = resolve_structure_path(row.get("structure_path"))
            if candidate is None:
                # Old CSV may contain missing paths; locate by seed and job name.
                seed = row.get("seed")
                if pd.notna(seed):
                    dirs = list((run_root / "replicates").glob(f"replicate_*_seed_{int(seed)}/results/*/predictions/{job_id}"))
                    for directory in dirs:
                        matches = sorted(directory.glob(f"{job_id}_model_*.cif"))
                        if matches:
                            candidate = matches[0]
                            break
            if candidate is not None and candidate.is_file():
                return candidate.resolve(), float(row["predicted_pic50"])
    except (OSError, ValueError, KeyError):
        return None, None
    return None, None


def browser_payload(results, structure_root=None, protein_chain="A", ligand_chain="B"):
    """Read and align saved models once; browser navigation performs no inference."""
    table = available_complexes(results, structure_root)
    if table.empty:
        return {"jobs": [], "proteinChain": protein_chain, "ligandChain": ligand_chain}
    reference = table.iloc[0]["resolved_structure_path"]
    reference_ca = _ca_atoms(_read_structure(reference), protein_chain)
    fields = ["job_id", "source_id", "transformation", "ligand_smiles", "predicted_pic50",
              "delta_pic50_vs_parent", "affinity_probability_binary", "ligand_iptm",
              "confidence_score", "cofold_status"]
    jobs = []
    cache = {}
    for _, row in table.iterrows():
        models = []
        for path in structure_models(row["resolved_structure_path"]):
            if str(path) not in cache:
                try:
                    data, rmsd, count = aligned_structure(path, reference_ca, protein_chain)
                    cache[str(path)] = {"name": path.name, "data": data, "format": "cif",
                                        "rmsd": rmsd, "caCount": count, "error": None}
                except Exception as exc:
                    cache[str(path)] = {"name": path.name, "error": str(exc)}
            models.append(cache[str(path)])
        # Prefer explicit metadata saved during aggregation. Keep the legacy
        # CSV lookup for results produced before the collector fix.
        best_path = resolve_structure_path(row.get("best_structure_path"), structure_root)
        best_pic50 = pd.to_numeric(row.get("best_pic50"), errors="coerce")
        if best_path is None or pd.isna(best_pic50):
            best_path, best_pic50 = _best_replicate_structure(
                row["resolved_structure_path"], row["job_id"]
            )
        best_model_index = None
        if best_path is not None:
            # Add separately aligned best replicate, keeping median pose selected by default.
            if all(str(best_path) != str(p) for p in structure_models(row["resolved_structure_path"])):
                if str(best_path) not in cache:
                    try:
                        data, rmsd, count = aligned_structure(best_path, reference_ca, protein_chain)
                        cache[str(best_path)] = {"name": best_path.name, "data": data, "format": "cif",
                                                 "rmsd": rmsd, "caCount": count, "error": None}
                    except Exception as exc:
                        cache[str(best_path)] = {"name": best_path.name, "error": str(exc)}
                models.append(cache[str(best_path)])
                if cache[str(best_path)].get("error") is None:
                    best_model_index = len(models) - 1
            else:
                best_model_index = next((i for i, m in enumerate(models) if m["name"] == best_path.name), None)
        metrics = {}
        for key in fields:
            value = row.get(key)
            metrics[key] = "—" if value is None or pd.isna(value) else (f"{value:.3f}" if isinstance(value, float) else str(value))
        jobs.append({"id": str(row["job_id"]), "parent": str(row.get("is_parent", "")).lower() == "true",
                     "metrics": metrics, "models": models, "bestModelIndex": best_model_index,
                     "bestPic50": best_pic50})
    return {"jobs": jobs, "proteinChain": protein_chain, "ligandChain": ligand_chain,
            "alignmentReference": str(table.iloc[0]["job_id"])}


def complex_browser(results, structure_root=None, ligand_chain="B", protein_chain="A"):
    """Persistent viewer; HTML template colors carbon per model and heteroatoms by element."""
    payload = browser_payload(results, structure_root, protein_chain, ligand_chain)
    if not payload["jobs"]:
        return widgets.VBox([widgets.HTML("No available complex structures. Parse a run or check structure paths.")])
    document = _browser_document(payload)
    # One frame owns both the controls and the WebGL viewer. It is never replaced
    # during navigation, so VS Code output isolation cannot discard its camera.
    frame = '<iframe title="Boltz complex browser" sandbox="allow-scripts allow-same-origin" style="border:0;width:100%;height:830px" srcdoc="' + html.escape(document, quote=True) + '"></iframe>'
    output = widgets.Output()
    output.append_display_data(HTML(frame))
    return widgets.VBox([output], layout=widgets.Layout(width="100%"))


def _browser_document(payload):
    template = (Path(__file__).parent / "complex_browser.html").read_text()
    # Escaping prevents molecule IDs or coordinate strings closing a script tag.
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace("__PAYLOAD__", encoded)
