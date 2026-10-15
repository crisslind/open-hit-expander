from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence
import math
import re
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
# Replace this example with one SMILES, several SMILES, or a file path.




ALL_ADDITIONS = (
    "Me", "F", "Cl", "Br", "OH", "OMe", "NH2_aromatic", "NHMe_aromatic"
)
ALL_AROMATIC_SWAPS = (
    "C_to_N", "N_to_C", "N_to_O", "O_to_N",
    "O_to_S", "S_to_O", "N_to_S", "S_to_N",
)


@dataclass(frozen=True)
class HitExpanderConfig:
    # Empty by default: choose transformations explicitly with the widget below,
    # or set these tuples when calling the Python API directly.
    additions: tuple[str, ...] = ()
    aromatic_swaps: tuple[str, ...] = ()

    # Apply transformations only at these zero-based atom indices. None means all
    # eligible atoms. For multiple parents, use {"mol_1": [0, 2], "name": [4]}.
    selected_atoms: dict[str, Sequence[int]] | None = None

    # Structural proximity to the parent.
    min_morgan_tanimoto: float = 0.50
    max_heavy_atom_delta: int = 4

    # Simple drug-like bounds. Set a bound to None to disable it.
    min_mw: float | None = 100.0
    max_mw: float | None = 600.0
    min_clogp: float | None = -2.0
    max_clogp: float | None = 6.0
    max_tpsa: float | None = 160.0
    max_hbd: int | None = 5
    max_hba: int | None = 10
    max_rotatable_bonds: int | None = 12
    max_abs_formal_charge: int | None = 2

    # Strict rejects every matching analogue. "new_only" rejects only alerts not
    # already present in the corresponding parent. Use () to disable alerts.
    reject_alert_catalogs: tuple[str, ...] = ("PAINS", "BRENK")
    alert_policy: str = "strict"  # "strict" or "new_only"

    # Molecule normalization and composition.
    keep_largest_fragment: bool = True
    uncharge: bool = True
    allowed_atomic_numbers: tuple[int, ...] = (
        5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53
    )

    max_analogs_per_parent: int = 1000
    generate_2d_coordinates: bool = True



FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ADDITION_FRAGMENTS = {
    "Me": "[*]C",
    "F": "[*]F",
    "Cl": "[*]Cl",
    "Br": "[*]Br",
    "OH": "[*]O",
    "OMe": "[*]OC",
    "NH2_aromatic": "[*]N",
    "NHMe_aromatic": "[*]NC",
}

CATALOG_ENUMS = {
    "PAINS": FilterCatalogParams.FilterCatalogs.PAINS,
    "BRENK": FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalogParams.FilterCatalogs.NIH,
    "ZINC": FilterCatalogParams.FilterCatalogs.ZINC,
    "CHEMBL": FilterCatalogParams.FilterCatalogs.CHEMBL,
}

AROMATIC_SWAP_SOURCES = {
    "C_to_N": 6, "N_to_C": 7, "N_to_O": 7, "O_to_N": 8,
    "O_to_S": 8, "S_to_O": 16, "N_to_S": 7, "S_to_N": 16,
}


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _standardize_mol(mol: Chem.Mol, config: HitExpanderConfig) -> Chem.Mol:
    """Return a fully sanitized, standardized copy or raise an exception."""
    if mol is None:
        raise ValueError("RDKit could not parse the molecule")
    out = Chem.Mol(mol)
    Chem.SanitizeMol(out)
    out = rdMolStandardize.Cleanup(out)
    if config.keep_largest_fragment:
        out = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(out)
    if config.uncharge:
        out = rdMolStandardize.Uncharger().uncharge(out)
    Chem.SanitizeMol(out)
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)

    disallowed = sorted(
        {a.GetAtomicNum() for a in out.GetAtoms()}
        - set(config.allowed_atomic_numbers)
    )
    if disallowed:
        symbols = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in disallowed]
        raise ValueError("disallowed element(s): " + ", ".join(symbols))
    if any(a.GetNumRadicalElectrons() for a in out.GetAtoms()):
        raise ValueError("radical atom present")
    if len(Chem.GetMolFrags(out)) != 1:
        raise ValueError("more than one disconnected component remains")
    return out


def _parse_smiles_record(smiles: str, name: str, config: HitExpanderConfig):
    raw = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = _standardize_mol(raw, config)
    mol.SetProp("_Name", name)
    return mol


def load_input(
    value: str | Path | Sequence[str], config: HitExpanderConfig
) -> tuple[list[dict], pd.DataFrame]:
    """Load and validate SMILES/SDF input without silently dropping failures."""
    raw_records: list[tuple[str, str, Chem.Mol | None]] = []

    if isinstance(value, Path) or (
        isinstance(value, str) and Path(value).expanduser().is_file()
    ):
        path = Path(value).expanduser()
        suffix = path.suffix.lower()
        if suffix in {".sdf", ".sd"}:
            supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
            for i, mol in enumerate(supplier, 1):
                name = mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else f"mol_{i}"
                raw_records.append((name or f"mol_{i}", "SDF record", mol))
        elif suffix in {".smi", ".smiles", ".txt"}:
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = re.split(r"\s+", line, maxsplit=1)
                smi = fields[0]
                name = fields[1] if len(fields) == 2 else f"mol_{line_number}"
                raw_records.append((name, smi, Chem.MolFromSmiles(smi, sanitize=False)))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    else:
        smiles_values = [value] if isinstance(value, str) else list(value)
        for i, smi in enumerate(smiles_values, 1):
            if not isinstance(smi, str):
                raw_records.append((f"mol_{i}", repr(smi), None))
            else:
                raw_records.append((f"mol_{i}", smi, Chem.MolFromSmiles(smi, sanitize=False)))

    valid: list[dict] = []
    failures: list[dict] = []
    used_ids: set[str] = set()
    for ordinal, (suggested_name, source_text, raw_mol) in enumerate(raw_records, 1):
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", suggested_name).strip("_") or f"mol_{ordinal}"
        source_id = base
        counter = 2
        while source_id in used_ids:
            source_id = f"{base}_{counter}"
            counter += 1
        used_ids.add(source_id)
        try:
            mol = _standardize_mol(raw_mol, config)
            mol.SetProp("_Name", source_id)
            valid.append({
                "source_id": source_id,
                "mol": mol,
                "source_smiles": canonical_smiles(mol),
            })
        except Exception as exc:
            failures.append({
                "source_id": source_id,
                "input": source_text,
                "reason": f"invalid input: {type(exc).__name__}: {exc}",
            })

    return valid, pd.DataFrame(failures, columns=["source_id", "input", "reason"])


def _selected(source_id: str, atom_index: int, config: HitExpanderConfig) -> bool:
    if config.selected_atoms is None:
        return True
    allowed = config.selected_atoms.get(source_id)
    return allowed is not None and atom_index in set(allowed)


def _attach_fragment(parent: Chem.Mol, target_index: int, fragment_smiles: str) -> Chem.Mol:
    """Replace one implicit/explicit H on target atom with a one-attachment fragment."""
    fragment = Chem.MolFromSmiles(fragment_smiles)
    dummy_atoms = [a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        raise ValueError("fragment must contain exactly one dummy attachment atom")
    dummy_index = dummy_atoms[0]
    dummy = fragment.GetAtomWithIdx(dummy_index)
    if dummy.GetDegree() != 1:
        raise ValueError("dummy atom must have exactly one neighbor")
    anchor_index = dummy.GetNeighbors()[0].GetIdx()
    attachment_bond = fragment.GetBondBetweenAtoms(dummy_index, anchor_index).GetBondType()

    frag_rw = Chem.RWMol(fragment)
    frag_rw.RemoveAtom(dummy_index)
    if anchor_index > dummy_index:
        anchor_index -= 1
    fragment_without_dummy = frag_rw.GetMol()
    Chem.SanitizeMol(fragment_without_dummy)

    parent_copy = Chem.Mol(parent)
    target = parent_copy.GetAtomWithIdx(target_index)
    if target.GetNumExplicitHs() > 0:
        target.SetNumExplicitHs(target.GetNumExplicitHs() - 1)

    combined = Chem.CombineMols(parent_copy, fragment_without_dummy)
    rw = Chem.RWMol(combined)
    rw.AddBond(
        target_index,
        parent_copy.GetNumAtoms() + anchor_index,
        attachment_bond,
    )
    product = rw.GetMol()
    Chem.SanitizeMol(product)
    return product


def _swap_aromatic_atom(parent: Chem.Mol, atom_index: int, transformation: str) -> Chem.Mol:
    rw = Chem.RWMol(parent)
    atom = rw.GetAtomWithIdx(atom_index)
    if not atom.GetIsAromatic():
        raise ValueError("target is not aromatic")

    # atomic number, formal charge, explicit H count, allow implicit H
    specifications = {
        "C_to_N": (7, 0, 0, False),   # [n]
        "N_to_C": (6, 0, 0, True),    # [cH] after sanitization
        "N_to_O": (8, 0, 0, False),   # [o]
        "O_to_N": (7, 0, 1, False),   # [nH]
        "O_to_S": (16, 0, 0, False),  # [s]
        "S_to_O": (8, 0, 0, False),   # [o]
        "N_to_S": (16, 0, 0, False),  # [s]
        "S_to_N": (7, 0, 1, False),   # [nH]
    }
    if atom.GetAtomicNum() != AROMATIC_SWAP_SOURCES[transformation]:
        raise ValueError("atom element does not match transformation")

    atomic_number, charge, explicit_h, allow_implicit = specifications[transformation]
    atom.SetAtomicNum(atomic_number)
    atom.SetFormalCharge(charge)
    atom.SetNumExplicitHs(explicit_h)
    atom.SetNoImplicit(not allow_implicit)
    product = rw.GetMol()
    product.UpdatePropertyCache(strict=False)
    # Expected invalid aromatic electron-count combinations are captured in the
    # audit table instead of emitted as noisy RDKit warnings.
    with rdBase.BlockLogs():
        Chem.SanitizeMol(product)
    return product


def _enumerate_raw(parent: Chem.Mol, source_id: str, config: HitExpanderConfig):
    """Enumerate one modification at one eligible aromatic position."""
    valid_attempts = []
    invalid_attempts = []

    for atom in parent.GetAtoms():
        atom_index = atom.GetIdx()
        if not _selected(source_id, atom_index, config):
            continue

        # Additions are strictly limited to aromatic C-H vectors. Aliphatic atoms
        # are never enumerated, even when this function is called without widgets.
        if (
            atom.GetIsAromatic()
            and atom.GetAtomicNum() == 6
            and atom.GetTotalNumHs(includeNeighbors=True) > 0
        ):
            for transformation in config.additions:
                try:
                    product = _attach_fragment(
                        parent, atom_index, ADDITION_FRAGMENTS[transformation]
                    )
                    valid_attempts.append((
                        product, f"add_{transformation}", (atom_index,)
                    ))
                except Exception as exc:
                    invalid_attempts.append({
                        "source_id": source_id,
                        "transformation": f"add_{transformation}",
                        "atom_index": atom_index,
                        "reason": f"invalid generated structure: {type(exc).__name__}: {exc}",
                    })

        if atom.GetIsAromatic():
            for transformation in config.aromatic_swaps:
                if atom.GetAtomicNum() != AROMATIC_SWAP_SOURCES[transformation]:
                    continue
                try:
                    product = _swap_aromatic_atom(parent, atom_index, transformation)
                    valid_attempts.append((product, transformation, (atom_index,)))
                except Exception as exc:
                    invalid_attempts.append({
                        "source_id": source_id,
                        "transformation": transformation,
                        "atom_index": atom_index,
                        "reason": f"invalid generated structure: {type(exc).__name__}: {exc}",
                    })

    return valid_attempts, invalid_attempts


def _build_alert_catalogs(names: Iterable[str]) -> dict[str, FilterCatalog]:
    catalogs = {}
    for name in names:
        key = name.upper()
        if key not in CATALOG_ENUMS:
            raise ValueError(f"Unknown alert catalog {name!r}; choose from {sorted(CATALOG_ENUMS)}")
        params = FilterCatalogParams()
        params.AddCatalog(CATALOG_ENUMS[key])
        catalogs[key] = FilterCatalog(params)
    return catalogs


def _alert_matches(mol: Chem.Mol, catalogs: dict[str, FilterCatalog]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for name, catalog in catalogs.items():
        descriptions = sorted({entry.GetDescription() for entry in catalog.GetMatches(mol)})
        if descriptions:
            matches[name] = descriptions
    return matches


def _properties(mol: Chem.Mol) -> dict[str, float | int]:
    return {
        "mw": round(Descriptors.MolWt(mol), 3),
        "clogp": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }


def _bound_reasons(props: dict, config: HitExpanderConfig) -> list[str]:
    checks = [
        (config.min_mw is not None and props["mw"] < config.min_mw, f"MW < {config.min_mw}"),
        (config.max_mw is not None and props["mw"] > config.max_mw, f"MW > {config.max_mw}"),
        (config.min_clogp is not None and props["clogp"] < config.min_clogp, f"cLogP < {config.min_clogp}"),
        (config.max_clogp is not None and props["clogp"] > config.max_clogp, f"cLogP > {config.max_clogp}"),
        (config.max_tpsa is not None and props["tpsa"] > config.max_tpsa, f"TPSA > {config.max_tpsa}"),
        (config.max_hbd is not None and props["hbd"] > config.max_hbd, f"HBD > {config.max_hbd}"),
        (config.max_hba is not None and props["hba"] > config.max_hba, f"HBA > {config.max_hba}"),
        (
            config.max_rotatable_bonds is not None
            and props["rotatable_bonds"] > config.max_rotatable_bonds,
            f"rotatable bonds > {config.max_rotatable_bonds}",
        ),
        (
            config.max_abs_formal_charge is not None
            and abs(props["formal_charge"]) > config.max_abs_formal_charge,
            f"|formal charge| > {config.max_abs_formal_charge}",
        ),
    ]
    return [message for failed, message in checks if failed]


def run_hit_expander(
    input_value: str | Path | Sequence[str],
    config: HitExpanderConfig | None = None,
):
    if config is None:
        config = HitExpanderConfig()
    if config.alert_policy not in {"strict", "new_only"}:
        raise ValueError("alert_policy must be 'strict' or 'new_only'")

    parents, input_failures = load_input(input_value, config)
    if not parents:
        raise ValueError("No valid input molecules were loaded")

    catalogs = _build_alert_catalogs(config.reject_alert_catalogs)
    parent_alerts = {
        p["source_id"]: _alert_matches(p["mol"], catalogs) for p in parents
    }
    input_smiles = {p["source_smiles"] for p in parents}

    unique_candidates: dict[tuple[str, str], dict] = {}
    invalid_generation: list[dict] = []
    attempt_counts: dict[str, int] = {}

    for parent in parents:
        source_id = parent["source_id"]
        attempts, invalid = _enumerate_raw(parent["mol"], source_id, config)
        attempt_counts[source_id] = len(attempts) + len(invalid)
        invalid_generation.extend(invalid)

        for product, transformation, atom_indices in attempts:
            try:
                standardized = _standardize_mol(product, config)
                smiles = canonical_smiles(standardized)
            except Exception as exc:
                invalid_generation.append({
                    "source_id": source_id,
                    "transformation": transformation,
                    "atom_index": ";".join(map(str, atom_indices)),
                    "reason": f"failed post-generation standardization: {type(exc).__name__}: {exc}",
                })
                continue

            key = (source_id, smiles)
            if key not in unique_candidates:
                unique_candidates[key] = {
                    "mol": standardized,
                    "source_id": source_id,
                    "analog_smiles": smiles,
                    "routes": set(),
                }
            unique_candidates[key]["routes"].add(
                (transformation, tuple(atom_indices))
            )

    parent_by_id = {p["source_id"]: p for p in parents}
    rows = []
    mol_by_key: dict[tuple[str, str], Chem.Mol] = {}

    for (source_id, smiles), candidate in unique_candidates.items():
        parent = parent_by_id[source_id]
        mol = candidate["mol"]
        props = _properties(mol)
        routes = sorted(candidate["routes"], key=lambda item: (item[1], item[0]))
        primary_transformation, primary_atom_indices = routes[0]
        similarity = DataStructs.TanimotoSimilarity(
            FP_GENERATOR.GetFingerprint(parent["mol"]),
            FP_GENERATOR.GetFingerprint(mol),
        )
        heavy_atom_delta = props["heavy_atoms"] - parent["mol"].GetNumHeavyAtoms()
        alerts = _alert_matches(mol, catalogs)

        reasons = []
        if smiles in input_smiles:
            reasons.append("identical to an input molecule")
        if similarity < config.min_morgan_tanimoto:
            reasons.append(f"Morgan Tanimoto < {config.min_morgan_tanimoto}")
        if abs(heavy_atom_delta) > config.max_heavy_atom_delta:
            reasons.append(f"|heavy-atom delta| > {config.max_heavy_atom_delta}")
        reasons.extend(_bound_reasons(props, config))

        if alerts:
            if config.alert_policy == "strict":
                alert_reasons = alerts
            else:
                baseline = parent_alerts[source_id]
                alert_reasons = {
                    catalog: sorted(set(names) - set(baseline.get(catalog, [])))
                    for catalog, names in alerts.items()
                }
                alert_reasons = {k: v for k, v in alert_reasons.items() if v}
            for catalog, descriptions in alert_reasons.items():
                reasons.append(f"{catalog}: " + ", ".join(descriptions))

        row = {
            "source_id": source_id,
            "source_smiles": parent["source_smiles"],
            "analog_smiles": smiles,
            "transformations": primary_transformation,
            "atom_indices": ";".join(map(str, primary_atom_indices)),
            "alternative_routes": " | ".join(
                f"{route_transformation} @ {','.join(map(str, route_indices))}"
                for route_transformation, route_indices in routes[1:]
            ),
            "morgan_tanimoto": round(float(similarity), 4),
            "heavy_atom_delta": int(heavy_atom_delta),
            **props,
            "structural_alerts": "; ".join(
                f"{catalog}: {', '.join(names)}" for catalog, names in alerts.items()
            ),
            "passed": not reasons,
            "rejection_reasons": "; ".join(reasons),
        }
        rows.append(row)
        mol_by_key[(source_id, smiles)] = mol

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            ["source_id", "passed", "morgan_tanimoto", "analog_smiles"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

        passed = results.loc[results["passed"]].copy().reset_index(drop=True)
        rejected = results.loc[~results["passed"]].copy().reset_index(drop=True)
    else:
        passed = results.copy()
        rejected = results.copy()

    invalid_df = pd.DataFrame(invalid_generation)
    summary_rows = []
    for parent in parents:
        source_id = parent["source_id"]
        source_results = results.loc[results["source_id"] == source_id] if not results.empty else results
        summary_rows.append({
            "source_id": source_id,
            "source_smiles": parent["source_smiles"],
            "attempted_transformations": attempt_counts.get(source_id, 0),
            "unique_valid_analogs": len(source_results),
            "passed": int(source_results["passed"].sum()) if not source_results.empty else 0,
            "filtered": int((~source_results["passed"]).sum()) if not source_results.empty else 0,
            "invalid_generation_attempts": (
                int((invalid_df["source_id"] == source_id).sum())
                if not invalid_df.empty else 0
            ),
            "parent_alerts": "; ".join(
                f"{catalog}: {', '.join(names)}"
                for catalog, names in parent_alerts[source_id].items()
            ),
        })

    if len(passed) > config.max_analogs_per_parent * len(parents):
        passed = (
            passed.sort_values("morgan_tanimoto", ascending=False)
            .groupby("source_id", as_index=False, group_keys=False)
            .head(config.max_analogs_per_parent)
            .reset_index(drop=True)
        )

    return {
        "parents": parents,
        "summary": pd.DataFrame(summary_rows),
        "passed": passed,
        "rejected": rejected,
        "invalid_inputs": input_failures,
        "invalid_generation": invalid_df,
        "mol_by_key": mol_by_key,
        "config": config,
    }


def export_results(run: dict, output_directory: str | Path = "hit_expander_output"):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    passed_csv = output_directory / "hit_expander_passed.csv"
    rejected_csv = output_directory / "hit_expander_rejected.csv"
    invalid_csv = output_directory / "hit_expander_invalid_generation.csv"
    summary_csv = output_directory / "hit_expander_summary.csv"
    sdf_path = output_directory / "hit_expander_passed.sdf"

    run["passed"].to_csv(passed_csv, index=False)
    run["rejected"].to_csv(rejected_csv, index=False)
    run["invalid_generation"].to_csv(invalid_csv, index=False)
    run["summary"].to_csv(summary_csv, index=False)

    writer = Chem.SDWriter(str(sdf_path))
    for row_number, row in run["passed"].iterrows():
        key = (row["source_id"], row["analog_smiles"])
        mol = Chem.Mol(run["mol_by_key"][key])
        if run["config"].generate_2d_coordinates:
            rdMolDescriptors.CalcNumRings(mol)  # initialize ring information explicitly
            Chem.rdDepictor.Compute2DCoords(mol)
        mol.SetProp("_Name", f"{row['source_id']}_analog_{row_number + 1}")
        for column, value in row.items():
            if pd.notna(value):
                mol.SetProp(str(column), str(value))
        writer.write(mol)
    writer.close()

    return {
        "passed_sdf": sdf_path,
        "passed_csv": passed_csv,
        "rejected_csv": rejected_csv,
        "invalid_generation_csv": invalid_csv,
        "summary_csv": summary_csv,
    }


def draw_passed(run: dict, max_molecules: int = 24, mols_per_row: int = 4):
    table = run["passed"].head(max_molecules)
    if table.empty:
        print("No analogues passed the configured filters.")
        return None
    mols, legends = [], []
    for _, row in table.iterrows():
        mols.append(Chem.Mol(run["mol_by_key"][(row["source_id"], row["analog_smiles"])]))
        legends.append(
            f"{row['source_id']} | {row['transformations']} | T={row['morgan_tanimoto']:.2f}"
        )
    return Draw.MolsToGridImage(
        mols, legends=legends, molsPerRow=mols_per_row, subImgSize=(300, 230)
    )