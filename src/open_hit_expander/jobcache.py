"""Content-addressed job keys and a cross-run cache of finished Boltz-2 predictions.

Nothing here imports RDKit or Boltz, so it is cheap to import and easy to test.

Why content-addressed?  Job IDs such as ``mol_1_analogue_0007`` depend on the
row order of one particular enumeration, so they change when you tweak a filter.
A key built from (protein sequence, ligand SMILES, seed, prediction settings)
identifies the *prediction itself* and therefore survives re-enumeration.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml

# Flags that change where or how fast a prediction runs, not what it predicts.
_FLAGS_WITH_VALUE = {
    "--out_dir", "--seed", "--accelerator", "--devices", "--num_workers",
    "--preprocessing-threads", "--preprocessing_threads",
}
_BARE_FLAGS = {"--no_kernels", "--use_msa_server"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sequence_key(sequence: str) -> str:
    """Stable short key for a protein sequence (whitespace/case insensitive)."""
    return _sha(re.sub(r"\s+", "", sequence).upper())[:16]


def ligand_key(smiles: str) -> str:
    """Stable short key for a ligand. Pass canonical SMILES for best reuse."""
    return _sha(str(smiles).strip())[:12]


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def prediction_settings(command) -> list[str]:
    """Prediction-relevant tokens of a ``boltz predict`` command.

    Drops the executable, sub-command, input directory, output directory, seed
    and hardware flags, which affect speed or location but not the result.
    """
    tokens = [str(t) for t in command]
    kept: list[str] = []
    skip_next = False
    for token in tokens[3:]:  # [executable, "predict", input_dir, ...]
        if skip_next:
            skip_next = False
            continue
        if token in _FLAGS_WITH_VALUE:
            skip_next = True
            continue
        if token in _BARE_FLAGS:
            continue
        kept.append(token)
    return kept


def cache_key(sequence: str, smiles: str, seed: int, command, msa_tag: str = "server") -> str:
    """Key identifying one prediction.

    ``msa_tag`` is "server" for server-generated MSAs (a deterministic function
    of the sequence, up to database updates) or a file digest for a user MSA.
    """
    payload = {
        "protein": sequence_key(sequence),
        "ligand": ligand_key(smiles),
        "seed": int(seed),
        "settings": prediction_settings(command),
        "msa": msa_tag,
    }
    return _sha(json.dumps(payload, sort_keys=True))


# ----------------------------------------------------------------- YAML helpers
def protein_sequence_from_job(yaml_path: str | Path) -> str:
    document = yaml.safe_load(Path(yaml_path).read_text())
    for entry in document.get("sequences", []):
        if "protein" in entry:
            return str(entry["protein"]["sequence"])
    raise ValueError(f"No protein entry in {yaml_path}")


def job_msa_path(yaml_path: str | Path) -> str | None:
    document = yaml.safe_load(Path(yaml_path).read_text())
    for entry in document.get("sequences", []):
        if "protein" in entry:
            return entry["protein"].get("msa")
    return None


def set_job_msa(yaml_path: str | Path, msa_path: str | Path) -> None:
    """Point the protein entry of a job YAML at a precomputed MSA (in place)."""
    path = Path(yaml_path)
    document = yaml.safe_load(path.read_text())
    for entry in document.get("sequences", []):
        if "protein" in entry:
            entry["protein"]["msa"] = str(Path(msa_path).expanduser().resolve())
            break
    else:
        raise ValueError(f"No protein entry in {yaml_path}")
    path.write_text(yaml.safe_dump(document, sort_keys=False))


# -------------------------------------------------------- prediction locations
def prediction_directory(output_dir: str | Path, job_id: str, *, input_name: str = "jobs") -> Path:
    """Where Boltz wrote (or would write) the prediction for ``job_id``.

    Boltz nests results under ``boltz_results_<input directory name>/predictions``;
    older runs may use ``predictions`` directly. Both are searched.
    """
    output_dir = Path(output_dir)
    direct = output_dir / "predictions" / job_id
    if direct.is_dir():
        return direct
    for predictions in sorted(output_dir.glob("boltz_results_*/predictions")):
        if (predictions / job_id).is_dir():
            return predictions / job_id
    return output_dir / f"boltz_results_{input_name}" / "predictions" / job_id


# ---------------------------------------------------------------- global cache
class CofoldCache:
    """Directory-backed store of finished predictions, keyed by :func:`cache_key`."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _entry(self, key: str) -> Path:
        return self.root / key[:2] / key

    def has(self, key: str) -> bool:
        return (self._entry(key) / "meta.json").is_file()

    def store(self, key: str, prediction_dir: str | Path, job_id: str) -> bool:
        """Copy a *successful* prediction directory in. Returns False if present."""
        if self.has(key):
            return False
        entry = self._entry(key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        staging = entry.with_name(entry.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(prediction_dir, staging / "prediction")
        (staging / "meta.json").write_text(json.dumps({"job_id": job_id}))
        shutil.rmtree(entry, ignore_errors=True)
        staging.rename(entry)
        return True

    def restore(self, key: str, destination: str | Path, job_id: str) -> bool:
        """Copy a cached prediction to ``destination``, renaming files to ``job_id``."""
        if not self.has(key):
            return False
        entry = self._entry(key)
        old_id = json.loads((entry / "meta.json").read_text())["job_id"]
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for source in (entry / "prediction").iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name.replace(old_id, job_id))
        return True
