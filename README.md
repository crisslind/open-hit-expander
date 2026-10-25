# Open Hit Expander + Boltz-2

Research workflow for RDKit-based analogue enumeration, CPU pre-filtering, Boltz-2 protein–ligand cofolding / predicted potency, PLIP interactions, and offline compound assessment. **Predictions are hypotheses for prioritisation, not measured potency or validated binding modes.**

## Installation

Requirements: Python 3.10–3.13, RDKit and the Python dependencies declared in `pyproject.toml`; configure the `boltz` executable separately for inference and PLIP for fresh interaction analysis.

```bash
python -m pip install -e '.[notebook]'
# Development
python -m pip install -e '.[dev]'
```

Use `Open_Hit_Expander_boltz2.ipynb` from the repository root. Avoid merging or replacing your existing project files without first backing up your current working repository.

## Notebook map

| # | Section | Runs in | Needs Boltz-2 |
|---|---------|---------|---------------|
| 1 | Configuration (all settings in one cell) | both | no |
| 2 | Environment check | both | no |
| 3 | Hit expansion | new | no |
| 4 | Pre-filter | new | no |
| 5 | Prepare batch and dry run | new | no |
| 6 | Cofolding replicates (resumable) | new | **yes — the only inference step** |
| 7 | Load and aggregate results | both | no |
| 8 | Pose viewer | both | no |
| 9 | Interaction analysis (PLIP) | new, or reload with `RUN_PLIP = True` | no |
| 10 | Interaction plots | both | no |
| 11 | Prediction stability | both | no |
| 12 | Compound assessment report | both | no |
| 13 | Files written | — | — |

## Existing run (no Boltz-2 inference)

In Section 1 of the notebook:

```python
RUN_MODE = "reload"
SAVED_TIMESTAMP = "YOUR_EXISTING_TIMESTAMP"
OUTPUT_DIRECTORY = Path("hit_expander_output")
```

Run the notebook from the top. Sections 3–6 print "Reload mode: … skipped", so the expensive inference step cannot run by accident. Sections 7–8 reload the saved results and open the pose viewer; sections 10–12 use previously saved PLIP results. PLIP is not rerun on reload unless you set `RUN_PLIP = True` in Section 9.

On reload, saved Boltz JSON/structure results are reparsed and the analysis tables are refreshed, but no new cofolding takes place. The viewer shows an existing replicate whose predicted pIC50 is closest to the median; an optional overlay shows the replicate with highest predicted pIC50. Neither is an averaged coordinate structure.

## New run

Set `RUN_MODE = "new"`, edit the configuration cell (ligand, target sequence, pre-filter, Boltz-2 settings) and run the notebook top to bottom. The notebook includes a demonstration ligand and a protein example—**replace them with your intended system**. Boltz-2 inference happens only in Section 6; Section 5 ends with a dry run that reports how much cofolding is still needed. `N_REPLICATES` sets the number of independent seeds. Inspect the command and sampling preset before launching large batches.

## Resuming, caching and MSA reuse

`run_boltz2_replicates` tracks work **per job**. If a run crashes or is interrupted, call it again and only the unfinished jobs are computed (earlier versions required `overwrite=True`, which also deleted the finished jobs). Finished predictions are also stored in a cross-run cache (`<output_directory>/cofold_cache`), keyed by protein sequence, ligand SMILES, seed and prediction settings, so re-enumerating and re-running does not repeat identical cofolding. `plan_boltz2_replicates(state, n_replicates)` reports what is still to compute without launching anything.

When no `MSA_PATH` is given, the target MSA is generated once (one structure-only job via the MSA server), cached under `<output_directory>/msa_cache`, and reused by every replicate and later run. Pass `reuse_msa=False` to keep the old per-batch server behaviour or `cache_dir=None` to disable the prediction cache. If the MSA file cannot be found after generation, the run falls back to server MSAs and says so in the log.

## Pre-filter

`open_hit_expander.prefilter.apply_prefilter` triages enumerated analogues on CPU before cofolding: property windows relative to the parent, alerts the analogue introduces compared with the parent, core retention (parent atom indices or SMARTS), Boltz-2 size limits, and an optional diversity-aware cap. Decisions go to `prefilter_log.csv` (with `prefilter_funnel.csv` and `prefilter_rejected.csv`) and flags appear in `compound_summary.csv`. Default thresholds are illustrative; set them for your series.

## Pose viewer

Section 8 writes `complex_browser.html` into the run folder and opens it in your browser. The inline notebook version (`VIEWER_INLINE = True`) can render as a blank box when the payload is large or the 3Dmol.js script is blocked. The standalone page loads 3Dmol.js from 3dmol.org by default; set `THREE_DMOL_JS` to a downloaded `3Dmol-min.js` to make it work offline.

## Figures

Interaction plots and stability plots use a colour-vision-deficiency-safe palette and are saved as 200-dpi PNGs (`interaction_analysis/figures/` and `figures/` in the run folder). The stability plots draw every replicate as a dot (with a median tick and min–max line), the paired change against the parent, and replicate spread against potency. Both plot functions accept `show=` and `save_dir=`.

## Runtime

Boltz-2 predicts structures for **all** jobs in a batch first and then runs the affinity step for all jobs, so a run shows two progress phases in `boltz.log`. As one observed example, not a benchmark: on a single Apple Silicon MacBook Pro (MPS, `balanced` preset) a replicate of six jobs took about 3.6 min per ligand for structures plus about 1.5 min per ligand for affinities. Expect large differences on other hardware, and keep the machine awake and plugged in for long runs (for example `caffeinate -dims jupyter lab` on macOS).

## Outputs

Run-specific files are written under `hit_expander_output/boltz2_cofold/<timestamp>/`. Key outputs:

- `replicate_predictions.csv`, `replicate_status.csv`: potency from independent replicate predictions and per-replicate status
- `replicates/replicate_NN_seed_S/`: per-seed `results/`, `boltz.log`, `job_status.csv`, `replicate_results.csv`
- `hit_expander_boltz2_all_results.csv`: median statistics, representative structure and highest-pIC50 pose metadata
- `notebook_settings.json`: settings and versions that produced the run
- `complex_browser.html`: standalone pose viewer
- `interaction_analysis/`: saved PLIP status, details, fingerprint matrix, heatmap and `figures/`
- `figures/`: stability plots
- `compound_assessment/compound_summary.csv`: sortable export with potency, interaction metrics and `prefilter_flags`
- `compound_assessment/compound_assessment.html`: standalone offline summary

Beside the run folders, `hit_expander_output/` also holds `prefilter_*.csv`, `cofold_cache/` and `msa_cache/`.

The report treats PLIP status `ok` and `success` as successful; unavailable PLIP analyses remain missing rather than silently becoming zero contacts. The Jaccard contact similarity uses residue–interaction-type features, not 3D pose similarity. Assessment labels depend on configurable effect/variability cutoffs and have not been calibrated to experimental uncertainty. Pose RMSD is not currently calculated.

## Known limitations

- The screening presets change only `--diffusion_samples`; Boltz-2's recycling, sampling-step and affinity-specific options are not exposed.
- `NUM_WORKERS` and `PREPROCESSING_THREADS` are accepted by `prepare_boltz2_batch` but are not added to the `boltz predict` command.
- The affinity head's accuracy depends on the target; check the ranking against known actives before relying on it.
- Enumeration is single-site (aromatic C–H substitution and aromatic atom swaps); no matched-pair, bioisostere or purchasability logic.

## Checks

```bash
python -m compileall -q src
python -m pytest -q tests
```

The same checks run in GitHub Actions (`.github/workflows/ci.yml`). Tests cover run metadata, PLIP status handling, resume / cache / MSA reuse (with a fake `boltz` executable), the pre-filter, the standalone viewer export and the plots. `tests/test_prefilter_chem.py` needs RDKit. **End-to-end Boltz-2 inference, interactive viewer rendering in a browser, and full scientific validation are not covered by these tests.**

See `CHANGELOG.md` for changes since 0.7.0.
