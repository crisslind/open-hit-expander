# Changelog

## 1.0.0 (2026-10-25)

### Added
- Per-job resumable cofolding: rerunning section 6 computes only unfinished jobs.
- Cross-run prediction cache (`cofold_cache/`) keyed by sequence, ligand, seed and settings; `plan_boltz2_replicates` dry run.
- Target MSA generated once and reused across replicates and runs (`msa_cache/`).
- `prefilter` module: parent-relative property windows, new-vs-parent alerts, core/SMARTS rules, Boltz-2 size limits, diversity-aware cap, decision log and funnel.
- `save_complex_browser`: standalone pose viewer page (optionally with a local 3Dmol.js).
- `show=` / `save_dir=` options on interaction and stability plots; PNG export.
- Stability plots redesigned for few replicates (replicate dots, paired change vs parent, spread vs potency).
- `prefilter_flags` column in `compound_summary.csv`; `ligand_key` column in the manifest.
- Tests for resume/cache/MSA reuse (fake `boltz`), pre-filter, viewer export and plots; GitHub Actions CI.
- Notebook reorganised into sections 1–13 with a single configuration cell, cell ids and cleared outputs; renamed from `Open_Hit_Expander_boltz2_v070.ipynb` to `Open_Hit_Expander_boltz2.ipynb`.

### Fixed
- `NameError` (`display` not imported) when a ligand exceeded the 128-atom Boltz-2 limit; the error now names the ligands.
- An interrupted replicate no longer requires `overwrite=True`, which previously deleted finished jobs.
- The IPython "Consider using IPython.display.IFrame" warning from the inline viewer is suppressed.

### Changed
- Notebook default `RUN_MODE` is `"new"`.
- Interaction plot palette is colour-vision-deficiency safe.
