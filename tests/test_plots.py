"""The stability and interaction plots render from saved CSVs and can save PNGs."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from open_hit_expander.plotting import plot_interaction_summary
from open_hit_expander.stability import plot_stability


def _replicates(folder, n_rep=3):
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(1, n_rep + 1):
        for job in ["p_parent", "p_a1", "p_a2", "p_a3"]:
            rows.append({"job_id": job, "replicate": rep, "seed": 1000 + rep, "is_parent": job == "p_parent",
                         "predicted_pic50": 6.0 + (0 if job == "p_parent" else 0.4) + rng.normal(0, 0.1)})
    pd.DataFrame(rows).to_csv(folder / "replicate_predictions.csv", index=False)


def test_stability_plots_render_and_save(tmp_path):
    _replicates(tmp_path)
    figures = plot_stability(tmp_path, show=False, save_dir=tmp_path / "figures")
    assert len(figures) == 3
    for name in ("stability_potency", "stability_delta_vs_parent", "stability_spread"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 1000


def test_stability_plots_with_two_replicates(tmp_path):
    _replicates(tmp_path, n_rep=2)
    assert len(plot_stability(tmp_path, show=False)) == 3


def test_interaction_plots_save_pngs(tmp_path):
    details = pd.DataFrame({"job_id": ["a", "a", "b"], "residue": ["A:LEU83", "A:GLU81", "A:LEU83"],
                            "interaction": ["hydrogen_bond", "hydrophobic_interaction", "hydrogen_bond"]})
    figures = plot_interaction_summary(details, show=False, save_dir=tmp_path / "fig")
    assert set(figures) == {"fingerprint", "counts", "conservation", "similarity"}
    assert len(list((tmp_path / "fig").glob("*.png"))) == 4
