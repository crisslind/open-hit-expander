from dataclasses import replace
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from .core import (ALL_ADDITIONS, ALL_AROMATIC_SWAPS, run_hit_expander, export_results, draw_passed)

def hit_expander_widget(INPUT, CONFIG, OUTPUT_DIRECTORY, state=None):
    if state is None:
        state = {"run": None, "output_files": None}
    ADDITION_LABELS = {
        "Methyl (Me)": "Me",
        "Fluorine (F)": "F",
        "Chlorine (Cl)": "Cl",
        "Bromine (Br)": "Br",
        "Hydroxy (OH)": "OH",
        "Methoxy (OMe)": "OMe",
        "Amino (NH2)": "NH2_aromatic",
        "Methylamino (NHMe)": "NHMe_aromatic",
    }
    SWAP_LABELS = {
        "Aromatic C → N": "C_to_N",
        "Aromatic N → C": "N_to_C",
        "Aromatic N → O": "N_to_O",
        "Aromatic O → N": "O_to_N",
        "Aromatic O → S": "O_to_S",
        "Aromatic S → O": "S_to_O",
        "Aromatic N → S": "N_to_S",
        "Aromatic S → N": "S_to_N",
    }

    addition_boxes = {
        value: widgets.ToggleButton(
            value=False,
            description=label,
            layout=widgets.Layout(width="155px", height="34px"),
            tooltip=f"Toggle {label}",
        )
        for label, value in ADDITION_LABELS.items()
    }
    swap_boxes = {
        value: widgets.ToggleButton(
            value=False,
            description=label,
            layout=widgets.Layout(width="155px", height="34px"),
            tooltip=f"Toggle {label}",
        )
        for label, value in SWAP_LABELS.items()
    }


    def _show_toggle_state(change):
        change["owner"].button_style = "info" if change["new"] else ""


    for box in (*addition_boxes.values(), *swap_boxes.values()):
        box.observe(_show_toggle_state, names="value")

    run_button = widgets.Button(
        description="Enumerate analogues",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="190px", height="38px"),
    )
    clear_button = widgets.Button(
        description="Reset",
        icon="refresh",
        layout=widgets.Layout(width="100px", height="38px"),
    )
    widget_output = widgets.Output()
    RUN_STATE = state


    def _clear_selections(_=None):
        for box in (*addition_boxes.values(), *swap_boxes.values()):
            box.value = False


    def _run_selected(_=None):
        selected_additions = tuple(
            name for name in ALL_ADDITIONS if addition_boxes[name].value
        )
        selected_swaps = tuple(
            name for name in ALL_AROMATIC_SWAPS if swap_boxes[name].value
        )

        with widget_output:
            clear_output(wait=True)
            if not selected_additions and not selected_swaps:
                print("Select at least one addition or aromatic atom swap.")
                return

            selected_config = replace(
                CONFIG,
                additions=selected_additions,
                aromatic_swaps=selected_swaps,
            )
            try:
                selected_run = run_hit_expander(INPUT, selected_config)
                output_files = export_results(selected_run, OUTPUT_DIRECTORY)
            except Exception as exc:
                print(f"Run failed: {type(exc).__name__}: {exc}")
                return

            RUN_STATE["run"] = selected_run
            RUN_STATE["output_files"] = output_files

            print("Selected additions:", ", ".join(selected_additions) or "none")
            print("Selected aromatic swaps:", ", ".join(selected_swaps) or "none")
            display(selected_run["summary"])

            if not selected_run["invalid_inputs"].empty:
                print("Invalid inputs:")
                display(selected_run["invalid_inputs"])

            print(f"Passed: {len(selected_run['passed'])}")
            print(f"Filtered after valid generation: {len(selected_run['rejected'])}")
            print(f"Invalid generation attempts: {len(selected_run['invalid_generation'])}")
            display(draw_passed(selected_run))

            if not selected_run["rejected"].empty:
                print("First 50 rejected structures:")
                display(
                    selected_run["rejected"][
                        [
                            "source_id", "analog_smiles", "transformations",
                            "morgan_tanimoto", "structural_alerts", "rejection_reasons",
                        ]
                    ].head(50)
                )

            print("Output files:")
            display(pd.Series(
                {name: str(path.resolve()) for name, path in output_files.items()},
                name="file",
            ))


    clear_button.on_click(_clear_selections)
    run_button.on_click(_run_selected)

    button_grid_layout = widgets.Layout(
        grid_template_columns="repeat(4, 155px)",
        grid_gap="8px 8px",
        width="650px",
    )
    additions_panel = widgets.VBox([
        widgets.HTML("<b>Additions at aromatic C–H positions</b>"),
        widgets.GridBox(
            children=list(addition_boxes.values()),
            layout=button_grid_layout,
        ),
    ])
    swaps_panel = widgets.VBox([
        widgets.HTML("<b>Aromatic atom swaps</b>"),
        widgets.GridBox(
            children=list(swap_boxes.values()),
            layout=button_grid_layout,
        ),
    ])
    window_title = widgets.HTML(
        "<div style='background:#263746;color:white;font-weight:600;"
        "padding:9px 12px;border-radius:5px 5px 0 0;'>Hit Expander</div>"
    )
    instructions = widgets.HTML(
        "<div style='color:#444;padding:2px 0 4px 0;'>"
        "Select transformation buttons, then run. Blue buttons are selected. "
        "Each analogue receives one modification at one aromatic position.</div>"
    )
    controls = widgets.VBox([
        window_title,
        instructions,
        additions_panel,
        swaps_panel,
        widgets.HBox(
            [run_button, clear_button],
            layout=widgets.Layout(gap="10px", padding="4px 0 2px 0"),
        ),
        widget_output,
    ], layout=widgets.Layout(
        border="1px solid #9aa3aa",
        padding="0 14px 14px 14px",
        width="700px",
    ))
    return controls