
import numpy as np

from sympy import *
from sympy.printing.mathml import mathml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random 
import math
from collections import Counter



import matplotlib.ticker as mticker





# =============================================================================
# Single model Visualziation
# =============================================================================

def plot_single_simulation(simulation_results: list[dict], nodes_number: int) -> None:
    """
    Display results from a single run_single_model call.

    Parameters
    ----------
    simulation_results : list[dict]
        Direct output of run_single_model.
    nodes_number : int, optional
        Number of states in the model topology (3 or 4). Controls which
        P columns are read and plotted. Default is 3.

    Returns
    -------
    None
    """
    # ── State palette — extend if nodes_number == 4 ───────────────────────────
    PALETTE = {
        'P1': ('#A9A9A9', '--', 'P_freeRNA'),
        'P2': ('#E67E22', '-',  'P_BP_open'),
        'P3': ('#27AE60', '-',  'P_Decoy'),
        'P4': ('#2980B9', '-',  'P_BP_closed'),   # only used when nodes_number == 4
    }
    states = [f'P{i}' for i in range(1, nodes_number + 1)]

    # ── Collect and aggregate results ─────────────────────────────────────────
    all_results = [res for sim in simulation_results for res in sim['results']]
    df = pd.DataFrame(all_results)
    agg = df.groupby('drug')[states].agg(['mean', 'std']).reset_index()

    model_name   = simulation_results[0]['model']
    mods_applied = simulation_results[0]['mods_applied']
    n_sets       = len(simulation_results)
    subtitle     = '  |  '.join(f"{p}: {fn}" for p, fn in mods_applied.items())

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5))

    drug = agg['drug']
    for state in states:
        color, ls, label = PALETTE[state]
        mean = agg[(state, 'mean')]
        std  = agg[(state, 'std')].fillna(0)

        ax.plot(drug, mean, color=color, linestyle=ls, linewidth=2.5, label=label)
        ax.fill_between(drug, (mean - std), (mean + std), color=color, alpha=0.15)

    # ── Formatting ────────────────────────────────────────────────────────────
    clean_title = (model_name.replace('[', ': ')
                             .replace(']', '')
                             .replace('__', '   ')
                             .title())
    ax.set_title(clean_title, fontsize=14, fontweight='bold', color='#2C3E50', pad=12)
    ax.set_xlabel(f'Drug Concentration\n'
                  f'({n_sets} parameter sets — shaded band: ±1 SD)', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)

    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=9,
             color='#7F8C8D', style='italic')

    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# =============================================================================
# Few model Visualization
# =============================================================================
    
def plot_selected_models(
    data: dict[str, list[dict]],
    nodes_number: int,
    title: str,
) -> None:
    """
    Display 2–4 model simulations side by side with a prominent shared title.

    Parameters
    ----------
    data : dict[str, list[dict]]
        Dictionary mapping model names to their simulation results.
    nodes_number : int
        Number of states in the model topology (3 or 4).
    title : str
        Main title displayed above all subplots.

    Returns
    -------
    None
    """
    PALETTE = {
    'P1': ('#94A3B8', '--', 'P_freeRNA'),    
    'P3': ('#E11D48', '-',  'P_Decoy'),      
    'P2': ('#38BDF8', '-',  'P_BP_open'),    
    'P4': ('#0F172A', '-',  'P_BP_closed'),  
}
    states = [f'P{i}' for i in range(1, nodes_number + 1)]

    models = list(data.keys())
    n = len(models)
    if not (2 <= n <= 4):
        raise ValueError(f"plot_simulation_small expects 2–4 models, got {n}.")

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(
        1, n,
        figsize=(5.5 * n, 5.5),
        sharey=True
    )
    if n == 1:
        axes = [axes]

    for i, (model_name, ax) in enumerate(zip(models, axes)):
        all_results = [res for sim in data[model_name] for res in sim['results']]
        df = pd.DataFrame(all_results).groupby('drug').mean().reset_index()

        available_states = [s for s in states if s in df.columns]
        for state in available_states:
            color, ls, label = PALETTE[state]
            ax.plot(df['drug'], df[state], color=color, linestyle=ls,
                    linewidth=2.5, label=label)

        clean_title = (model_name.replace('[', ': ')
                                 .replace(']', '')
                                 .replace('__', '\n')   # line break between params
                                 .title())
        ax.set_title(clean_title, fontsize=11, fontweight='bold',
                     color='#2C3E50', pad=12, linespacing=1.5)

        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.set_xlabel('Drug Concentration', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

        if i == 0:
            ax.set_ylabel('Probability', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(available_states),
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08)
    )

    fig.suptitle(title, fontsize=15, fontweight='bold', color='#1A252F', y=1.03)

    plt.tight_layout()
    plt.show()

# =============================================================================
# Grid multiple visualization
# =============================================================================

def plot_simulation_grid(data: dict[str, list[dict]], nodes_number: int, cols: int = 4) -> None:
    """
    Display results from multiple model simulations in a subplot grid.

    Parameters
    ----------
    data : dict[str, list[dict]]
        Dictionary mapping model names to their simulation results.
    cols : int, optional
        Number of columns in the subplot grid. Default is 4.
    nodes_number : int, optional
        Number of states in the model topology (3 or 4). Controls which
        P columns are read and plotted. Default is 3.

    Returns
    -------
    None
    """
    # ── State palette — extend if nodes_number == 4 ───────────────────────────
    PALETTE = {
        'P1': ('#A9A9A9', '--', 'P_freeRNA'),
        'P2': ('#E67E22', '-',  'P_BP_open'),
        'P3': ('#27AE60', '-',  'P_Decoy'),
        'P4': ('#2980B9', '-',  'P_BP_closed'),   # only used when nodes_number == 4
    }
    states = [f'P{i}' for i in range(1, nodes_number + 1)]

    models = list(data.keys())
    num_models = len(models)
    rows = math.ceil(num_models / cols)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.5), sharey=True)
    axes_flat = axes.flatten()

    for i, model_name in enumerate(models):
        ax = axes_flat[i]

        all_results = [res for sim in data[model_name] for res in sim['results']]
        df = pd.DataFrame(all_results).groupby('drug').mean().reset_index()

        for state in states:
            color, ls, label = PALETTE[state]
            ax.plot(df['drug'], df[state], color=color, linestyle=ls,
                    linewidth=2.5, label=label)

        clean_title = (model_name.replace('[', ': ')
                                 .replace(']', '')
                                 .replace('__', '   ')
                                 .title())
        ax.set_title(clean_title, fontsize=12, fontweight='bold', color='#2C3E50', pad=15)
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)

        if i % cols == 0:
            ax.set_ylabel('Probability', fontsize=10)
        if i >= (num_models - cols):
            ax.set_xlabel('Drug Concentration', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=nodes_number,
               fontsize=12, frameon=True)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# ── helper di formatting (evita ripetizione) ─────────────────────────────────
def _format_ax(ax, axes, row_idx, col_idx, n_states,
               param_name, state, STATE_LABELS, log_scale_x, log_scale_y):
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")
    ax.set_xlabel(param_name, fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(STATE_LABELS.get(state, state), fontsize=11,
                 fontweight="bold", color="#2C3E50", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)






def plot_perturbation_2d(
    data_grid: dict[tuple[float, float], dict[str, list[dict]]],
    param_x: str,              # asse x — es. "alpha"
    param_y: str,              # asse y — es. "pf_ratio"
    nodes_number: int,
    states_to_plot: list[str] | None = None,   # default: tutti
    drug_agg: str | float = "mean",
    log_scale_x: bool = True,
    log_scale_y: bool = True,
) -> None:
    """
    Heatmap 2D: asse x = param_x, asse y = param_y, colore = P(state).

    data_grid è un dict { (val_x, val_y): run_results }
    dove run_results ha la stessa struttura di run_all_models().

    Una riga di heatmap per ogni stato, una colonna per ogni modello.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd

    STATE_LABELS = {
        "P1": "P_freeRNA",
        "P2": "P_BP_open",
        "P3": "P_Decoy",
        "P4": "P_BP_closed",
    }
    CMAPS = {
        "P1": "Greys",
        "P2": "Blues",
        "P3": "Reds",
        "P4": "Purples",
    }

    all_states = [f"P{i}" for i in range(1, nodes_number + 1)]
    states = states_to_plot or all_states

    # valori unici degli assi
    x_vals = sorted(set(k[0] for k in data_grid))
    y_vals = sorted(set(k[1] for k in data_grid))

    # nomi modelli (stessi per tutte le celle)
    model_names = list(next(iter(data_grid.values())).keys())
    n_models = len(model_names)
    n_states = len(states)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_states, n_models,
        figsize=(5 * n_models, 4 * n_states),
        squeeze=False,
    )

    for col_idx, model_name in enumerate(model_names):
        for row_idx, state in enumerate(states):
            ax = axes[row_idx, col_idx]

            # costruisci la matrice (y_vals × x_vals)
            matrix = np.full((len(y_vals), len(x_vals)), np.nan)

            for xi, xv in enumerate(x_vals):
                for yi, yv in enumerate(y_vals):
                    runs = data_grid.get((xv, yv), {}).get(model_name, [])
                    if not runs:
                        continue

                    all_results = [
                        res for run in runs for res in run["results"]
                    ]
                    df = pd.DataFrame(all_results)

                    if drug_agg == "mean":
                        val = df[state].mean()
                    elif drug_agg == "max":
                        val = df[state].max()
                    else:
                        target = float(drug_agg)
                        idx = (df["drug"] - target).abs().idxmin()
                        val = df.loc[idx, state]

                    matrix[yi, xi] = val

            im = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap=CMAPS[state],
                vmin=0, vmax=1,
            )

            # tick labels
            ax.set_xticks(range(len(x_vals)))
            ax.set_yticks(range(len(y_vals)))
            ax.set_xticklabels(
                [f"{v:.2g}" for v in x_vals],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_yticklabels([f"{v:.2g}" for v in y_vals], fontsize=8)

            ax.set_xlabel(param_x, fontsize=10)
            ax.set_ylabel(param_y, fontsize=10)

            clean_model = (model_name.replace("[", ": ").replace("]", "")
                                     .replace("__", " | ").title())
            ax.set_title(
                f"{STATE_LABELS.get(state, state)}\n{clean_model}",
                fontsize=10, fontweight="bold", color="#2C3E50", pad=8,
            )

            cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=20)
            cb.set_label("probability", fontsize=8)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Robustness — {param_y} vs {param_x}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()