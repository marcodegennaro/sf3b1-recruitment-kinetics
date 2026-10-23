import networkx as nx   
import numpy as np
import sympy as sym
from sympy import *
from sympy.printing.mathml import mathml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random 
import math
from collections import Counter
from scipy import stats

# =============================================================================
# PARAMETER GENERATION
# =============================================================================


def generate_parameter_sets(n_sets: int,
                            k_2_multiplic_cost: int | float,
                            assumption: bool = True,
                            overrides: dict | None = None,) -> list[dict]:
    """
    Genera `n_sets` set di parametri rispettando vincoli biologici.

    Return
    ----------
        k1        : tasso di associazione al sito corretto
        k_1       : tasso di dissociazione basale dal sito corretto
        k2        : tasso di associazione al sito errato (≈ k1, ±20%)
        k_2       : tasso di dissociazione dal sito errato (>> k_1_basal)
        k3        : tasso di transizione allo stato chiuso - sito corretto
        k4        : tasso di dissociazione dallo stato chiuso - sito corretto
        alpha     : coefficiente che scala l'effetto del farmaco linearmente  
        beta      : coefficiente che scala l'effetto del farmaco 
        n         : coefficiente di Hill (intero in [1, 5])

    Vincoli (proofreading kinetic)
    ----------
        1. k1  ≈ k2:  the assotiation rate of BP and Decoy should be the same
        2. k_2 ≥ MULTIPLICATION_COSTANT * k_1: the dwell time on site should be greater in the BP case
        3. k_1 ≈ k4
        (not implemented ) 4. k3 / k4 >> 0: the intermediate step is a very fast step          
    
    Vincoli:
    ----------
    - All the parameter are drawn randomly from the range 10^-3 ; 10^3
    - alpha lower bound is -1 (not log_min) because values below 0.1 produce 
     negligible drug effects and are biologically irrelevant.
    
    Parameters
    ----------
    n_sets: int, numero di set di parametri da generare 

    k_2_multiplication_costant: parametro da motliplicare a k_1 per ottenere k_2 (secondo l'assunzioen 
    che k_2 sia molto maggiore rispetto a k_1)

    assumption = True, parametro che considera se le assunzioni vanno applicate o meno ( se non sono applicate allora 
    tutti i parametri cinetici assumono valori da 10^-3 a 10^3)


    Overrides : parametri che vuoi fissare ad un valore preciso invece
                di campionarli. Es.: {"alpha": 10.0, "k_2": 0.5}
                Il valore viene applicato DOPO tutta la generazione,
                sovrascrivendo la variabile campionata.             
    """
    overrides = dict(overrides or {})
    MULTIPLICATION_COSTANT = k_2_multiplic_cost

    pf_ratio = overrides.pop("pf_ratio", None)   # float or None
    param_sets = []
    log_min, log_max = -3, 3

    # First checks
    if n_sets <= 0:
        raise ValueError(f"n_sets must be a positive integer, got {n_sets}.")
    if k_2_multiplic_cost <= 0:
        raise ValueError(f"k_2_multiplic_cost must be positive, got {k_2_multiplic_cost}.")
        
    for _ in range(n_sets):
        
        # Fixed ki generation
        k1        = 10 ** np.random.uniform(log_min, log_max)
        k_1       = 10 ** np.random.uniform(log_min, log_max)
        k3       = 10 ** np.random.uniform(log_min, log_max)

        # Parameter Generation 
        alpha = 10 ** np.random.uniform(-1, 3)
        beta = 10 ** np.random.uniform(log_min, log_max)
        n     = np.random.randint(1, 6)                         # Hill coefficient ≥ 1

        # Remeaning ki generation ---- with assumption ----
        if assumption:                                          
            # Assumption 1: k1  ≈ k2

            k2        = k1 * np.random.uniform(0.8, 1.2)

            # Assumption 2: k_2 ≥ MULTIPLICATION_COSTANT * k_1

            if pf_ratio is not None:
                k_2 = k_1 * pf_ratio          # fixed rateo, k_1 still random 
            else:
                min_k_2 = k_1 * MULTIPLICATION_COSTANT
                if min_k_2 > 10 ** log_max:
                    k_2 = 10 ** log_max
                else:
                    k_2 = 10 ** np.random.uniform(np.log10(min_k_2), log_max)

            # Assumption 3: k_1 ≈ k4

            k4        = k_1 * np.random.uniform(0.8, 1.2)

        # Remeaning ki generation ---- without assumption ----
        else:
            k2  = 10 ** np.random.uniform(log_min, log_max)
            k4  = 10 ** np.random.uniform(log_min, log_max)
            
            if pf_ratio is not None:
                k_2 = k_1 * pf_ratio 
            else: 
                k_2 = 10 ** np.random.uniform(log_min, log_max)


        ps = {
            "k1":    round(k1,    6),
            "k_1":   round(k_1,   6),
            "k2":    round(k2,    6),
            "k_2":   round(k_2,   6),
            "k3":    round(k3,    6),
            "k4":    round(k4,    6),
            "alpha": round(alpha, 6),
            "beta":  round(beta,  6),
            "n":     int(n),
            "pf_ratio": round(k_2 / k_1, 6),
        }

        for key, val in overrides.items():
            if key not in ps:
                raise KeyError(f"Override key '{key}' non è un parametro valido.")
            ps[key] = val

        param_sets.append(ps)

    return param_sets


# =============================================================================
# Filtering Models by Expected (data) Trends
# =============================================================================

def filter_model_by_trend(result_dict: dict, nodes_num: int, joint = False) -> dict:
    """
    Analyzes dictionary-based model simulations to identify parameter sets
    matching biological trends as a function of increasing drug concentration.

    Expected trends:
        - P1: increasing with drug
        - P2: decreasing with drug
        - P3: increasing with drug
        - P4 (only if nodes_num == 4): decreasing with drug
        ( if joint ):
        - P2 + P3 decrease with drug  

    Parameters
    ----------
    result_dict : dict
        Input structure: result_dict[model_name] -> list of parameter_set dicts.
        Each parameter_set has 'model', 'mods_applied', 'params', and 'results'.
    nodes_num : int
        Number of nodes in the topology (3 or 4).
    joint : bool
        If true, check for the joint probability of State 2 and State 3 that decrease 

    Returns
    -------
    dict
        A filtered copy of result_dict with the same structure, containing only
        parameter sets that satisfy all trend conditions.
    """
    filtered_dict = {}

    for model_name, parameter_sets in result_dict.items():
        passing_sets = []

        for parameter_set in parameter_sets:
            sim_data = parameter_set['results']

            slopes_p1 = []
            slopes_p2 = []
            slopes_p3 = []
            slopes_p4 = []  # populated only when nodes_num == 4

            for i in range(len(sim_data) - 1):
                dx = sim_data[i + 1]['drug'] - sim_data[i]['drug']

                if dx == 0:
                    continue

                slopes_p1.append((sim_data[i + 1]['P1'] - sim_data[i]['P1']) / dx)
                slopes_p2.append((sim_data[i + 1]['P2'] - sim_data[i]['P2']) / dx)
                slopes_p3.append((sim_data[i + 1]['P3'] - sim_data[i]['P3']) / dx)

                if nodes_num == 4:
                    slopes_p4.append((sim_data[i + 1]['P4'] - sim_data[i]['P4']) / dx)

            if not slopes_p1:
                continue

            avg_p1 = sum(slopes_p1) / len(slopes_p1)
            avg_p2 = sum(slopes_p2) / len(slopes_p2)
            avg_p3 = sum(slopes_p3) / len(slopes_p3)

            # Core trend conditions (nodes 3 and 4)
            p1_increases = avg_p1 > 0                   # increase
            p2_decreases = avg_p2 < 0                   # decrease
            p3_increases = avg_p3 > 0                   # increase

            passes = p1_increases and p2_decreases and p3_increases

            # Additional P4 condition for 4-node topology
            if nodes_num == 4 and passes:                       # Consider also the previous condition
                avg_p4 = sum(slopes_p4) / len(slopes_p4)
                p4_decrease = avg_p4 < 0                # decrease
                passes = p4_decrease


            # Additional Joint condition: correct and incorrect site decrease jointly 
            if joint and passes: 
                passes = (avg_p2 + avg_p3) < 0         # sum decrease

            if passes:                                 # Check the passing condition
                passing_sets.append(parameter_set)

        if passing_sets:
            filtered_dict[model_name] = passing_sets

    return filtered_dict


# =============================================================================
# Analyzing result (to update)
# =============================================================================

def _analysis_summary(result_dict, nodes_number):
    successful_sets = filter_model_by_trend(result_dict, nodes_number)
    print(f"Analysis Summary:")

    # Calcolo del totale: iteriamo sui valori (le liste di parametri) del dizionario
    total_analyzed = sum(len(parameter_sets) for parameter_sets in result_dict.values())

    print(f"Total sets analyzed: {total_analyzed}")
    print(f"Sets matching the required trend: {len(successful_sets)}")

    # Opzionale: Mostra quali modelli hanno contribuito di più
    if successful_sets:
        from collections import Counter
        origins = Counter(s['origin_model'] for s in successful_sets)
        print("\nTop contributing models:")
        for model, count in origins.most_common(5):
            print(f"- {model}: {count} sets")


# =============================================================================
# Parameter analysis (to update)
# =============================================================================

def analyse_parameter_distribution(
    run_results: list[dict],
    param_keys: list[str] | None = None,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
    plot: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "Parameter distribution — selected runs",
) -> dict:
    """
    Compute summary statistics and (optionally) plot the distribution of kinetic
    parameters from a list of already-filtered run_model output dicts.

    Parameters
    ----------
    run_results : list[dict]
        Direct output of run_model / run_single_model, pre-filtered to the
        parameter sets of interest.
    param_keys : list[str] | None
        Parameters to analyse. If None, all keys found in the first `params`
        dict are used (plus any derived quantities defined in DERIVED_META).
    percentiles : tuple[float]
        Percentiles to compute (values in [0, 100]).
    plot : bool
        Whether to produce a figure.
    figsize : tuple | None
        Override default figure size.
    title : str
        Figure suptitle.

    Returns
    -------
    dict with keys:
        "stats_raw"     → pd.DataFrame  — statistics for raw parameters
        "stats_derived" → pd.DataFrame  — statistics for derived quantities
        "df_params"     → pd.DataFrame  — raw per-run parameter values
        "df_derived"    → pd.DataFrame  — raw per-run derived values
        "fig"           → matplotlib Figure (or None if plot=False)
        "interpretation"→ dict[param → str]  biological interpretation strings
    """

    # ── 1. Extract parameter dicts ────────────────────────────────────────────
    param_list = [entry["params"] for entry in run_results]
    if not param_list:
        raise ValueError("run_results is empty.")

    if param_keys is None:
        param_keys = list(param_list[0].keys())

    df_params = pd.DataFrame(
        [{k: float(p[k]) for k in param_keys if k in p} for p in param_list]
    )

    # ── 2. Compute derived quantities ─────────────────────────────────────────
    derived_rows = []
    for p in param_list:
        row = {}
        for dname, dmeta in DERIVED_META.items():
            try:
                row[dname] = dmeta["fn"](p)
            except (KeyError, ZeroDivisionError):
                row[dname] = np.nan
        derived_rows.append(row)
    df_derived = pd.DataFrame(derived_rows).dropna(axis=1, how="all")

    # ── 3. Summary statistics ─────────────────────────────────────────────────
    def _summary(df: pd.DataFrame) -> pd.DataFrame:
        pct_labels = [f"p{int(p)}" for p in percentiles]
        records = []
        for col in df.columns:
            s = df[col].dropna()
            row = {
                "n":    len(s),
                "mean": s.mean(),
                "std":  s.std(),
                "cv%":  s.std() / s.mean() * 100 if s.mean() != 0 else np.nan,
            }
            for p, lbl in zip(percentiles, pct_labels):
                row[lbl] = np.percentile(s, p)
            sk = stats.skew(s)
            row["skewness"] = sk
            row["skew_interp"] = (
                "symmetric" if abs(sk) < 0.5
                else ("right-skewed" if sk > 0 else "left-skewed")
            )
            records.append(pd.Series(row, name=col))
        return pd.DataFrame(records)

    stats_raw     = _summary(df_params)
    stats_derived = _summary(df_derived)

    # ── 4. Biological interpretation strings ──────────────────────────────────
    interpretation = {}
    for k in param_keys:
        if k in PARAM_META:
            interpretation[k] = PARAM_META[k]["label"] + f" [{PARAM_META[k]['unit']}]"
    for dname, dmeta in DERIVED_META.items():
        if dname in df_derived.columns:
            interpretation[dname] = dmeta["interp"]

    # ── 5. Print summary table ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"  N = {len(run_results)} parameter sets")
    print(f"{'═'*60}\n")

    print("── Raw parameters ──────────────────────────────────────────")
    _print_table(stats_raw, param_keys, PARAM_META, percentiles)

    print("\n── Derived quantities ──────────────────────────────────────")
    _print_table(stats_derived, list(df_derived.columns), DERIVED_META, percentiles)

    print("\n── Biological interpretation ───────────────────────────────")
    _print_interpretation(stats_raw, stats_derived, df_params, df_derived)

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    fig = None
    if plot:
        fig = _make_figure(
            df_params, df_derived, stats_raw, stats_derived,
            param_keys, figsize, title, percentiles
        )

    return {
        "stats_raw":      stats_raw,
        "stats_derived":  stats_derived,
        "df_params":      df_params,
        "df_derived":     df_derived,
        "fig":            fig,
        "interpretation": interpretation,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_table(summary_df, keys, meta, percentiles):
    pct_labels = [f"p{int(p)}" for p in percentiles]
    for k in keys:
        if k not in summary_df.index:
            continue
        row  = summary_df.loc[k]
        lbl  = meta.get(k, {}).get("label", k)
        unit = meta.get(k, {}).get("unit", "")
        print(f"  {lbl}  [{unit}]")
        print(f"    mean ± std : {row['mean']:.4f} ± {row['std']:.4f}   "
              f"(CV = {row['cv%']:.1f}%,  {row['skew_interp']})")
        pct_str = "   ".join(f"p{int(p)}={row[lbl]:.4f}" for p, lbl in zip(percentiles, pct_labels))
        print(f"    percentiles: {pct_str}")
        print()


def _print_interpretation(stats_raw, stats_derived, df_params, df_derived):
    lines = []

    # Kd comparison
    if "Kd_BP" in stats_derived.index and "Kd_decoy" in stats_derived.index:
        kd_bp    = stats_derived.loc["Kd_BP",    "p50"]
        kd_decoy = stats_derived.loc["Kd_decoy", "p50"]
        ratio    = kd_decoy / kd_bp if kd_bp > 0 else np.nan
        lines.append(
            f"  • Median Kd_BP = {kd_bp:.3f}, Kd_decoy = {kd_decoy:.3f}  "
            f"→ decoy is ~{ratio:.1f}× weaker binder (drug-free)."
        )

    # Selectivity
    if "selectivity" in stats_derived.index:
        sel = stats_derived.loc["selectivity", "p50"]
        lines.append(
            f"  • Median selectivity (Kd_decoy/Kd_BP) = {sel:.2f}  "
            + ("→ clear preference for BP site." if sel > 1 else "→ decoy competitive.")
        )

    # Drug effect
    if "k_1_fold_max" in stats_derived.index:
        fold = stats_derived.loc["k_1_fold_max", "p50"]
        lines.append(
            f"  • Drug can increase k₋₁ by up to {fold:.1f}× at saturation "
            f"(median α = {stats_raw.loc['alpha','p50']:.2f})."
        )

    # Hill cooperativity
    if "n" in stats_raw.index:
        n_med = stats_raw.loc["n", "p50"]
        coop  = "cooperative (sigmoidal response)" if n_med > 1 else "non-cooperative (hyperbolic)"
        lines.append(f"  • Median Hill coefficient n = {n_med:.1f} → {coop}.")

    # Variability flag
    for k in ["k1", "k_1", "k2", "k_2"]:
        if k in stats_raw.index and stats_raw.loc[k, "cv%"] > 50:
            lines.append(
                f"  ⚠ {k} shows high variability (CV = {stats_raw.loc[k,'cv%']:.0f}%) "
                "— the trend is robust across a wide kinetic range."
            )

    print("\n".join(lines) if lines else "  (not enough derived data to interpret)")


def _make_figure(df_params, df_derived, stats_raw, stats_derived,
                 param_keys, figsize, title, percentiles):

    present_keys = [k for k in param_keys if k in df_params.columns]
    present_derived = [k for k in df_derived.columns]

    n_raw     = len(present_keys)
    n_derived = len(present_derived)
    n_cols    = 3
    n_rows_raw = int(np.ceil(n_raw / n_cols))
    n_rows_der = int(np.ceil(n_derived / n_cols))
    n_rows_total = n_rows_raw + n_rows_der + 1  # +1 for correlation heatmap row

    if figsize is None:
        figsize = (n_cols * 4, n_rows_total * 3.2)

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=13, fontweight="normal", y=1.01)
    gs = gridspec.GridSpec(n_rows_total, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35)

    def _color(k):
        return PARAM_META.get(k, {}).get("color", "#888888")

    # Row block 1: raw parameters
    for idx, k in enumerate(present_keys):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        data = df_params[k].dropna().values
        ax.hist(data, bins=30, color=_color(k), alpha=0.75, edgecolor="white", linewidth=0.4)
        for p, ls in zip([25, 50, 75], ["--", "-", "--"]):
            ax.axvline(np.percentile(data, p), color="#333", linewidth=0.9,
                       linestyle=ls, alpha=0.7)
        meta = PARAM_META.get(k, {})
        ax.set_title(meta.get("label", k), fontsize=10, pad=4)
        ax.set_xlabel(meta.get("unit", ""), fontsize=8, color="#666")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        mean_v = stats_raw.loc[k, "mean"] if k in stats_raw.index else np.nan
        cv_v   = stats_raw.loc[k, "cv%"]  if k in stats_raw.index else np.nan
        ax.text(0.97, 0.95, f"μ={mean_v:.3f}\nCV={cv_v:.0f}%",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
                color="#555")

    # Row block 2: derived quantities
    row_offset = n_rows_raw
    for idx, k in enumerate(present_derived):
        ax = fig.add_subplot(gs[row_offset + idx // n_cols, idx % n_cols])
        data = df_derived[k].dropna().values
        color = "#b5a0d4"
        ax.hist(data, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        for p, ls in zip([25, 50, 75], ["--", "-", "--"]):
            ax.axvline(np.percentile(data, p), color="#333", linewidth=0.9,
                       linestyle=ls, alpha=0.7)
        meta = DERIVED_META.get(k, {})
        ax.set_title(meta.get("label", k), fontsize=10, pad=4)
        ax.set_xlabel("value", fontsize=8, color="#666")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        med_v = stats_derived.loc[k, "p50"] if k in stats_derived.index else np.nan
        ax.text(0.97, 0.95, f"med={med_v:.3f}",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
                color="#555")

    # Row block 3: correlation heatmap (raw params only)
    ax_heat = fig.add_subplot(gs[row_offset + n_rows_der, :])
    corr = df_params[present_keys].corr(method="spearman")
    labels = [PARAM_META.get(k, {}).get("label", k) for k in present_keys]
    im = ax_heat.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax_heat.set_xticks(range(len(present_keys)))
    ax_heat.set_yticks(range(len(present_keys)))
    ax_heat.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_heat.set_yticklabels(labels, fontsize=8)
    for i in range(len(present_keys)):
        for j in range(len(present_keys)):
            val = corr.values[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7.5, color="white" if abs(val) > 0.5 else "#333")
    plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01, label="Spearman ρ")
    ax_heat.set_title("Spearman correlation — raw parameters", fontsize=10, pad=6)

    plt.tight_layout()
    return fig





# =============================================================================
# Entry Point
# =============================================================================
def main():
    print("\nProgram's running...!\n")

if __name__ == "__main__":
    main()


