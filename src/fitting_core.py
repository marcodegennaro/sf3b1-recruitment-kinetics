# Standard libraries used throughout the notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import itertools
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import model_database as db
import simulation_core as sim
import warnings
from scipy.optimize import minimize


# =============================================================================
# Observed Values
# =============================================================================

def _observed_lfc(data: pd.DataFrame, vector_conc_without_control) -> dict:
    """Avoid to consider Control explicitly because the LFC is computed with respect to Ctrl."""
    behaviour = {}
    vector_means = []
    vectors_std = []

    cols = [col for col in data.columns if 'lfc_' in col and 'Ctrl' not in col]
    data = data.loc[:, cols].copy()
    for col in cols:
        lfc_mean = np.mean(data[col])
        lfc_std = np.std(data[col])
        if lfc_std == 0:
            lfc_std = 1e-11
        vector_means.append(lfc_mean)
        vectors_std.append(lfc_std)

    ps = {
        'conc':    np.asarray(vector_conc_without_control),
        'means':   np.asarray(vector_means),
        'stds':    np.asarray(vectors_std),
        'weights': 1 / np.asarray(vectors_std)**2
    }

    return ps


# =============================================================================
# Predicted Values
# =============================================================================

def _p_bound(result: dict, nodes_number: int) -> float:
    """
    P_bound = P_BP_site + P_Decoy_site (+ P_BP_closed if 4-state).

    Node map (from simulation_core):
        3-state:  P1=free  P2=BP_open   P3=decoy
        4-state:  P1=free  P2=BP_open   P3=decoy   P4=BP_closed
    """
    if nodes_number == 3:
        return result["P2"] + result["P3"]
    elif nodes_number == 4:
        return result["P2"] + result["P3"] + result["P4"]
    else:
        raise ValueError(f"Unsupported nodes_number: {nodes_number}")


def _param_array_to_dict(
    free_parameters_values: list,
    free_param_names: list,
    fixed_params: dict | None = None,
) -> dict:

    # 1. Combine fixed and free parameters into the initial dictionary
    if fixed_params is not None:
        result_dict = {**dict(zip(free_param_names, free_parameters_values)), **fixed_params}
    else:
        result_dict = dict(zip(free_param_names, free_parameters_values))

    required_keys = ['k1', 'k_1', 'k2', 'k_2', 'k3', 'k4', 'alpha', 'beta', 'delta']
    missing_keys = []

    for key in required_keys:
        if key not in result_dict:
            result_dict[key] = 0
            missing_keys.append(key)

    if missing_keys:
        message = (
            f"The following parameters were missing and have been automatically "
            f"set to 0: {', '.join(missing_keys)}"
        )
        warnings.warn(message, category=UserWarning, stacklevel=2)

    return result_dict


def _predict_lfc(
    free_parameters_values: list,
    free_param_names: list,
    drug_conc: np.array,
    model_name: str,
    fixed_params: dict | None = None,
):
    """
    Predict log fold-change (LFC) values for a given parameter set and model.

    The function applies two class-specific scalers to baseline kinetic rates
    before computing steady-state probabilities:
        - delta   scales k_2 (decoy dissociation rate) — encodes decoy strength
        - beta scales k_1 (correct-site dissociation rate) — encodes BP strength

    Both scalers are expected to lie in [0, 1] and are injected as fixed_params
    by the multiclass fitting routine. A value of 1 means no scaling (reference
    class); lower values reduce the respective rate.

    Parameters
    ----------
    free_parameters_values : list of floats
        Values of the free (shared) kinetic parameters in linear scale.
    free_param_names : list of str
        Names corresponding to free_parameters_values.
    drug_conc : np.array
        Drug concentrations to evaluate (must include 0 or a near-zero value
        as the control reference).
    model_name : str
        Key identifying the drug-modulation model in db.ALL_MODELS.
    fixed_params : dict or None
        Parameters that are not optimised (e.g. class-specific scalers
        injected by the multiclass wrapper).

    Returns
    -------
    np.array
        LFC values at each non-control drug concentration.
    """

    drug_conc = np.sort(drug_conc.copy())
    if np.isclose(drug_conc[0], 0):
        drug_conc = drug_conc[1:]

    params = _param_array_to_dict(free_parameters_values, free_param_names, fixed_params)


    # Both scalers are applied multiplicatively so that a value of 1 = no effect.
    params['k_2'] = params['delta']   * params['k_2']

    params['k_1'] = params['beta']    * params['k_1']

    mod = None

    # Assume the uniqueness of the model ## WRONG !!! Same model, different topologies 
    for model in db.ALL_MODELS:
        if model_name in model:
            mod = model
            break
    if mod is None:
        raise ValueError(
            f"The model: '{model_name}' is not in the database. "
            f"Please check available models with: db.list_models()"
        )
    # Retrieve the model info (check in the model_database module for more information)
    model_info  = db.ALL_MODELS[mod]
    mods        = model_info['mods']
    model_nodes = model_info['nodes_number']

    # Bound probabilities at control (drug = 0)
    res_Ctrl = sim._steady_state(
        param_set   = params,
        drug        = 0.,
        mods        = mods,
        drug_params = {},
        nodes_number= model_nodes,
    )
    bound_probabilites_Ctrl = _p_bound(result=res_Ctrl, nodes_number=model_nodes)

    # Bound probabilities at each drug concentration
    bound_probabilites_drug = []
    for d in drug_conc:
        res_drug = sim._steady_state(
            param_set   = params,
            drug        = d,
            mods        = mods,
            drug_params = {},
            nodes_number= model_nodes,
        )
        bound_probabilites_drug.append(_p_bound(result=res_drug, nodes_number=model_nodes))

    lfc_pred = np.log(np.asarray(bound_probabilites_drug) / np.asarray(bound_probabilites_Ctrl))
    return lfc_pred


# =============================================================================
# Single-class loss
# =============================================================================

def _weighted_mse(
    log_theta,
    free_param_names,
    drug_conc: np.array,
    lfc_obs: np.array,
    weights: np.array,
    model_name: str,
    fixed_params: None | dict = None
):
    """Scalar loss — operates in log10(θ) space."""
    theta    = 10.0 ** log_theta
    lfc_pred = _predict_lfc(
        free_parameters_values = theta,
        free_param_names       = free_param_names,
        drug_conc              = drug_conc,
        model_name             = model_name,
        fixed_params           = fixed_params,
    )

    if np.any(~np.isfinite(lfc_pred)):
        return 1e10

    residuals = lfc_obs - lfc_pred
    return float(np.sum(weights * residuals**2) / weights.sum())


# =============================================================================
# MULTICLASS LOSS
# =============================================================================

def _weighted_mse_multiclass(
    log_theta,
    free_param_names: list,
    drug_conc: np.array,
    obs_list: list,    
    class_names: list,      
    model_name: str,
    fixed_params: None | dict = None,
):
    """
    Compute a scalar weighted-MSE loss summed over all structural classes.

    The parameter vector log_theta is partitioned into two segments:
      1. Shared kinetic parameters (in log10 scale, converted to linear internally).
      2. Class-specific scalers delta and beta (already in linear [0,1] scale,
         NOT exponentiated — they are optimised directly in their natural domain).

    For each class, delta scales k_2 (decoy strength) and beta scales k_1
    (branch-point strength). These are injected into fixed_params before calling
    _predict_lfc, so the shared kinetic scaffold remains unmodified across classes.

    The convention for identifying class-specific parameters relies on their name
    prefix: any parameter whose name starts with 'delta_' or 'beta_' is treated
    as class-specific. All others are treated as shared.

    Parameters
    ----------
    log_theta : np.array
        Flattened optimisation vector. First n_shared entries are log10(shared params);
        remaining entries are linear class-specific scalers ordered as
        [delta_class0, beta_class0, delta_class1, beta_class1, ...].
    free_param_names : list of str
        Full list of parameter names, in the same order as log_theta.
        Built automatically by multistart_fit_multiclass.
    drug_conc : np.array
        Drug concentrations (including control at 0).
    obs_list : list of dict
        One observed-LFC dict per class (output of _observed_lfc).
        Must be aligned with class_names.
    class_names : list of str
        Class identifiers (e.g. ['strongDecoy', 'noDecoy']).
        Used to look up the correct scaler values in log_theta.
    model_name : str
        Drug-modulation model key in db.ALL_MODELS.
    fixed_params : dict or None
        Any additional fixed parameters common to all classes.

    Returns
    -------
    float
        Sum of per-class weighted MSE values (penalised to 1e10 on failure).
    """

    # Identify shared vs class-specific parameters by name prefix.

    class_specific_names = [
        p for p in free_param_names
        if p.startswith('delta_') or p.startswith('beta_')
    ]
    shared_names = [p for p in free_param_names if p not in class_specific_names]
    n_shared     = len(shared_names)

    # Shared parameters: log10 → linear
    shared_values = 10.0 ** log_theta[:n_shared]

    # Class-specific scalers: already linear, just slice them out by index.
    def _build_class_params(class_name: str) -> dict:
        """
        Inject the two class-specific scalers for `class_name` into fixed_params.
        Scalers are read directly from log_theta (no exponentiation) because
        they are bounded to [0, 1] and optimised in linear space.
        """
        fp = dict(fixed_params) if fixed_params is not None else {}
        fp['delta']   = log_theta[free_param_names.index(f'delta_{class_name}')]
        fp['beta'] = log_theta[free_param_names.index(f'beta_{class_name}')]
        return fp

    # Compute per-class predictions and accumulate loss.
    total_loss = 0.0
    try:
        for obs, cname in zip(obs_list, class_names):
            fp_class = _build_class_params(cname)
            lfc_pred = _predict_lfc(
                free_parameters_values = shared_values,
                free_param_names       = shared_names,
                drug_conc              = drug_conc.copy(),
                model_name             = model_name,
                fixed_params           = fp_class,
            )
            if np.any(~np.isfinite(lfc_pred)):
                return 1e10

            residuals   = obs['means'] - lfc_pred
            class_loss  = np.sum(obs['weights'] * residuals**2) / obs['weights'].sum()
            total_loss += class_loss        # adding loss across all the classes -> global fitting

    except Exception:
        return 1e10

    return float(total_loss)


# =============================================================================
# MULTICLASS MULTI-START OPTIMIZER
# =============================================================================

def multistart_fit_multiclass(
    classes_data: list,       
    class_names: list,     
    shared_param_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_params: dict | None = None,
    n_starts: int = 100,
    log_bounds: tuple = (-3, 3),
    show_messages: bool = True,
    show_history: bool = False,
    show_dict: bool = False,
):
    """
    Multi-start gradient-based fitting over an arbitrary number of structural classes.

    Each class shares a common set of kinetic parameters (rates, drug-effect
    parameters) but has two private scalers:
      - delta   in [0, 1]: scales k_2, encoding decoy-site strength.
      - beta in [0, 1]: scales k_1, encoding branch-point retention strength.

    The optimisation vector is partitioned as:
        [ log10(shared_params) | delta_c0, beta_c0, delta_c1, beta_c1, ... ]

    Shared parameters are optimised in log10 space (bounds: log_bounds).
    Class-specific scalers are optimised in linear space (bounds: [0, 1]).

    The loss is the sum of per-class inverse-variance-weighted MSE between
    observed and predicted LFC values across all drug concentrations.

    Parameters
    ----------
    classes_data : list of pd.DataFrame
        One DataFrame per class; each must contain lfc_* columns (excluding Ctrl).
        Must be aligned positionally with class_names.
    class_names : list of str
        Identifiers for each class. Used to name the class-specific parameters
        (e.g. class_name='strongDecoy' → 'delta_strongDecoy', 'beta_strongDecoy').
    shared_param_names : list of str
        Names of kinetic parameters shared across all classes (e.g. ['k1', 'k_1', ...]).
        Do NOT include delta or beta here — they are added automatically.
    drug_conc : np.ndarray
        Drug concentrations to fit (including 0 as control reference).
    model_name : str
        Key of the drug-modulation model in db.ALL_MODELS.
    fixed_params : dict or None
        Any additional parameters held constant during optimisation (e.g. Hill
        coefficient n, or any rate you want to pin).
    n_starts : int
        Number of random restarts for the multi-start routine.
    log_bounds : tuple of (float, float)
        Search bounds in log10 space for shared kinetic parameters.
    show_messages : bool
        If True, print a summary of the best fit after the run.
    show_history : bool
        If True, append the full history DataFrame to the return value.
    show_dict : bool
        If True, append the best-parameter dictionary to the return value.

    Returns
    -------
    scipy.optimize.OptimizeResult, or tuple thereof with history and/or best_dict
    depending on show_history and show_dict flags.
    """

    # Build the full parameter name list: shared first, then class-specific pairs.
    # Order within each class: delta first, then beta — must match _build_class_params.
    class_specific_names = [
        f'{scaler}_{name}'
        for name in class_names
        for scaler in ('delta', 'beta')
    ]
    full_param_names = shared_param_names + class_specific_names

    # Compute observed LFC once, outside the optimisation loop.
    obs_list = [_observed_lfc(df, drug_conc) for df in classes_data]

    # Bounds: log space for shared params, linear [0,1] for class-specific scalers.
    bounds = []
    for name in shared_param_names:
        if name == 'alpha':
            bounds.append((0.0, 2.0))   # log10(100) = 2  →  alpha max = 100
        else:
            bounds.append(log_bounds)
    bounds += [(0.0, 1.0)] * len(class_specific_names)


    best_loss   = np.inf
    best_result = None
    history     = []

    for k in range(n_starts):

        # Random starting point: log-uniform for shared, uniform [0,1] for scalers.
        log_theta_shared = np.array([
            np.random.uniform(*(0.0, 2.0) if name == 'alpha' else log_bounds)
            for name in shared_param_names
        ])
        class_specific_start = np.random.uniform(0.0, 1.0,
                                                  size=len(class_specific_names))
        x0 = np.concatenate([log_theta_shared, class_specific_start])

        result = minimize(
            fun    = _weighted_mse_multiclass,
            x0     = x0,
            args   = (full_param_names, drug_conc,
                      obs_list, class_names,
                      model_name, fixed_params),
            method = 'L-BFGS-B',
            bounds = bounds,
        )

        history.append({
            "run_id"      : k,
            "success"     : bool(result.success),
            "final_cost"  : float(result.fun),
            "params_start": x0.tolist(),
            "params_final": result.x.tolist(),
        })

        if result.fun < best_loss:
            best_loss   = result.fun
            best_result = result

    if show_messages:
        n_sh = len(shared_param_names)
        shared_linear = 10 ** best_result.x[:n_sh]
        cs_vals       = best_result.x[n_sh:]

        print(f"\n==================== FIT COMPLETED =====================")
        print(f"Best loss   : {best_loss:.4f}")
        print(f"Convergence : {best_result.success}")
        print(f"Shared params (linear):")
        for name, val in zip(shared_param_names, shared_linear):
            print(f"  {name:12s} = {val:.4e}")
        print(f"Class-specific scalers:")
        for cname, delta, beta in zip(
            class_names,
            cs_vals[0::2],   # delta values (even positions)
            cs_vals[1::2],   # beta values (odd positions)
        ):
            print(f"  {cname:20s}  delta={delta:.4f}  beta={beta:.4f}")

    # Reconstruct best-parameter dictionary in linear scale.
    n_sh          = len(shared_param_names)
    shared_linear = 10 ** best_result.x[:n_sh]
    cs_vals       = best_result.x[n_sh:]
    best_dict     = dict(zip(shared_param_names, shared_linear))
    best_dict.update(dict(zip(class_specific_names, cs_vals)))

    output = [best_result]
    if show_history:
        output.append(pd.DataFrame(history))
    if show_dict:
        output.append(best_dict)

    if len(output) == 1:
        return output[0]
    return tuple(output)









# =============================================================================
# Fit esplorativo di (delta, beta) a livello di SINGOLA SEQUENZA
# =============================================================================


import numpy as np
import pandas as pd
from scipy.optimize import minimize




def fit_single_sequence(
    row: pd.Series,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_shared_params: dict,
    n_starts: int = 10,
    bounds_scalers: tuple = (0.0, 1.0),
) -> dict:
    """
    Fitta (delta, beta) per UNA sequenza, a parametri condivisi fissi.

    Parameters
    ----------
    row : pd.Series
        Una riga del DataFrame di classe (deve contenere colonne 'lfc_*',
        esclusa 'Ctrl' — stesso formato atteso da _observed_lfc).
    drug_conc : np.ndarray
        Concentrazioni di farmaco (incluso 0, come nel resto della pipeline).
    model_name : str
        Chiave del modello in db.ALL_MODELS.
    fixed_shared_params : dict
        Parametri condivisi GIA' TROVATI dal fit di classe (k1, k_1, k2, k_2,
        k3, k4, alpha, n, ...) — vengono passati come fixed_params e NON
        vengono ottimizzati qui.
    n_starts : int
        Multistart per evitare minimi locali (il problema è piccolo, 10
        starts sono già abbondanti).
    bounds_scalers : tuple
        Bound per delta e beta (default [0,1], come nel fit di classe).

    Returns
    -------
    dict con delta, beta, loss finale, e flag di convergenza.

    Nota
    ----
    Pesi uniformi sui 4 punti (una sequenza non ha una distribuzione di
    replicati su cui stimare una varianza, a differenza della media di
    classe). Con 2 parametri liberi su 4 punti dato, il fit è vicino
    all'interpolazione: interpretare la spread risultante con cautela,
    può riflettere rumore di conteggio più che eterogeneità biologica.
    """
    cols = [c for c in row.index if 'lfc_' in c and 'Ctrl' not in c]
    lfc_obs = row[cols].to_numpy(dtype=float)
    weights = np.ones_like(lfc_obs)

    def _loss(x):
        delta, beta = x
        fp = dict(fixed_shared_params)
        fp['delta'] = delta
        fp['beta'] = beta
        try:
            lfc_pred = _predict_lfc(
                free_parameters_values=[],
                free_param_names=[],
                drug_conc=drug_conc.copy(),
                model_name=model_name,
                fixed_params=fp,
            )
        except Exception:
            return 1e10
        if np.any(~np.isfinite(lfc_pred)):
            return 1e10
        residuals = lfc_obs - lfc_pred
        return float(np.sum(weights * residuals ** 2) / weights.sum())

    best = None
    for _ in range(n_starts):
        x0 = np.random.uniform(*bounds_scalers, size=2)
        res = minimize(
            _loss, x0, method='L-BFGS-B',
            bounds=[bounds_scalers, bounds_scalers],
        )
        if best is None or res.fun < best.fun:
            best = res

    return {
        'delta': float(best.x[0]),
        'beta': float(best.x[1]),
        'loss': float(best.fun),
        'success': bool(best.success),
    }


def fit_class_sequence_distribution(
    classes_data: list,
    class_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    shared_params: dict,
    n_sequences_per_class: int = 80,
    n_starts: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Applica fit_single_sequence a un sottocampione di sequenze per ciascuna
    classe, per ottenere una distribuzione empirica di delta/beta.

    Parameters
    ----------
    classes_data : list of pd.DataFrame
        Stessa struttura passata a multistart_fit_multiclass.
    class_names : list of str
        Allineati posizionalmente a classes_data.
    shared_params : dict
        Solo i parametri CONDIVISI dal best_dict del fit di classe
        (cioè best_dict SENZA le chiavi 'delta_*' / 'beta_*').
    n_sequences_per_class : int
        Sottocampione per classe (non serve fittare tutte le ~1000+).

    Returns
    -------
    pd.DataFrame con colonne: class, seq_id, delta, beta, loss, success.
    Pronto per un boxplot/stripplot per classe.
    """
    records = []
    for df, cname in zip(classes_data, class_names):
        n = min(n_sequences_per_class, len(df))
        sample = df.sample(n=n, random_state=random_state)
        for idx, row in sample.iterrows():
            fit = fit_single_sequence(
                row, drug_conc, model_name, shared_params, n_starts=n_starts,
            )
            fit['class'] = cname
            fit['seq_id'] = idx
            records.append(fit)
    return pd.DataFrame(records)




# =============================================================================
# GOODNESS OF FIT 
# =============================================================================
def goodness_of_fit_multiclass(
    best_dict: dict,
    shared_param_names: list,
    class_names: list,
    classes_data: list,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    """
    Compute R^2 (plain and weighted) per class, and overall, comparing observed
    vs predicted LFC. Observed values are computed internally from classes_data
    via _observed_lfc, positionally aligned with class_names.
    """
    rows = []
    all_obs     = []
    all_pred    = []
    all_weights = []

    for cname, df in zip(class_names, classes_data):
        obs = _observed_lfc(df, drug_conc)

        class_fixed = {
            'delta': best_dict[f'delta_{cname}'],
            'beta' : best_dict[f'beta_{cname}'],
        }
        if fixed_params is not None:
            class_fixed = {**fixed_params, **class_fixed}

        shared_values = [best_dict[name] for name in shared_param_names]

        pred = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_param_names,
            drug_conc              = drug_conc,
            model_name             = model_name,
            fixed_params           = class_fixed,
        )

        means   = obs['means']
        weights = obs['weights']
        pred    = np.asarray(pred)

        # Plain (unweighted) R^2
        ss_res = np.sum((means - pred) ** 2)
        ss_tot = np.sum((means - np.mean(means)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Weighted R^2 (same weighting as the fitting loss)
        w_mean = np.sum(weights * means) / np.sum(weights)
        ss_res_w = np.sum(weights * (means - pred) ** 2)
        ss_tot_w = np.sum(weights * (means - w_mean) ** 2)
        r2_w = 1 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan

        rows.append({
            "class": cname,
            "r2": r2,
            "r2_weighted": r2_w,
            "ss_res": ss_res,
            "ss_tot": ss_tot,
        })

        all_obs.append(means)
        all_pred.append(pred)
        all_weights.append(weights)

    # Overall, pooling all classes together.
    all_obs     = np.concatenate(all_obs)
    all_pred    = np.concatenate(all_pred)
    all_weights = np.concatenate(all_weights)

    ss_res_all = np.sum((all_obs - all_pred) ** 2)
    ss_tot_all = np.sum((all_obs - np.mean(all_obs)) ** 2)
    r2_all = 1 - ss_res_all / ss_tot_all if ss_tot_all > 0 else np.nan

    w_mean_all = np.sum(all_weights * all_obs) / np.sum(all_weights)
    ss_res_w_all = np.sum(all_weights * (all_obs - all_pred) ** 2)
    ss_tot_w_all = np.sum(all_weights * (all_obs - w_mean_all) ** 2)
    r2_w_all = 1 - ss_res_w_all / ss_tot_w_all if ss_tot_w_all > 0 else np.nan

    rows.append({
        "class": "OVERALL",
        "r2": r2_all,
        "r2_weighted": r2_w_all,
        "ss_res": ss_res_all,
        "ss_tot": ss_tot_all,
    })

    return pd.DataFrame(rows)
# =============================================================================
# MULTICLASS PLOT
# =============================================================================

def plot_fit_comparison_multiclass(
    classes_data: list,
    classes_names: list,
    best_fit,
    fit_dict: dict,
    drug_conc: np.ndarray,
    real_concentrations: list,
    model_name: str,
    fixed_params: dict | None = None,
    colors: list | None = None,
    titles: list | None = None,
):
    """
    Plot observed vs predicted LFC for each class after multiclass fitting.

    For each class, the class-specific scalers (delta, beta) are extracted
    from fit_dict and injected into fixed_params before calling _predict_lfc,
    exactly mirroring the logic used during optimisation.

    Parameters
    ----------
    classes_data : list of pd.DataFrame
    classes_names : list of str
        Must match the class identifiers used during fitting.
    best_fit : OptimizeResult
    fit_dict : dict
        Best-parameter dictionary as returned by multistart_fit_multiclass
        with show_dict=True. Keys follow the convention:
        shared params in linear scale; class-specific as 'delta_{name}' and
        'beta_{name}' in [0,1].
    drug_conc : np.ndarray
    real_concentrations : list
        Human-readable concentration labels for the x-axis.
    model_name : str
    fixed_params : dict or None
    colors : list or None
    titles : list or None
    """
    n = len(classes_data)
    if colors is None:
        colors = [f'C{i}' for i in range(n)]
    if titles is None:
        titles = classes_names

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # Identify shared parameter names: everything that is not a class-specific scaler.
    shared_names  = [k for k in fit_dict if not k.startswith('delta')
                                         and not k.startswith('beta')]
    shared_values = np.array([fit_dict[k] for k in shared_names])

    for i, (data, name) in enumerate(zip(classes_data, classes_names)):
        ax    = axes[i]
        color = colors[i]

        obs = _observed_lfc(data, drug_conc)
        ax.errorbar(
            drug_conc, obs['means'],
            yerr   = obs['stds'],
            fmt    = 'o',
            color  = 'black',
            ecolor = 'black',
            capsize= 4,
            label  = 'Observed (mean ± std)',
            zorder = 3,
        )

        fp = dict(fixed_params) if fixed_params is not None else {}
        fp['delta']   = fit_dict[f'delta_{name}']
        fp['beta'] = fit_dict[f'beta_{name}']

        lfc_pred = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_names,
            drug_conc              = drug_conc.copy(),
            model_name             = model_name,
            fixed_params           = fp,
        )

        ax.plot(
            drug_conc, lfc_pred,
            color  = color,
            lw     = 2,
            ls     = '--',
            label  = f'Predicted (loss = {best_fit.fun:.4f})',
            zorder = 2,
        )

        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.set_xticks(drug_conc)
        ax.set_xticklabels(
            [f'{c:.1e}' for c in real_concentrations],
            rotation=45, ha='right', fontsize=8,
        )
        ax.set_xlabel('Drug concentration (M)', fontsize=10)
        ax.set_ylabel('LFC vs Ctrl', fontsize=10)
        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Observed vs Predicted LFC — Kinetic Model Fit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def _format_condition_label(raw_label: str) -> str:
    """Clean a raw class identifier into an academic-style condition title,
    e.g. 'BP_Strong_Decoy 1_PYT_Strong' -> 'Strong BP | Strong Decoy'."""
    label = raw_label.split('_PYT_')[0]

    bp_class = None
    for bp in ('Strong', 'Intermediate', 'Weak'):
        if bp in label:
            bp_class = bp
            break

    if 'Decoy 1' in label:
        decoy_label = 'Strong Decoy'
    elif 'Decoy 7' in label:
        decoy_label = 'No Decoy'
    else:
        decoy_label = label.split('Decoy')[-1].strip() if 'Decoy' in label else label

    return f"{bp_class} BP | {decoy_label}"


def plot_dual_drug_fit_comparison(
    classes_data_1: list,
    classes_data_2: list,
    classes_names: list,
    drug_conc_1: np.ndarray,
    real_concentrations_1: list,
    drug_conc_2: np.ndarray,
    real_concentrations_2: list,
    model_name: str,
    fit_results_drugA: dict,
    fit_results_drugB: dict,
    drug_name_1: str = "Drug A",
    drug_name_2: str = "Drug B",
    fixed_params: dict | None = None,
    figsize: tuple | None = None,
    palette: list | None = None,
    dpi: int = 150,
):
    """
    Publication-ready side-by-side comparison of observed vs predicted LFC
    curves for two drugs, sharing the same set of experimental classes but
    each evaluated on its own concentration range/points, and each with its
    OWN observed data (classes_data_1, classes_data_2) since the two drugs
    can have a different number of dosing points per class.

    Layout
    ------
    2-row grid: top row = drug_name_1, bottom row = drug_name_2.
    Each column corresponds to the same experimental class (BP strength /
    decoy condition) for both drugs, allowing direct visual comparison.
    x-axis scale/ticks are independent per row since the two drugs are
    dosed on different concentration grids.

    Parameters
    ----------
    classes_data_1 : list of pd.DataFrame
        Raw experimental data per class for Drug 1. Each DataFrame's non-Ctrl
        'lfc_*' column count must match len(drug_conc_1).
    classes_data_2 : list of pd.DataFrame
        Raw experimental data per class for Drug 2. Each DataFrame's non-Ctrl
        'lfc_*' column count must match len(drug_conc_2).
    classes_names : list of str
        Raw class identifiers (e.g. "BP_Strong_Decoy 1_PYT_Strong"), matching
        the keys used in both fit_results dictionaries, positionally aligned
        with classes_data_1 and classes_data_2.
    drug_conc_1 : np.ndarray
        Drug concentrations (model/scaled space) used for prediction — Drug 1.
    real_concentrations_1 : list
        Human-readable concentration labels for the x-axis ticks — Drug 1.
    drug_conc_2 : np.ndarray
        Drug concentrations (model/scaled space) used for prediction — Drug 2.
    real_concentrations_2 : list
        Human-readable concentration labels for the x-axis ticks — Drug 2.
    model_name : str
        Key identifying the drug-modulation model in db.ALL_MODELS.
    fit_results_drugA, fit_results_drugB : dict
        Each must contain:
            - 'best_fit'  : OptimizeResult (used for the loss value in the legend)
            - 'fit_dict'  : best-parameter dictionary (shared params in linear
                             scale; class-specific as 'delta_{name}' / 'beta_{name}')
    drug_name_1, drug_name_2 : str
        Row labels identifying each drug.
    fixed_params : dict or None
        Additional parameters not optimised, passed through to _predict_lfc.
    figsize : tuple or None
    palette : list or None
        Colors for the predicted-fit lines, one per column. Defaults to the
        default matplotlib cycle.
    dpi : int

    Raises
    ------
    ValueError
        If, for any class, the number of non-Ctrl 'lfc_*' columns in the
        corresponding drug's data does not match that drug's drug_conc length,
        or if drug_conc_i and real_concentrations_i have mismatched lengths,
        or if classes_data_1/classes_data_2/classes_names have mismatched
        lengths. This surfaces the mismatch explicitly instead of letting it
        fail deep inside matplotlib's errorbar().
    """

    def _format_condition_label(raw_label: str) -> str:
        """Clean a raw class identifier into an academic-style condition title."""
        label = raw_label.split('_PYT_')[0]

        bp_class = None
        for bp in ('Strong', 'Intermediate', 'Weak'):
            if bp in label:
                bp_class = bp
                break

        if 'Decoy 1' in label:
            decoy_label = 'Strong Decoy'
        elif 'Decoy 7' in label:
            decoy_label = 'No Decoy'
        else:
            decoy_label = label.split('Decoy')[-1].strip() if 'Decoy' in label else label

        return f"BP: {bp_class} | {decoy_label}"

    # ------------------------------------------------------------------
    # Validation: catch length mismatches early with a readable message
    # ------------------------------------------------------------------
    if len(classes_data_1) != len(classes_names):
        raise ValueError(
            f"classes_data_1 (len={len(classes_data_1)}) and classes_names "
            f"(len={len(classes_names)}) must have the same length."
        )
    if len(classes_data_2) != len(classes_names):
        raise ValueError(
            f"classes_data_2 (len={len(classes_data_2)}) and classes_names "
            f"(len={len(classes_names)}) must have the same length."
        )
    if len(drug_conc_1) != len(real_concentrations_1):
        raise ValueError(
            f"drug_conc_1 (len={len(drug_conc_1)}) and real_concentrations_1 "
            f"(len={len(real_concentrations_1)}) must have the same length."
        )
    if len(drug_conc_2) != len(real_concentrations_2):
        raise ValueError(
            f"drug_conc_2 (len={len(drug_conc_2)}) and real_concentrations_2 "
            f"(len={len(real_concentrations_2)}) must have the same length."
        )

    # Each drug's data is validated only against that drug's own concentration grid.
    for data, raw_name in zip(classes_data_1, classes_names):
        n_obs_cols = len([c for c in data.columns if 'lfc_' in c and 'Ctrl' not in c])
        if n_obs_cols != len(drug_conc_1):
            raise ValueError(
                f"Class '{raw_name}' (Drug 1 data): found {n_obs_cols} non-Ctrl "
                f"'lfc_*' columns, but drug_conc_1 has {len(drug_conc_1)} points. "
                f"These must match — check whether the control (0) has been "
                f"included/excluded consistently in drug_conc_1."
            )
    for data, raw_name in zip(classes_data_2, classes_names):
        n_obs_cols = len([c for c in data.columns if 'lfc_' in c and 'Ctrl' not in c])
        if n_obs_cols != len(drug_conc_2):
            raise ValueError(
                f"Class '{raw_name}' (Drug 2 data): found {n_obs_cols} non-Ctrl "
                f"'lfc_*' columns, but drug_conc_2 has {len(drug_conc_2)} points. "
                f"These must match — check whether the control (0) has been "
                f"included/excluded consistently in drug_conc_2."
            )

    n_cols = len(classes_names)
    if palette is None:
        palette = [f'C{i}' for i in range(n_cols)]
    if figsize is None:
        figsize = (5.5 * n_cols, 9)

    # sharex=False: the two rows use different concentration grids.
    fig, axes = plt.subplots(
        2, n_cols, figsize=figsize, sharex=False, sharey='row', dpi=dpi,
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    drug_rows = [
        (drug_name_1, fit_results_drugA, axes[0, :], classes_data_1, drug_conc_1, real_concentrations_1),
        (drug_name_2, fit_results_drugB, axes[1, :], classes_data_2, drug_conc_2, real_concentrations_2),
    ]

    for row_idx, (drug_label, fit_results, row_axes, drug_classes_data, drug_conc, real_concentrations) in enumerate(drug_rows):
        best_fit = fit_results['best_fit']
        fit_dict = fit_results['fit_dict']

        shared_names  = [k for k in fit_dict if not k.startswith('delta')
                                             and not k.startswith('beta')]
        shared_values = np.array([fit_dict[k] for k in shared_names])

        for col_idx, (data, raw_name) in enumerate(zip(drug_classes_data, classes_names)):
            ax    = row_axes[col_idx]
            color = palette[col_idx]

            obs = _observed_lfc(data, drug_conc)
            ax.errorbar(
                drug_conc, obs['means'],
                yerr    = obs['stds'],
                fmt     = 'o',
                markersize   = 5,
                markerfacecolor = 'white',
                markeredgecolor = 'black',
                markeredgewidth = 1.0,
                color   = 'black',
                ecolor  = 'black',
                capsize = 4,
                alpha   = 0.9,
                label   = 'Observed (mean ± std)',
                zorder  = 3,
            )

            fp = dict(fixed_params) if fixed_params is not None else {}
            fp['delta'] = fit_dict[f'delta_{raw_name}']
            fp['beta']  = fit_dict[f'beta_{raw_name}']

            lfc_pred = _predict_lfc(
                free_parameters_values = shared_values,
                free_param_names       = shared_names,
                drug_conc              = drug_conc.copy(),
                model_name             = model_name,
                fixed_params           = fp,
            )

            ax.plot(
                drug_conc, lfc_pred,
                color  = color,
                lw     = 2.2,
                ls     = '-',
                label  = f'Predicted',
                zorder = 2,
            )

            ax.axhline(0, color='grey', lw=0.8, ls=':', zorder=1)
            ax.set_xticks(drug_conc)
            ax.set_xticklabels(
                [f'{c:.1e}' for c in real_concentrations],
                rotation=45, ha='right', fontsize=10,
            )
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)

            if row_idx == 0:
                ax.set_title(_format_condition_label(raw_name), fontsize=12, fontweight='bold')
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Drug concentration (M)', fontsize=10)

            if col_idx == 0:
                ax.set_ylabel('LFC vs Ctrl', fontsize=10)
            else:
                ax.set_ylabel('')

            legend = ax.legend(fontsize=8, loc='best', framealpha=0.6)
            legend.get_frame().set_edgecolor('lightgrey')

        row_axes[0].annotate(
            drug_label,
            xy=(-0.35, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold',
            rotation=90, va='center', ha='center',
        )

    fig.suptitle(
        'Observed vs Predicted LFC — Dual-Drug Kinetic Model Fit Comparison',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    plt.show()




def plot_optimization_diagnostic(fit_history: list, drug_type: str):
    '''a
    
    to add
    
    '''
    # 1. Filter for successful optimization runs
    successful_runs = fit_history[fit_history['success'] == True].copy()

    # 2. Sort the runs by final cost from lowest (best) to highest
    successful_runs_sorted = successful_runs.sort_values(by='final_cost').reset_index(drop=True)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plotting as both lines and points for clear visualization
    plt.plot(successful_runs_sorted.index, successful_runs_sorted['final_cost'], 
            linestyle='-', marker='o', color='#1f77b4', markersize=4, label='Ordered Fits')

    # Add titles and labels in B2 English
    plt.title(f'Multi-Start Optimization Diagnostic: {drug_type}', fontsize=14, fontweight='bold')
    plt.xlabel('Sorted Runs ', fontsize=12)
    plt.ylabel('Final Loss Value  (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # Display the plot
    plt.tight_layout()
    plt.show()




def plot_parameter_summary(
    fit_dict: dict,
    class_names: list,
    log_bounds: tuple = (-3, 3),
    figsize: tuple | None = None,
):
    """
    Visualise fitted parameters from multistart_fit_multiclass in a two-panel layout.

    Left panel — Shared kinetic parameters
        Horizontal bar chart showing log10 of each shared parameter.
        One bar per parameter, not repeated per class.
        Bars are coloured by sign (positive = warm, negative = cool) relative
        to the log10 value, giving an immediate sense of magnitude.

    Right panel — Class-specific scalers (delta and beta)
        Heatmap with classes on the x-axis and the two scalers on the y-axis.
        Values are in linear [0, 1] scale (not log-transformed) because they
        are bounded multiplicative scalers, not rates.
        Annotated with the numeric value for readability.

    Parameters
    ----------
    fit_dict : dict
        Best-parameter dictionary as returned by multistart_fit_multiclass
        with show_dict=True. Shared params are in linear scale; class-specific
        scalers follow the naming convention 'delta_{class}' and 'beta_{class}'.
    class_names : list of str
        Class identifiers — must match those used during fitting.
    log_bounds : tuple
        (min, max) used during optimisation, for reference lines on the left panel.
    figsize : tuple or None
        Override figure size. Default scales with number of classes.
    """

    # ── Split fit_dict into shared and class-specific ─────────────────────────
    shared = {
        k: v for k, v in fit_dict.items()
        if not k.startswith('delta_') and not k.startswith('beta_')
    }
    # Filtra i parametri con valore 0 o negativo prima del log10
    shared = {k: v for k, v in shared.items() if v > 0}

    # Convert shared params to log10 for display
    shared_log = {k: np.log10(v) for k, v in shared.items()}

    # Build class-specific matrix: rows = ['delta', 'beta'], cols = class_names
    scaler_matrix = np.array([
        [fit_dict[f'delta_{c}']   for c in class_names],
        [fit_dict[f'beta_{c}'] for c in class_names],
    ])
    scaler_labels = ['δ  (decoy strength\nscales k₋₂)', 'β  (BP strength\nscales k₋₁)']

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_classes = len(class_names)
    if figsize is None:
        figsize = (6 + n_classes * 1.1, max(5, len(shared) * 0.55 + 2))

    fig = plt.figure(figsize=figsize)
    # Left panel gets fixed width; right panel expands with number of classes
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[2, max(2, n_classes * 0.9)],
        wspace=0.45,
    )
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ── Left panel: shared parameters barchart ────────────────────────────────
    param_names = list(shared_log.keys())
    values      = list(shared_log.values())
    y_pos       = np.arange(len(param_names))

    # Colour: diverging from 0 in log space
    norm   = mcolors.TwoSlopeNorm(vmin=log_bounds[0], vcenter=0, vmax=log_bounds[1])
    colors = [plt.cm.RdBu_r(norm(v)) for v in values]

    bars = ax_left.barh(y_pos, values, color=colors, height=0.6,
                        edgecolor='white', linewidth=0.8)

    # Annotate each bar with the linear value
    for bar, v, lin in zip(bars, values, shared.values()):
        x_ann = v + (0.05 if v >= 0 else -0.05)
        ha    = 'left' if v >= 0 else 'right'
        ax_left.text(x_ann, bar.get_y() + bar.get_height() / 2,
                     f'{lin:.2e}', va='center', ha=ha, fontsize=7.5,
                     color='#333333')

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(param_names, fontsize=9)
    ax_left.set_xlabel('log₁₀(parameter value)', fontsize=9)
    ax_left.set_title('Shared kinetic\nparameters', fontsize=10, fontweight='bold', pad=10)
    ax_left.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax_left.set_xlim(log_bounds[0] - 0.5, log_bounds[1] + 0.8)
    ax_left.spines[['top', 'right']].set_visible(False)
    ax_left.invert_yaxis()  # top-to-bottom reading order

    # ── Right panel: class-specific scalers heatmap ───────────────────────────
    im = ax_right.imshow(
        scaler_matrix,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        aspect='auto',
    )

    # Annotate cells
    for row in range(2):
        for col in range(n_classes):
            val = scaler_matrix[row, col]
            text_color = 'white' if val > 0.65 else '#333333'
            ax_right.text(col, row, f'{val:.3f}',
                          ha='center', va='center',
                          fontsize=8, color=text_color, fontweight='bold')

    ax_right.set_xticks(np.arange(n_classes))
    ax_right.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels(scaler_labels, fontsize=9)
    ax_right.set_title('Class-specific scalers\n[0 – 1 linear scale]',
                        fontsize=10, fontweight='bold', pad=10)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    cbar.set_label('scaler value', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle('Fitted Parameter Summary', fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()



def plot_steady_state_probabilities(
    fit_dict: dict,
    class_names: list,
    drug_conc: np.ndarray,
    real_concentrations: list,
    model_name: str,
    fixed_params: dict | None = None,
    nodes_number: int = 4,
):
    """
    Plot steady-state probabilities (P1, P2, P3, P4) for each structural class
    across drug concentrations, using the parameters from a completed fit.

    For each class, the shared kinetic parameters are taken from fit_dict,
    and the class-specific scalers (delta, beta) are injected before calling
    _steady_state directly — bypassing LFC computation entirely.

    This is useful to:
      - verify that the model topology is behaving as expected
      - check whether drug modulation is actually changing state occupancy
      - inspect whether delta/beta are differentiating classes at steady state

    Layout: one row per class, one subplot per drug concentration point.
    Within each subplot, a bar chart shows P1..P4.

    Parameters
    ----------
    fit_dict : dict
        Best-parameter dictionary from multistart_fit_multiclass (show_dict=True).
        Shared params in linear scale; class-specific as 'delta_{name}', 'beta_{name}'.
    class_names : list of str
        Class identifiers — must match those used during fitting.
    drug_conc : np.ndarray
        Drug concentrations to evaluate (excluding control 0 — it is added internally).
    real_concentrations : list
        Human-readable labels for drug_conc (same length, used for x-axis titles).
    model_name : str
        Key in db.ALL_MODELS.
    fixed_params : dict or None
        Parameters fixed during fitting (e.g. {'k1': 1., 'k2': 1.}).
    nodes_number : int
        3 or 4, must match the model topology.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    mod_info  = db.ALL_MODELS[model_name]
    all_concs = np.concatenate([[0.], np.asarray(drug_conc)])
    all_labels = ['Ctrl'] + [f'{c:.1e}' for c in real_concentrations]

    n_classes = len(class_names)
    n_concs   = len(all_concs)
    node_keys = [f'P{i}' for i in range(1, nodes_number + 1)]
    node_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # P1..P4

    # Identify shared params
    shared_names = [
        k for k in fit_dict
        if not k.startswith('delta_') and not k.startswith('beta_')
    ]

    fig, axes = plt.subplots(
        n_classes, n_concs,
        figsize=(n_concs * 1.8, n_classes * 2.8),
        sharey='row',
    )
    # Normalise axes indexing to 2D always
    if n_classes == 1:
        axes = axes[np.newaxis, :]
    if n_concs == 1:
        axes = axes[:, np.newaxis]

    for row, name in enumerate(class_names):

        # Build param dict for this class
        params = {k: fit_dict[k] for k in shared_names}
        if fixed_params:
            params.update(fixed_params)
        params['delta'] = fit_dict[f'delta_{name}']
        params['beta']  = fit_dict[f'beta_{name}']

        # Apply class-specific scalers (mirrors _predict_lfc)
        params_scaled = dict(params)
        params_scaled['k_2'] = params_scaled.get('delta', 1.0) * params_scaled.get('k_2', 0.0)
        params_scaled['k_1'] = params_scaled.get('beta',  1.0) * params_scaled.get('k_1', 0.0)

        for col, (drug, label) in enumerate(zip(all_concs, all_labels)):
            ax = axes[row, col]

            res = sim._steady_state(
                param_set    = params_scaled,
                drug         = float(drug),
                mods         = mod_info['mods'],
                drug_params  = {},
                nodes_number = nodes_number,
            )

            probs  = [res.get(k, 0.0) for k in node_keys]
            x_pos  = np.arange(len(node_keys))

            bars = ax.bar(x_pos, probs, color=node_colors[:len(node_keys)],
                          width=0.6, edgecolor='white', linewidth=0.6)

            # Annotate bars with value if > 0.05
            for bar, p in zip(bars, probs):
                if p > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01,
                            f'{p:.2f}', ha='center', va='bottom',
                            fontsize=6.5, color='#333333')

            ax.set_ylim(0, 1.12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(node_keys, fontsize=7)
            ax.set_title(label, fontsize=7.5, pad=3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='y', labelsize=6.5)

            # Row label on the leftmost subplot only
            if col == 0:
                ax.set_ylabel(name.replace('_', ' '), fontsize=7.5, labelpad=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    node_labels = ['P1 — free', 'P2 — BP open', 'P3 — decoy', 'P4 — BP closed']
    legend_elements = [
        Patch(facecolor=node_colors[i], label=node_labels[i])
        for i in range(nodes_number)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=nodes_number,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f'Steady-state probabilities by class — {model_name}',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.show()









###############################################################################
# BOOTSTRAP FITTING TECHNIQUE
###############################################################################
# =============================================================================
# BOOTSTRAP FIT — CI sui parametri condivisi e class-specific (Variante B)
# =============================================================================
#
# Da incollare in fitting_core.py (usa le funzioni gia' presenti nel modulo:
# _observed_lfc, _weighted_mse_multiclass). Non serve importare nulla di nuovo
# oltre a quello gia' importato in cima al file (pandas, numpy, minimize).
#
# NOTA sul self-import in fondo a fitting_core.py:
#   "from fitting_core import _predict_lfc" e' un refuso e va rimosso —
#   _predict_lfc e' gia' definita piu' in alto nello stesso file.


def _resample_class_df(df: pd.DataFrame, random_state: np.random.Generator) -> pd.DataFrame:
    """Ricampiona con reinserimento le righe (sequenze) di una classe, stessa N originale."""
    n = len(df)
    idx = random_state.integers(0, n, size=n)   # campionamento con reinserimento
    return df.iloc[idx].reset_index(drop=True)


def _single_multistart_run(
    obs_list: list,
    class_names: list,
    shared_param_names: list,
    class_specific_names: list,
    full_param_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_params: dict | None,
    bounds: list,
    n_starts: int,
    rng: np.random.Generator,
    log_bounds: tuple,
):
    """
    Multistart ridotto, riusato identico ad ogni replica bootstrap.
    Stessa logica di init/bounds di multistart_fit_multiclass, ma con RNG
    dedicato (cosi' ogni replica bootstrap e' riproducibile se si fissa un seed).
    """
    best_loss   = np.inf
    best_result = None

    for _ in range(n_starts):
        log_theta_shared = np.array([
            rng.uniform(*(0.0, 2.0) if name == 'alpha' else log_bounds)
            for name in shared_param_names
        ])
        class_specific_start = rng.uniform(0.0, 1.0, size=len(class_specific_names))
        x0 = np.concatenate([log_theta_shared, class_specific_start])

        result = minimize(
            fun    = _weighted_mse_multiclass,
            x0     = x0,
            args   = (full_param_names, drug_conc, obs_list, class_names,
                      model_name, fixed_params),
            method = 'L-BFGS-B',
            bounds = bounds,
        )

        if result.fun < best_loss:
            best_loss   = result.fun
            best_result = result

    return best_result


def bootstrap_fit_multiclass(
    classes_data: list,
    class_names: list,
    shared_param_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    log_bounds: tuple,
    fixed_params: dict | None = None,
    n_bootstrap: int = 200,
    n_starts_bootstrap: int = 15,
    random_seed: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Bootstrap non-parametrico (Variante B) sull'intero modello multiclasse.

    Ad ogni replica:
      1. ogni classe viene ricampionata indipendentemente CON reinserimento
         (stessa dimensione della classe originale)
      2. media/std/pesi (_observed_lfc) vengono ricalcolati sul campione
         ricampionato — quindi anche i pesi della loss cambiano ad ogni replica
      3. si rifitta l'intero modello multiclasse (parametri condivisi +
         delta/beta per classe) con multistart ridotto (n_starts_bootstrap)
      4. si tiene solo il best result della replica (nessuna history salvata)

    Il risultato e' un DataFrame con una riga per replica bootstrap e una
    colonna per ciascun parametro (condivisi in scala lineare, scaler
    delta/beta in [0,1]), da cui si estraggono i percentili per il CI.

    Parameters
    ----------
    classes_data, class_names, shared_param_names, drug_conc, model_name,
    fixed_params : stessi argomenti di multistart_fit_multiclass.
    n_bootstrap : int
        Numero di repliche bootstrap.
    n_starts_bootstrap : int
        Multistart per replica (ridotto rispetto al fit singolo, per costo
        computazionale: n_bootstrap * n_starts_bootstrap fit totali).
    log_bounds : tuple
        Bounds in log10 per i parametri condivisi (stesso significato di
        multistart_fit_multiclass).
    random_seed : int or None
        Seed per riproducibilita' del ricampionamento e degli start.
    show_progress : bool
        Se True, stampa un avanzamento ogni 10% delle repliche.

    Returns
    -------
    pd.DataFrame
        Colonne: shared_param_names + ['delta_{c}', 'beta_{c}' per c in class_names]
                 + ['bootstrap_id', 'success', 'final_cost'].
        Una riga per replica bootstrap. Su questo DataFrame si calcolano
        percentili / CI con df.quantile([0.025, 0.5, 0.975]).
    """
    rng = np.random.default_rng(random_seed)

    class_specific_names = [
        f'{scaler}_{name}'
        for name in class_names
        for scaler in ('delta', 'beta')
    ]
    full_param_names = shared_param_names + class_specific_names

    bounds = []
    for name in shared_param_names:
        if name == 'alpha':
            bounds.append((0.0, 2.0))
        else:
            bounds.append(log_bounds)
    bounds += [(0.0, 1.0)] * len(class_specific_names)

    records = []
    checkpoint = max(1, n_bootstrap // 10)

    for b in range(n_bootstrap):

        # 1. Ricampiona ogni classe indipendentemente
        resampled_data = [_resample_class_df(df, rng) for df in classes_data]

        # 2. Ricalcola osservati (media, std, pesi) sul campione ricampionato
        obs_list = [_observed_lfc(df, drug_conc) for df in resampled_data]

        # 3. Rifit completo con multistart ridotto
        result = _single_multistart_run(
            obs_list             = obs_list,
            class_names           = class_names,
            shared_param_names    = shared_param_names,
            class_specific_names  = class_specific_names,
            full_param_names      = full_param_names,
            drug_conc             = drug_conc,
            model_name             = model_name,
            fixed_params           = fixed_params,
            bounds                 = bounds,
            n_starts                = n_starts_bootstrap,
            rng                     = rng,
            log_bounds              = log_bounds,
        )

        # 4. Salva solo il best result della replica
        n_sh          = len(shared_param_names)
        shared_linear = 10.0 ** result.x[:n_sh]
        cs_vals       = result.x[n_sh:]

        row = dict(zip(shared_param_names, shared_linear))
        row.update(dict(zip(class_specific_names, cs_vals)))
        row['bootstrap_id'] = b
        row['success']      = bool(result.success)
        row['final_cost']   = float(result.fun)
        records.append(row)

        if show_progress and (b + 1) % checkpoint == 0:
            print(f"  bootstrap {b + 1}/{n_bootstrap} completato "
                  f"(ultimo loss={result.fun:.4f}, success={result.success})")

    df_boot = pd.DataFrame(records)

    n_failed = (~df_boot['success']).sum()
    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} repliche bootstrap non sono convergute "
            f"(success=False). Valutare se escluderle dal CI.",
            category=UserWarning,
        )

    return df_boot


def summarize_bootstrap_ci(
    df_boot: pd.DataFrame,
    param_names: list,
    ci: float = 0.95,
    only_successful: bool = True,
) -> pd.DataFrame:
    """
    Calcola median e CI da un DataFrame prodotto da bootstrap_fit_multiclass.

    Parameters
    ----------
    df_boot : pd.DataFrame
        Output di bootstrap_fit_multiclass.
    param_names : list of str
        Sottoinsieme di colonne da riassumere (es. shared_param_names +
        ['delta_classA', 'beta_classA', ...]).
    ci : float
        Livello di confidenza (es. 0.95 → percentili 2.5 / 97.5).
    only_successful : bool
        Se True, esclude le repliche con success=False dal calcolo.

    Returns
    -------
    pd.DataFrame con colonne: param, median, ci_low, ci_high, n_reps.
    """
    data = df_boot[df_boot['success']] if only_successful else df_boot
    alpha = (1 - ci) / 2

    rows = []
    for p in param_names:
        s = data[p].dropna()
        rows.append({
            'param':   p,
            'median':  s.median(),
            'ci_low':  s.quantile(alpha),
            'ci_high': s.quantile(1 - alpha),
            'n_reps':  len(s),
        })

    return pd.DataFrame(rows)

# =============================================================================
# BOOTSTRAP PLOT — visualizzazione dei CI per parametri condivisi e class-specific
# =============================================================================
#
# Da incollare in fitting_core.py, vicino alle altre funzioni di plot
# (plot_fit_comparison_multiclass, plot_parameter_summary, ...).
# Usa solo librerie gia' importate in cima al file (matplotlib, numpy, pandas).


def plot_bootstrap_scalers(
    df_boot: pd.DataFrame,
    class_names: list,
    only_successful: bool = True,
    figsize: tuple | None = None,
):
    """
    Boxplot dei delta/beta bootstrap per ciascuna classe, con il best-fit
    puntuale sovrapposto se disponibile come mediana bootstrap.

    Un pannello per delta, uno per beta. Ogni box rappresenta la distribuzione
    bootstrap del parametro class-specific su n_bootstrap repliche.
    """
    data = df_boot[df_boot['success']] if only_successful else df_boot

    if figsize is None:
        figsize = (max(6, len(class_names) * 1.4), 8)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, scaler, label in zip(axes, ['delta', 'beta'],
                                  ['δ — decoy strength (scales k₋₂)',
                                   'β — BP strength (scales k₋₁)']):
        cols = [f'{scaler}_{c}' for c in class_names]
        box_data = [data[c].dropna().values for c in cols]

        bp = ax.boxplot(
            box_data, labels=class_names, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white',
                           markeredgecolor='black', markersize=6),
            medianprops=dict(color='#C44E52', linewidth=1.8),
            patch_artist=True,
        )
        for patch in bp['boxes']:
            patch.set_facecolor('#8CA6C9')
            patch.set_alpha(0.6)

        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('Class', fontsize=10)
    axes[0].set_title(
        f'Bootstrap distribution of class-specific scalers ({len(data)} iteration)',
        fontsize=11, fontweight='bold',
    )
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


def plot_bootstrap_shared_params(
    df_boot: pd.DataFrame,
    shared_param_names: list,
    log_scale: bool = True,
    only_successful: bool = True,
    figsize: tuple | None = None,
):
    """
    Boxplot della distribuzione bootstrap dei parametri condivisi
    (k1, k_1, k2, k3, k4, alpha, ...), un pannello per parametro.

    Parameters
    ----------
    log_scale : bool
        Se True, mostra i parametri cinetici (non alpha) in scala log10,
        coerentemente con lo spazio in cui vengono ottimizzati.
    """
    data = df_boot[df_boot['success']] if only_successful else df_boot

    n = len(shared_param_names)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.2, n_rows * 3.2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(shared_param_names):
        ax = axes[i]
        vals = data[name].dropna().values
        plot_vals = np.log10(vals) if (log_scale and name != 'alpha') else vals

        ax.boxplot(
            plot_vals, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white',
                           markeredgecolor='black', markersize=6),
            medianprops=dict(color='#C44E52', linewidth=1.8),
            patch_artist=True,
            boxprops=dict(facecolor='#8CA6C9', alpha=0.6),
        )
        ax.set_xticks([])
        ylabel = f'log₁₀({name})' if (log_scale and name != 'alpha') else name
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'Bootstrap distribution of shared kinetic parameters (n={len(data)} iteration)',
        fontsize=12, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.show()

# =============================================================================
# CROSS-DRUG COMPARISON — Mann-Whitney U test + annotated grouped boxplot
# =============================================================================
#



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SHARED_PARAM_YLIM = (1e-5, 1e5)   # hardcoded — matches the optimisation log_bounds


def _significance_stars(p_value: float) -> str:
    """Map a p-value to the conventional asterisk notation."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def _clean_label(name: str) -> str:
    """Strip the delta_/beta_ prefix and the trailing _PYT_Strong suffix for display."""
    label = name
    for prefix in ('delta_', 'beta_'):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    suffix = '_PYT_Strong'
    if label.endswith(suffix):
        label = label[: -len(suffix)]
    return label


def _bracket_y_position(local_max: float, ymin: float, ymax: float, log_scale: bool, frac: float = 0.06):
    """Bracket placed a fixed fraction of the AXIS range above local_max."""
    if log_scale:
        rng = np.log10(ymax) - np.log10(ymin)
        bracket_y = local_max * 10 ** (frac * rng)
        bracket_h = local_max * (10 ** (0.4 * frac * rng) - 1)
    else:
        rng = ymax - ymin
        bracket_y = local_max + frac * rng
        bracket_h = 0.4 * frac * rng
    return bracket_y, bracket_h


def _clamp_bracket(bracket_y, bracket_h, ymin, ymax, log_scale):
    """Used when the axis range is hard-enforced (never extended): pulls the
    bracket back under the ceiling instead of letting it stick out."""
    top = bracket_y + bracket_h
    ceiling = ymax * 0.97 if log_scale else ymax - 0.02 * (ymax - ymin)
    if top > ceiling:
        if log_scale:
            bracket_h = bracket_y * 0.03
            bracket_y = ceiling - bracket_h
        else:
            bracket_h = 0.02 * (ymax - ymin)
            bracket_y = ceiling - bracket_h
    return bracket_y, bracket_h


def _draw_significance_bracket(ax, x1: float, x2: float, y: float, h: float, text: str):
    """Draw a horizontal significance bracket between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='#333333')
    ax.text(
        (x1 + x2) / 2, y + h, text,
        ha='center', va='bottom', fontsize=10, color='#333333',
    )


def build_scaler_param_names(class_names: list) -> list:
    """Build 'delta_<class>'/'beta_<class>' column names from a list of class names."""
    return [f'{scaler}_{name}' for name in class_names for scaler in ('delta', 'beta')]


def _plot_param_group(ax, data_pb, data_h3b, names_pb, names_h3b, labels, log_scale, ylim, color_pb, color_h3b,
                       enforce_ylim: bool = False):
    """
    ylim is the reference range. If enforce_ylim=True the range is HARD —
    never changes, brackets are clamped to fit inside it. If False, the axis
    top is extended just enough to fit a bracket that would otherwise be
    clipped, and the reference range is respected as a minimum.
    """
    n = len(names_pb)
    group_centers = np.arange(n) * 2.2
    offset = 0.5
    pos_pb  = group_centers - offset
    pos_h3b = group_centers + offset
    records = []

    ymin, ymax = ylim
    if log_scale:
        ax.set_yscale('log')

    max_bracket_top = ymax

    for i, (name_pb, name_h3b, label) in enumerate(zip(names_pb, names_h3b, labels)):
        for df, name, side in [(data_pb, name_pb, 'PladB'), (data_h3b, name_h3b, 'H3B')]:
            if name not in df.columns:
                raise KeyError(
                    f"'{name}' non e' una colonna del dataframe {side}. "
                    f"Colonne che contengono '{label.split('_')[-1]}': "
                    f"{[c for c in df.columns if label.split('_')[-1] in c][:10]}"
                )

        vals_pb  = data_pb[name_pb].dropna().values
        vals_h3b = data_h3b[name_h3b].dropna().values

        for pos, vals, color in [(pos_pb[i], vals_pb, color_pb), (pos_h3b[i], vals_h3b, color_h3b)]:
            ax.boxplot(
                vals, positions=[pos], widths=0.8, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white',
                                markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=1.6),
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.65, edgecolor='#333333'),
                whiskerprops=dict(color='#333333'),
                capprops=dict(color='#333333'),
                flierprops=dict(marker='o', markersize=3, markerfacecolor=color,
                                 markeredgecolor='none', alpha=0.5),
            )

        u_stat, p_value = stats.mannwhitneyu(vals_pb, vals_h3b, alternative='two-sided')
        sig = _significance_stars(p_value)
        records.append({
            'parameter':    label,
            'U_statistic':  float(u_stat),
            'p_value':      float(p_value),
            'significance': sig,
            'n_pb':         len(vals_pb),
            'n_h3b':        len(vals_h3b),
        })

        local_max = max(vals_pb.max(), vals_h3b.max())
        bracket_y, bracket_h = _bracket_y_position(local_max, ymin, ymax, log_scale)
        if enforce_ylim:
            bracket_y, bracket_h = _clamp_bracket(bracket_y, bracket_h, ymin, ymax, log_scale)
        _draw_significance_bracket(ax, pos_pb[i], pos_h3b[i], bracket_y, bracket_h, sig)

        top = bracket_y + bracket_h
        if top > max_bracket_top:
            max_bracket_top = top

    if enforce_ylim:
        final_ymax = ymax
    elif log_scale:
        final_ymax = max_bracket_top * 1.05 if max_bracket_top > ymax else ymax
    else:
        final_ymax = max_bracket_top + 0.02 * (ymax - ymin) if max_bracket_top > ymax else ymax
    ax.set_ylim(ymin, final_ymax)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    return records, group_centers


def plot_cross_drug_comparison(
    df_pb: pd.DataFrame,
    df_h3b: pd.DataFrame,
    param_names,
    title: str,
    log_scale: bool = False,
    only_successful: bool = True,
    figsize: tuple | None = None,
    param_names_h3b: list | None = None,
    labels: list | None = None,
    y_limits = (1e-5, 1e5)
) -> pd.DataFrame:
    """
    Compare bootstrap parameter distributions between two drugs (PladB vs H3B)
    with a Mann-Whitney U test per parameter and a grouped, annotated boxplot.

    'alpha' is always drawn in its own panel with a linear [0, 100] axis.
    Other shared parameters, when log_scale=True, use a real log-scale
    y-axis HARD-LOCKED to SHARED_PARAM_YLIM = (1e-5, 1e5) — this never
    changes regardless of the data.

    By default, x-axis labels strip the delta_/beta_ prefix and the
    _PYT_Strong suffix (see _clean_label) and are drawn at 45°; pass
    `labels` to override.
    """
    data_pb  = df_pb[df_pb['success']]  if only_successful else df_pb
    data_h3b = df_h3b[df_h3b['success']] if only_successful else df_h3b

    color_pb  = '#8CA6C9'
    color_h3b = '#A6414B'

    names_pb  = list(param_names)
    names_h3b = list(param_names_h3b) if param_names_h3b is not None else list(param_names)
    labels    = list(labels) if labels is not None else [_clean_label(n) for n in names_pb]

    if len(names_pb) != len(names_h3b):
        raise ValueError("param_names e param_names_h3b devono avere la stessa lunghezza (stesso ordine).")

    has_alpha = 'alpha' in names_pb
    other_idx = [i for i, n in enumerate(names_pb) if n != 'alpha']
    other_pb, other_h3b, other_labels = (
        [names_pb[i] for i in other_idx],
        [names_h3b[i] for i in other_idx],
        [labels[i] for i in other_idx],
    )

    if has_alpha and other_idx:
        fig, (ax_main, ax_alpha) = plt.subplots(
            1, 2, figsize=figsize or (max(6, len(other_idx) * 2.0) + 2.5, 6.0),
            gridspec_kw={'width_ratios': [max(len(other_idx), 1), 1]},
        )
    elif has_alpha:
        fig, ax_alpha = plt.subplots(figsize=figsize or (3.5, 6.0))
        ax_main = None
    else:
        fig, ax_main = plt.subplots(figsize=figsize or (max(6, len(other_idx) * 2.0), 6.0))
        ax_alpha = None

    all_records = []

    if ax_main is not None and other_idx:
        # if log_scale:
        #     ylim_main = SHARED_PARAM_YLIM
        # else:
        #     vals_all = np.concatenate(
        #         [data_pb[n].dropna().values for n in other_pb]
        #         + [data_h3b[n].dropna().values for n in other_h3b]
        #     )
        #     pad = 0.1 * (vals_all.max() - vals_all.min() or 1.0)
        #     ylim_main = (vals_all.min() - pad, vals_all.max() + pad)
        ylim_main = y_limits
        recs, _ = _plot_param_group(
            ax_main, data_pb, data_h3b, other_pb, other_h3b, other_labels,
            log_scale, ylim_main, color_pb, color_h3b,
            enforce_ylim=log_scale,
        )
        all_records += recs
        ax_main.set_ylabel(
            'Parameter value' if log_scale else 'Parameter value', fontsize=10,
        )
        if log_scale:
            ax_main.set_ylim(*SHARED_PARAM_YLIM)

    if ax_alpha is not None:
        idx_alpha = names_pb.index('alpha')
        recs, _ = _plot_param_group(
            ax_alpha, data_pb, data_h3b,
            [names_pb[idx_alpha]], [names_h3b[idx_alpha]], ['alpha'],
            False, (0, 100), color_pb, color_h3b,
            enforce_ylim=False,
        )
        all_records += recs
        ax_alpha.set_ylabel('Parameter value', fontsize=10)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_pb, alpha=0.65, edgecolor='#333333', label='PladB'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_h3b, alpha=0.65, edgecolor='#333333', label='H3B'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10, bbox_to_anchor=(0.98, 0.95))

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    return pd.DataFrame(all_records)


def plot_class_specific_comparison(
    df_pb: pd.DataFrame,
    df_h3b: pd.DataFrame,
    class_names: list,
    title_prefix: str,
    class_names_h3b: list | None = None,
    only_successful: bool = True,
    figsize: tuple | None = None,
) -> tuple:
    """
    Convenience wrapper: produces TWO separate figures for class-specific
    scalers — one for delta (decoy strength), one for beta (BP strength) —
    instead of mixing both in a single plot.

    class_names_h3b : list or None
        Pass only if PladB and H3B were fit with different class_names
        (same order as class_names). If None, class_names is used for both.

    Returns
    -------
    (df_stats_delta, df_stats_beta) : tuple of pd.DataFrame
    """
    names_pb_delta = [f'delta_{c}' for c in class_names]
    names_pb_beta  = [f'beta_{c}'  for c in class_names]

    if class_names_h3b is not None:
        names_h3b_delta = [f'delta_{c}' for c in class_names_h3b]
        names_h3b_beta  = [f'beta_{c}'  for c in class_names_h3b]
    else:
        names_h3b_delta = None
        names_h3b_beta  = None

    df_stats_delta = plot_cross_drug_comparison(
        df_pb, df_h3b, names_pb_delta,
        title=f'{title_prefix} — δ (decoy strength)',
        log_scale=False, only_successful=only_successful, figsize=figsize,
        param_names_h3b=names_h3b_delta,
    )
    df_stats_beta = plot_cross_drug_comparison(
        df_pb, df_h3b, names_pb_beta,
        title=f'{title_prefix} — β (BP strength)',
        log_scale=False, only_successful=only_successful, figsize=figsize,
        param_names_h3b=names_h3b_beta,
    )
    return df_stats_delta, df_stats_beta




# def plot_class_specific_comparison_symbolic(
#     df_pb: pd.DataFrame,
#     df_h3b: pd.DataFrame,
#     class_names: list,
#     class_names_h3b: list | None = None,
#     only_successful: bool = True,
#     figsize: tuple | None = None,
#     y_limits = (1e-5, 1e5)
# ) -> tuple:
#     """
#     Come plot_class_specific_comparison, ma con titolo in forma simbolica:
#     'Condition-specific parameters: δ - PladB vs H3B' e
#     'Condition-specific parameters: β - PladB vs H3B'.

#     Produce DUE figure separate (una per delta, una per beta).

#     Returns
#     -------
#     (df_stats_delta, df_stats_beta) : tuple of pd.DataFrame
#     """
#     names_pb_delta = [f'delta_{c}' for c in class_names]
#     names_pb_beta  = [f'beta_{c}'  for c in class_names]

#     if class_names_h3b is not None:
#         names_h3b_delta = [f'delta_{c}' for c in class_names_h3b]
#         names_h3b_beta  = [f'beta_{c}'  for c in class_names_h3b]
#     else:
#         names_h3b_delta = None
#         names_h3b_beta  = None

#     df_stats_delta = plot_cross_drug_comparison(
#         df_pb, df_h3b, names_pb_delta,
#         title='Condition-specific parameters: δ - PladB vs H3B',
#         log_scale=False, only_successful=only_successful, figsize=figsize,
#         param_names_h3b=names_h3b_delta,
#         y_limits = y_limits
#     )
#     df_stats_beta = plot_cross_drug_comparison(
#         df_pb, df_h3b, names_pb_beta,
#         title='Condition-specific parameters: β - PladB vs H3B',
#         log_scale=False, only_successful=only_successful, figsize=figsize,
#         param_names_h3b=names_h3b_beta,
#         y_limits = y_limits
#     )
#     return df_stats_delta, df_stats_beta



def plot_class_specific_comparison_symbolic(
    df_pb: pd.DataFrame,
    df_h3b: pd.DataFrame,
    class_names: list,
    class_names_h3b: list | None = None,
    only_successful: bool = True,
    y_limits = (1e-5, 1e5),
    figsize: tuple | None = None,

) -> tuple:
    """
    Come plot_class_specific_comparison, ma con titolo in forma simbolica:
    'Condition-specific parameters: δ - PladB vs H3B' e
    'Condition-specific parameters: β - PladB vs H3B'.

    Le etichette sull'asse x sono formattate con _format_condition_label
    (es. 'BP_Strong_Decoy 1_PYT_Strong' -> 'Strong BP | Strong Decoy')
    invece del semplice _clean_label usato di default da plot_cross_drug_comparison.

    Produce DUE figure separate (una per delta, una per beta).

    Returns
    -------
    (df_stats_delta, df_stats_beta) : tuple of pd.DataFrame
    """
    names_pb_delta = [f'delta_{c}' for c in class_names]
    names_pb_beta  = [f'beta_{c}'  for c in class_names]

    if class_names_h3b is not None:
        names_h3b_delta = [f'delta_{c}' for c in class_names_h3b]
        names_h3b_beta  = [f'beta_{c}'  for c in class_names_h3b]
    else:
        names_h3b_delta = None
        names_h3b_beta  = None

    # Etichette leggibili derivate dai nomi classe originali (non dalle colonne
    # con prefisso delta_/beta_), usando la stessa formattazione del dual-drug plot.
    display_labels = [_format_condition_label(c) for c in class_names]

    df_stats_delta = plot_cross_drug_comparison(
        df_pb, df_h3b, names_pb_delta,
        title='Condition-specific parameters: δ - PladB vs H3B',
        log_scale=False, only_successful=only_successful, figsize=figsize,
        param_names_h3b=names_h3b_delta,
        labels=display_labels,
        y_limits = y_limits
    )
    df_stats_beta = plot_cross_drug_comparison(
        df_pb, df_h3b, names_pb_beta,
        title='Condition-specific parameters: β - PladB vs H3B',
        log_scale=False, only_successful=only_successful, figsize=figsize,
        param_names_h3b=names_h3b_beta,
        labels=display_labels,
        y_limits = y_limits
    )
    return df_stats_delta, df_stats_beta






def _class_name_to_stratified_key(class_name: str) -> tuple:
    """
    Converte un class_name tipo 'BP_Strong_Decoy 1_PYT_Strong' nella tupla
    chiave usata da stratified_data / stratified_data_h3b:
        ('BP_Strong', 'Decoy 1', 'Strong')

    Split: prima su '_Decoy ' per separare il blocco BP dal blocco Decoy+PYT,
    poi su '_PYT_' per separare il numero di Decoy dal terzo campo (sempre 'Strong').
    """
    bp_part, rest = class_name.split('_Decoy ', 1)
    decoy_num, pyt_label = rest.split('_PYT_', 1)
    return (bp_part, f'Decoy {decoy_num}', pyt_label)


def _experimental_lfc_range(strat_df: pd.DataFrame) -> tuple:
    """
    Trova il minimo e massimo valore LFC tra tutte le colonne 'new_lfc_*'
    di un dataframe stratificato (tutte le sequenze/drug points della condizione).
    """
    lfc_cols = [c for c in strat_df.columns if c.startswith('new_lfc_')]
    if not lfc_cols:
        raise ValueError(
            f"Nessuna colonna 'new_lfc_*' trovata. Colonne disponibili: "
            f"{list(strat_df.columns)[:15]}"
        )
    vals = strat_df[lfc_cols].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.min(vals)), float(np.max(vals))

def build_publication_gof_table(
    best_dict: dict,
    shared_param_names: list,
    class_names: list,
    classes_data: list,
    drug_conc: np.ndarray,
    model_name: str,
    stratified_data: dict,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    """
    Costruisce una tabella publication-ready con, per ciascuna condizione
    sperimentale (classe):
        - Condition           : etichetta leggibile (es. 'Strong BP | Strong Decoy')
        - Experimental Range   : [min, max] degli LFC osservati su TUTTE le
                                  sequenze/drug points della condizione,
                                  presa da stratified_data
        - Range Length          : max - min
        - RMSE                  : RMSE fra media osservata e predizione del
                                   modello per quella classe

    Tutti i valori numerici sono arrotondati alla seconda cifra decimale.

    Parameters
    ----------
    best_dict : dict
        Best-parameter dictionary del fit (shared params + delta_/beta_ per classe).
    shared_param_names : list of str
    class_names : list of str
        Chiavi 'raw' tipo 'BP_Strong_Decoy 1_PYT_Strong', allineate a classes_data.
    classes_data : list of pd.DataFrame
        Dati di classe usati per il fit (stesso formato di goodness_of_fit_multiclass).
    drug_conc : np.ndarray
    model_name : str
    stratified_data : dict
        Dizionario chiave=tupla (BP, Decoy N, PYT_label) -> dataframe con colonne
        'new_lfc_*', da cui si estrae il range sperimentale grezzo.
    fixed_params : dict or None

    Returns
    -------
    pd.DataFrame con colonne:
        Condition, Experimental Range, Range Length, RMSE
    """
    rows = []

    for cname, df in zip(class_names, classes_data):

        # --- RMSE ---
        obs = _observed_lfc(df, drug_conc)
        class_fixed = {
            'delta': best_dict[f'delta_{cname}'],
            'beta' : best_dict[f'beta_{cname}'],
        }
        if fixed_params is not None:
            class_fixed = {**fixed_params, **class_fixed}
        shared_values = [best_dict[name] for name in shared_param_names]
        pred = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_param_names,
            drug_conc              = drug_conc,
            model_name             = model_name,
            fixed_params           = class_fixed,
        )
        rmse = float(np.sqrt(np.mean((obs['means'] - np.asarray(pred)) ** 2)))

        # --- Experimental range da stratified_data ---
        strat_key = _class_name_to_stratified_key(cname)
        if strat_key not in stratified_data:
            raise KeyError(
                f"Chiave {strat_key} (derivata da class_name '{cname}') non trovata "
                f"in stratified_data. Chiavi disponibili (prime 10): "
                f"{list(stratified_data.keys())[:10]}"
            )
        strat_df = stratified_data[strat_key]
        lfc_min, lfc_max = _experimental_lfc_range(strat_df)
        range_length = lfc_max - lfc_min

        rows.append({
            'Condition'          : _format_condition_label(cname),
            'Experimental Range' : f'[{lfc_min:.2f}, {lfc_max:.2f}]',
            'Range Length'       : round(range_length, 2),
            'RMSE'               : round(rmse, 2),
        })

    return pd.DataFrame(rows)