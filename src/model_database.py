
# =============================================================================
# Catalogue – 11 explicit models ( to update )
# =============================================================================
#
#  Naming convention:  param[function_name]  joined by  __
#  e.g.  "k1[hill_decay]__k2[hill_activation]"
#
# Each entry:
#   "mods"             : {param_name: (func_name, callable)}
#   "description"      : human-readable summary
#   "params_modulated" : list of parameter names touched
#   "n_params"         : int

from typing import Callable


# =============================================================================
# Drug modulation primitives  (only the three functions actually used)
# =============================================================================

def _hill_decay(k: float, drug: float, *, K_hill: float, n: int, **__) -> float:
    """k · (1 – drugⁿ / (K_hillⁿ + drugⁿ))"""
    if drug <= 0:
        return k
    return k * (1.0 - drug**n / (K_hill**n + drug**n))

# def _hyperbolic_decrease(k: float, drug: float, *, alpha: float, **__) -> float:
#     """k / (α · drug) """
#     if drug < 1e-12: 
#         return k
#     return k / (alpha * drug)

def _linear_decrease(k: float, drug: float, *, alpha: float, **__) -> float:

    """k / (α - drug) """
    if drug == 0:
        return k
    else:
        return k / (alpha * drug)


def _linear_increase(k: float, drug: float, *, alpha: float, **__) -> float:
    """k + α · drug"""
    return k + alpha * drug


def _hill_activation(
    k: float, drug: float, *, Vmax: float, K_hill: float, n: int, **__
) -> float:
    
    """k + Vmax · drugⁿ / (K_hillⁿ + drugⁿ) """
    if drug <= 0:
        return k
    return k + Vmax * drug**n / (K_hill**n + drug**n)



# =============================================================================
# Database generating functions
# =============================================================================

def _make_entry(mods_raw: list[tuple[str, str, Callable]], description: str, nodes_num: int ) -> tuple[str, dict]:
    mods = {p: (fname, func) for p, fname, func in mods_raw}
    
    params_str = "__".join(f"{p}[{fname}]" for p, fname, _ in mods_raw)
    name = f"{params_str}__{nodes_num}nodes"

    return name, {
        "mods":             mods,
        "description":      description,
        "params_modulated": [p for p, _, _ in mods_raw],
        "n_params":         len(mods_raw),
        "nodes_number":     nodes_num,
    }

def _database_generation():

    return dict([

        # ── 3 states topology─────────────────────────────────

        _make_entry(
            [("k1",  "hill_decay",       _hill_decay)],
            "Drug modulates: k1 ↓ via hill_decay",
            nodes_num = 3
        ),  # Model 01

        _make_entry(
            [("k1",  "linear_decrease",       _linear_decrease)],
            "Drug modulates: k1 ↓ via hill_decay",
            nodes_num = 3
        ),  # Model 02

        _make_entry(
            [("k_1", "linear_increase",  _linear_increase)],
            "Drug modulates: k_1 ↑ via linear_increase",
            nodes_num = 3
        ),  # Model 02

        _make_entry(
            [("k_1", "hill_activation",  _hill_activation)],
            "Drug modulates: k_1 ↑ via hill_activation",
            nodes_num = 3
        ),  # Model 03

        _make_entry(
            [("k_2", "hill_decay",       _hill_decay)],
            "Drug modulates: k_2 ↓ via hill_decay",
            nodes_num = 3
        ),  # Model 04


        _make_entry(
            [("k2",  "hill_activation",  _hill_activation)],
            "Drug modulates: k2 ↑ via hill_activation",
            nodes_num = 3
        ),  # Model 06

        # ──  4 states topology─────────────────────────────────

        _make_entry(
            [("k3",  "hill_decay",  _hill_decay)],
            "Drug modulates: k3 ↓ via hill_decay",
            nodes_num = 4,
        ),  # Model 14

        _make_entry(
            [("k3",  "linear_decrease",  _linear_decrease)],
            "Drug modulates: k3  via linear decrease",
            nodes_num = 4,
        ),  # Model 14

        _make_entry(
            [("k4",  "hill_activation",  _hill_activation)],
            "Drug modulates: k4 ↓ via hill_decay",
            nodes_num = 4,
        ),  # Model 15

                _make_entry(
            [("k1",  "linear_decrease",       _linear_decrease)],
            "Drug modulates: k1 ↓ via hill_decay",
            nodes_num = 4
        ),  # Model 02

        _make_entry(
            [("k_1", "linear_increase",  _linear_increase)],
            "Drug modulates: k_1 ↑ via linear_increase",
            nodes_num = 4
        ),  # Model 02



    ])

ALL_MODELS = _database_generation()



# =============================================================================
# Catalogue inspection utilities
# =============================================================================

def list_models(
    verbose: bool = False,
    nodes_num: int | None = None,
    show_formula: bool = False,          
) -> None:
    models = ALL_MODELS

    if nodes_num is not None:
        models = {k: v for k, v in models.items() if v["nodes_number"] == nodes_num}

    print(f"{'─'*70}")
    print(f"  Models displayed : {len(models)}  (total in catalogue: {len(ALL_MODELS)})")
    print(f"{'─'*70}")
    for i, (name, info) in enumerate(models.items(), 1):
        print(f"  [{i:02d}]  {name}")
        if show_formula:                 # ← condizione aggiunta
            for param, (fname, func) in info["mods"].items():
                formula = func.__doc__.splitlines()[0].strip()
                print(f"        {param} [{fname}] → {formula}")
        if verbose:
            print(f"        └─ {info['description']}")
    print(f"{'─'*70}")




# =============================================================================
# Entry Point 
# =============================================================================

def main():
    print('Module Succesfully run!! ')

if __name__ == "__main__":
    main()
