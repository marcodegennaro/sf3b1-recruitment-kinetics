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

import warnings
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split

# Personal libraries 

import model_database as db
import simulation_core as sim
import fitting_core as fit

# To use directely module variation
%load_ext autoreload
%autoreload 2
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

# --------------------------------------------------------------------------
# PATH_DATASET uploading 
# --------------------------------------------------------------------------

# first normalization method
PATH_RAW_DATASET = r"C:\Users\marco\Desktop\Tesi + Laurea\.LabWork\Data\Data from Suzanne\fitness_calc_PB_H3B.xlsx"
PATH_LOG_READS_NORMALIZED = r"C:/Users/marco/Desktop/Tesi + Laurea/.LabWork/Results/log_reads_norm.csv"
PATH_LFC = "C:/Users/marco/Desktop/Tesi + Laurea/.LabWork/Results/lfc_results.csv"

# secondth normalization method
path_H = r"C:\Users\marco\Desktop\Tesi + Laurea\.LabWork\Code\4. Data Transformation\normalized_matrix_H.csv"
path_PB = r"C:\Users\marco\Desktop\Tesi + Laurea\.LabWork\Code\4. Data Transformation\normalized_matrix_PB.csv"

# We load the raw file with no changes.
# data_raw will never be modified — it stays as the original reference.
data_raw = pd.read_excel(PATH_RAW_DATASET)
data_reads_norm = pd.read_csv(PATH_LOG_READS_NORMALIZED)
data_lfc = pd.read_csv(PATH_LFC)

data_reads_norm.columns = data_reads_norm.columns.str.replace(
    "output_", "norm_reads_", regex=False
)

data_lfc.drop(columns=["DECOY", "PYT", "ANCHOR"], inplace=True)

data_H_new = pd.read_csv(path_H)
data_PB_new = pd.read_csv(path_PB)

df_unito = pd.merge(data_H_new, data_PB_new, on='seq_id', how='inner')
df_unito.drop('norm_Control_y', axis=1, inplace=True)
df_unito.rename(columns={'norm_Control_x': 'norm_Control'}, inplace=True)
df_unito.columns = df_unito.columns.str.replace(
    "norm_", "new_norm_reads_", regex=False
)

# --------------------------------------------------------------------------
# Data Merging ( unify the datasets)
# --------------------------------------------------------------------------
df_CLR_norm = pd.merge(data_reads_norm, data_lfc, on="seq_id")
df_CLR_norm['seq_id'] = df_CLR_norm['seq_id'].astype(int)
data = pd.merge(data_raw, df_CLR_norm, on="seq_id")
data = pd.merge(data, df_unito, on="seq_id")


# --------------------------------------------------------------------------
# Data Filtering
# --------------------------------------------------------------------------

MIN_READS_THRESHOLD = 20

n_total   = len(data_raw)
mask_ok   = data["Input_counts"] >= MIN_READS_THRESHOLD
n_removed = (~mask_ok).sum()
n_kept    = mask_ok.sum()

data_filtered = data[mask_ok].copy().reset_index(drop=True)
data_filtered['ANCHOR'] = data_filtered['ANCHOR'].astype(str)

# --------------------------------------------------------------------------
# addign the new normalization lfc method 
# --------------------------------------------------------------------------


numeri_pb = [1, 2, 4, 8]

# Costante per evitare log(0) o divisioni per 0 (pseudocount standard = 1)
pseudocount = 1

for num in numeri_pb:
    col_trattata = f'new_norm_reads_PB{num}'
    col_controllo = 'new_norm_reads_Control'
    nuova_colonna = f'new_lfc_PB{num}'
    
    # Calcolo del Log2 Fold Change con pseudocount
    data_filtered[nuova_colonna] = np.log2(
        (data_filtered[col_trattata] + pseudocount) / (data_filtered[col_controllo] + pseudocount)
    )

# --------------------------------------------------------------------------
# Adding the BP feature 
# --------------------------------------------------------------------------

def filter_sequence(sequence: str, to_find: str) -> str:
    """Optimizes the input sequence for the SVM finder by removing the final

    section that follows the specified sub-string (e.g., 'CAG').

    Parameters:
        - sequence: str, the string to be cut
        - to_find: str, the sub-string to locate

    Returns:
        - str: The subsetted string, including the 'to_find' sequence at the
        end.
    Raise:
        - ValueError if the to_find string isn't in the sequence
    """
    # Find the index of the last occurrence (searching from right to left)
    idx = sequence.rfind(to_find)

    # If the sub-string is not found, rfind returns -1.
    if idx == -1:
    # In this case, we return the original sequence.
        raise ValueError(f"The string {to_find} is not in {sequence}.")

    # Slice the string to keep everything from the beginning
    # up to the end of 'to_find' (inclusive)
    return sequence[: idx + len(to_find)]

def BP_extractor(data):
    final_str = 'CAG'
    linker = 'GCG'
    strong_pyt = 'UUUUUCUUUUCUUUU'.replace('U', 'T')
    
    suffix_len = len(linker + strong_pyt + final_str)  # 21nt
    
    return data['full_seq'].apply(lambda seq: seq[-(suffix_len+7):-suffix_len])



pattern = 'CAG'

data_filtered['full_seq'] = data_filtered['full_seq'].apply(
    lambda seq: filter_sequence(seq, pattern)
)

data_filtered['BP_extracted'] = BP_extractor(data_filtered)


focus_sub_data = data_filtered[(data_filtered['DECOY'] == 'Decoy 7') & (data_filtered['PYT'] == 'Strong')]
data_filtered['BP_extracted'] = BP_extractor(data_filtered)

focus_sub_data = data_filtered[(data_filtered['DECOY'] == 'Decoy 7') & (data_filtered['PYT'] == 'Strong')]

# 1. Compute mean normalized reads per unique BP — reference condition only
bp_distribution = focus_sub_data.groupby('BP_extracted')['norm_reads_Ctrl'].mean()

# 2. Sort from highest to lowest
bp_ranked = bp_distribution.sort_values(ascending=False)

ranked_values = bp_ranked.values
ranked_categories = bp_ranked.index.tolist()

# 3. Rank-based quartile cuts
total_unique_bps = len(ranked_categories)
cut_strong       = int(total_unique_bps * 0.25)
cut_intermediate = int(total_unique_bps * 0.75)

strong_bp_names       = set(ranked_categories[:cut_strong])
intermediate_bp_names = set(ranked_categories[cut_strong:cut_intermediate])
weak_bp_names         = set(ranked_categories[cut_intermediate:])

# 4. Assign class to the full dataset
conditions = [
    data_filtered['BP_extracted'].isin(strong_bp_names),
    data_filtered['BP_extracted'].isin(intermediate_bp_names),
    data_filtered['BP_extracted'].isin(weak_bp_names)
]
choices = ['BP_Strong', 'BP_Intermediate', 'BP_Weak']

data_filtered['BP_class'] = np.select(conditions, choices, default='Unknown')



# --------------------------------------------------------------------------
# LFC old and new normalization 
# --------------------------------------------------------------------------
lfc = [col for col in data_filtered.columns if 'lfc' in col]            # old normalization 
lfc_new = [col for col in data_filtered.columns if 'new_lfc' in col]    # new normalization 




# --------------------------------------------------------------------------
# Stratification of classes (BP_class x DECOY x PYT)
# --------------------------------------------------------------------------

variables = ['BP_class', 'DECOY', 'PYT']
normalized = [col for col in data_filtered.columns if 'norm_reads' in col]

focus_cols = ['seq_id', *variables, *lfc_new, *normalized]                          # using new lfc
df_lfc = data_filtered.loc[:, focus_cols]




unique_values = [df_lfc[col].dropna().unique() for col in variables]
combinations_names = list(itertools.product(*unique_values))

stratified_data =  {}
for key_classes, subset in df_lfc.groupby(variables):
    stratified_data[key_classes] = subset.drop(
        columns = variables
    )




# --------------------------------------------------------------------------
# CONSTANT definition
# --------------------------------------------------------------------------
DRUG_CONCENTRATIONS_raw = [1e-11, 1e-10, 2e-10, 4e-10, 1e-09]
def scale_drug_concentration(drug_linear, 
                             min_real=1e-11, 
                             max_real=1e-9, 
                             min_scaled=0., 
                             max_scaled=100.0):
    """
    Scala la concentrazione lineare del farmaco in un range lineare personalizzato (es. 1-100)
    passando prima per lo spazio logaritmico, per preservare la sensibilità alle basse dosi.
    """
    # 1. Passaggio allo spazio logaritmico (esponenti)
    log_drug = np.log10(drug_linear)
    log_min = np.log10(min_real)  # -11
    log_max = np.log10(max_real)  # -9
    
    # 2. Traslazione e riscalamento lineare dell'esponente nel nuovo range [min_scaled, max_scaled]
    drug_scaled = min_scaled + (log_drug - log_min) * (max_scaled - min_scaled) / (log_max - log_min)
    
    return drug_scaled

DRUG_CONCENTRATIONS = scale_drug_concentration(DRUG_CONCENTRATIONS_raw)


