##############################################################################
# COMPOSITIONAL DATA ANALYSIS PIPELINE — STREAMLINED VERSION (v2)
#
# Steps: zero check -> CLR transform -> stability scoring (CLR variance +
# raw log-SD, diagnostic only) -> elbow-based selection -> leave-one-
# condition-out validation -> intersection of the invariant sets found in
# PladB and H3B -> reference-based normalization using ONLY that shared
# invariant set -> export.
#
# No biological cross-mapping: reference sequences are defined purely from
# CLR-variance stability, and only those invariant in BOTH drug series are
# used to normalize either series.
##############################################################################

library(readxl)
library(zCompositions)
library(matrixStats)
library(tidyverse)

PATH <- "C:/Users/marco/Desktop/Tesi + Laurea/.LabWork/Data/Data from Suzanne/fitness_calc_PB_H3B.xlsx"
df   <- as.data.frame(read_excel(PATH))

# ──────────────────────────────────────────────────────────────────────────
# Helper 1: elbow detection (kneedle method, on log-scaled scores)
# Finds the point of maximum distance from the straight line connecting the
# first and last point of the sorted-score curve, computed on log10(score)
# so it matches the log-scale plot and is not biased by the several orders
# of magnitude the scores span.
# ──────────────────────────────────────────────────────────────────────────
find_elbow <- function(sorted_scores, log_scale = TRUE) {
  n <- length(sorted_scores)
  x <- (seq_len(n) - 1) / (n - 1)
  
  y_raw <- if (log_scale) log10(pmax(sorted_scores, min(sorted_scores[sorted_scores > 0]) * 1e-3)) else sorted_scores
  y <- (y_raw - min(y_raw)) / (max(y_raw) - min(y_raw))
  
  line_vec <- c(x[n] - x[1], y[n] - y[1])
  line_vec <- line_vec / sqrt(sum(line_vec^2))
  
  vec_from_first <- cbind(x - x[1], y - y[1])
  proj_len   <- vec_from_first %*% line_vec
  proj       <- outer(as.numeric(proj_len), line_vec)
  dist_line  <- sqrt(rowSums((vec_from_first - proj)^2))
  
  idx <- which.max(dist_line)
  list(index = idx, value = sorted_scores[idx])
}

# ──────────────────────────────────────────────────────────────────────────
# Helper 2: stability scores for one log-count matrix
#   - CLR variance (Method 1): variance of the per-condition centered
#     values. This is the methodologically valid CoDA stability metric and
#     is the ONLY one used to define the invariant / reference sequences.
#   - Raw log-SD (Method 2): SD on the un-centered log-counts. Kept only as
#     a reported diagnostic (it is not corrected for per-condition library
#     size, so it is not a valid standalone stability metric).
# ──────────────────────────────────────────────────────────────────────────
compute_stability <- function(log_mat) {
  col_means  <- colMeans(log_mat)
  clr_matrix <- sweep(log_mat, 2, col_means, FUN = "-")
  
  list(
    clr_matrix = clr_matrix,
    var_score  = apply(clr_matrix, 1, var),   # Method 1 — used downstream
    sd_score   = apply(log_mat, 1, sd)         # Method 2 — diagnostic only
  )
}

# ──────────────────────────────────────────────────────────────────────────
# Phase A: stability analysis for one drug series (no normalization yet)
# ──────────────────────────────────────────────────────────────────────────
analyze_stability <- function(df, count_cols, cond_labels, label = "run") {
  
  cat("\n================ ANALYSIS:", label, "================\n")
  
  ## 1. Zero check ------------------------------------------------------------
  count_matrix <- as.data.frame(df[, count_cols])
  rownames(count_matrix) <- df$seq_id
  colnames(count_matrix) <- cond_labels
  
  n_zeros <- sum(count_matrix == 0)
  cat("Zeros found:", n_zeros, "(", round(mean(count_matrix == 0) * 100, 3), "% )\n")
  
  if (n_zeros > 0) {
    count_matrix <- cmultRepl(count_matrix, method = "CZM", output = "p-counts")
  }
  count_mat <- as.matrix(count_matrix)
  
  ## 2. Log-transform + CLR + stability scores --------------------------------
  log_mat <- log(count_mat)
  st      <- compute_stability(log_mat)
  n_seqs  <- nrow(log_mat)
  
  ## 3. Elbow-based selection (Method 1 = reference; Method 2 = diagnostic) ---
  rank_var <- sort(st$var_score, decreasing = FALSE)
  rank_sd  <- sort(st$sd_score,  decreasing = FALSE)
  
  elbow_var <- find_elbow(rank_var)
  elbow_sd  <- find_elbow(rank_sd)
  
  top_stable_var <- names(rank_var)[1:elbow_var$index]   # <- the invariant set used downstream
  top_stable_sd  <- names(rank_sd)[1:elbow_sd$index]      # <- diagnostic only
  
  cat("Method 1 (CLR variance) elbow: n =", elbow_var$index,
      "(", round(elbow_var$index / n_seqs * 100, 2), "% ) | threshold =",
      signif(elbow_var$value, 4), "\n")
  cat("Method 2 (raw log-SD, diagnostic) elbow: n =", elbow_sd$index,
      "(", round(elbow_sd$index / n_seqs * 100, 2), "% ) | threshold =",
      signif(elbow_sd$value, 4), "\n")
  
  par(mfrow = c(1, 2))
  plot(seq_along(rank_var), rank_var, type = "l", log = "y", col = "darkblue",
       xlab = "Sequence rank", ylab = "CLR variance (log scale)",
       main = paste0(label, " — Method 1: CLR variance"))
  abline(v = elbow_var$index, col = "red", lty = 2)
  abline(h = elbow_var$value, col = "orange", lty = 2)
  
  plot(seq_along(rank_sd), rank_sd, type = "l", log = "y", col = "darkgreen",
       xlab = "Sequence rank", ylab = "Raw log-SD (log scale)",
       main = paste0(label, " — Method 2 (diagnostic): log-SD"))
  abline(v = elbow_sd$index, col = "red", lty = 2)
  abline(h = elbow_sd$value, col = "orange", lty = 2)
  par(mfrow = c(1, 1))
  
  ## 4. Method comparison (diagnostic only, does not affect selection) --------
  overlap_seqs  <- intersect(top_stable_var, top_stable_sd)
  spearman_corr <- cor(st$var_score, st$sd_score, method = "spearman")
  
  cat("Diagnostic overlap Method1/Method2:", length(overlap_seqs), "sequences (",
      round(length(overlap_seqs) / length(top_stable_var) * 100, 1),
      "% of Method-1 set )\n")
  cat("Spearman correlation between the two rankings (all seqs):",
      round(spearman_corr, 3), "\n")
  
  ## 5. Leave-one-condition-out validation of the Method-1 (CLR variance) set -
  loo_df <- data.frame()
  for (k in seq_along(cond_labels)) {
    log_mat_loo <- log_mat[, -k, drop = FALSE]
    st_loo      <- compute_stability(log_mat_loo)
    rank_loo    <- sort(st_loo$var_score, decreasing = FALSE)
    elbow_loo   <- find_elbow(rank_loo)
    top_loo     <- names(rank_loo)[1:elbow_loo$index]
    
    overlap_n   <- sum(top_stable_var %in% top_loo)
    overlap_pct <- round(overlap_n / length(top_stable_var) * 100, 1)
    
    loo_df <- rbind(loo_df, data.frame(removed = cond_labels[k],
                                       n_top_loo = length(top_loo),
                                       overlap_n = overlap_n,
                                       overlap_pct = overlap_pct))
  }
  cat("\nLeave-one-condition-out summary:\n"); print(loo_df)
  cat("Mean overlap:", round(mean(loo_df$overlap_pct), 1), "% | Min overlap:",
      round(min(loo_df$overlap_pct), 1), "%\n")
  
  invisible(list(
    label          = label,
    count_mat      = count_mat,
    cond_labels    = cond_labels,
    stability      = st,
    elbow_var      = elbow_var,
    elbow_sd       = elbow_sd,
    top_stable_var = top_stable_var,   # invariant set (CLR variance) for this series
    top_stable_sd  = top_stable_sd,    # diagnostic only
    spearman_corr  = spearman_corr,
    loo_df         = loo_df
  ))
}

# ──────────────────────────────────────────────────────────────────────────
# Phase B: normalize a series using a GIVEN reference set (geometric mean)
# ──────────────────────────────────────────────────────────────────────────
normalize_and_export <- function(result, reference_ids, out_path) {
  
  count_mat   <- result$count_mat
  cond_labels <- result$cond_labels
  
  denom_mat <- count_mat[rownames(count_mat) %in% reference_ids, , drop = FALSE]
  stopifnot(nrow(denom_mat) > 0)   # reference set must actually be present
  
  geom_mean_ref <- exp(colMeans(log(denom_mat)))
  reads_norm    <- sweep(count_mat, 2, geom_mean_ref, FUN = "/")
  
  cat("\n[", result$label, "] Reference sequences used:", nrow(denom_mat), "\n")
  cat("[", result$label, "] Geometric mean per condition:\n")
  print(round(geom_mean_ref, 3))
  
  out_df <- as.data.frame(reads_norm)
  colnames(out_df) <- paste0("norm_", cond_labels)
  out_df <- tibble::rownames_to_column(out_df, "seq_id")
  
  write.csv(out_df, out_path, row.names = FALSE)
  cat("[", result$label, "] Saved normalized matrix to:", out_path, "\n")
  
  invisible(out_df)
}

# ──────────────────────────────────────────────────────────────────────────
# Run Phase A on both series
# ──────────────────────────────────────────────────────────────────────────
pb_stab <- analyze_stability(
  df          = df,
  count_cols  = c("output_Ctrl", "output_PB1", "output_PB2", "output_PB4", "output_PB8"),
  cond_labels = c("Control", "PB1", "PB2", "PB4", "PB8"),
  label       = "PladB (PB series)"
)

h_stab <- analyze_stability(
  df          = df,
  count_cols  = c("output_Ctrl", "output_H1", "output_H2", "output_H3", "output_H4", "output_H5"),
  cond_labels = c("Control", "H1", "H2", "H3", "H4", "H5"),
  label       = "H3B (H series)"
)

# ──────────────────────────────────────────────────────────────────────────
# Intersection of the invariant sets (CLR variance) between PladB and H3B —
# this shared set is the ONLY reference used for normalization below.
# ──────────────────────────────────────────────────────────────────────────
common_invariant <- intersect(pb_stab$top_stable_var, h_stab$top_stable_var)

cat("\n================ PladB vs H3B — shared invariant set ================\n")
cat("Invariant in PladB:      ", length(pb_stab$top_stable_var), "\n")
cat("Invariant in H3B:        ", length(h_stab$top_stable_var), "\n")
cat("Invariant in BOTH:       ", length(common_invariant), "\n")
cat("As % of PladB set:       ", round(length(common_invariant) / length(pb_stab$top_stable_var) * 100, 2), "%\n")
cat("As % of H3B set:         ", round(length(common_invariant) / length(h_stab$top_stable_var) * 100, 2), "%\n")
cat("As % of total sequences: ", round(length(common_invariant) / nrow(df) * 100, 3), "%\n")

# ──────────────────────────────────────────────────────────────────────────
# Phase B: normalize both series using ONLY the shared invariant set
# ──────────────────────────────────────────────────────────────────────────
pb_norm <- normalize_and_export(pb_stab, common_invariant, "normalized_matrix_PB.csv")
h_norm  <- normalize_and_export(h_stab,  common_invariant, "normalized_matrix_H.csv")