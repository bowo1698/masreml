# examples/02_basic_binary_trait.R
#
# REML-BLUP genomic prediction for a binary (0/1) disease trait using SNP
# additive markers. The binary GLMM is solved via a single-step Laplace
# approximation on the liability scale.
#
# Demonstrates:
#   1. Simulating a binary phenotype from an underlying liability model
#   2. Fitting binary GBLUP with logit and probit link functions
#   3. Extracting h2 on both liability and observed (0/1) scales
#   4. NEW predict() API: auto-compute metrics with `y_new` argument
#   5. $prob field (predicted probabilities P(y=1))
#   6. In-sample mode: predict(fit) returns training_metrics
#   7. Calibration diagnostics: the "bias" slope is the calibration slope
#      (1.0 = well-calibrated, <1 over-dispersion, >1 under-dispersion)
#   8. Visualization: ROC curve + probability distribution + calibration plot
#
# Reference: Dempster & Falconer (1950) Ann. Hum. Genet. 31:195-203
#            Da (2015) BMC Genet. 16:144
#            Steyerberg (2010) Clinical Prediction Models — calibration slope
# Requires : masreml  (devtools::install_github("bowo1698/masreml"))

library(masreml)

set.seed(42)
n          <- 500   # total individuals
p_snp      <- 200   # number of SNP markers
p_causal   <- 10    # number of causal SNPs
prevalence <- 0.30  # target disease prevalence

# ── Simulate SNP genotype matrix (0 / 1 / 2 allele counts) ──────────────────
cat("Simulating SNP genotype data...\n")
W <- matrix(
  rbinom(n * p_snp, size = 2, prob = 0.3),
  nrow = n, ncol = p_snp
)
rownames(W) <- paste0("ind", seq_len(n))
colnames(W) <- paste0("SNP", seq_len(p_snp))

# ── Simulate binary disease phenotype (liability threshold model) ─────────────
# Continuous liability = genetic component + environmental noise
# Threshold applied at (1 - prevalence) quantile to obtain 0/1 status
cat("Simulating binary disease phenotype (liability threshold model)...\n")
beta_causal <- rnorm(p_causal, mean = 0, sd = 0.3)
W_scaled    <- scale(W[, seq_len(p_causal)])
g_true      <- as.numeric(W_scaled %*% beta_causal)   # true liability BV
e           <- rnorm(n, mean = 0, sd = 1)
liability   <- g_true + e

threshold        <- quantile(liability, probs = 1 - prevalence)
y_binary         <- as.integer(liability > threshold)
names(y_binary)  <- rownames(W)
n_cases          <- sum(y_binary)
n_controls       <- n - n_cases
prev_obs         <- mean(y_binary)
h2_liability_sim <- var(g_true) / var(liability)

cat(sprintf(
  "  Prevalence: %.1f%% (cases=%d, controls=%d)\n",
  prev_obs * 100, n_cases, n_controls
))
cat(sprintf("  Simulated h2 (liability scale): %.4f\n", h2_liability_sim))

# ── Train / test split (80 / 20), stratified by case/control ─────────────────
set.seed(123)
idx_cases    <- which(y_binary == 1)
idx_controls <- which(y_binary == 0)
idx_train <- c(
  sample(idx_cases,    floor(0.8 * length(idx_cases))),
  sample(idx_controls, floor(0.8 * length(idx_controls)))
)
W_train <- W[idx_train, ]
y_train <- y_binary[idx_train]
W_test  <- W[-idx_train, ]
y_test  <- y_binary[-idx_train]

cat(sprintf(
  "  Training: n=%d (cases=%d) | Test: n=%d (cases=%d)\n\n",
  length(y_train), sum(y_train), length(y_test), sum(y_test)
))

# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Binary GBLUP with LOGIT link
# ════════════════════════════════════════════════════════════════════════════

cat("== MODEL 1: Binary trait with LOGIT link ==\n")
cat("(sigma2_e fixed at pi^2/3 on the logistic liability scale)\n")

fit_logit <- masreml(
  y       = y_train,
  markers = list(snp_add = W_train),
  method  = "auto",      # defaults to HE-regression for binary traits
  trait   = "binary",
  link    = "logit",
  solver  = "auto"
)
# ↑ Note: auto post-fit summary banner is printed (verbose=TRUE by default).
#   Set verbose=FALSE to silence it (e.g. inside loops/CV).

cat("\n")
summary(fit_logit)
# The summary now includes a "Training Performance (observed/probability scale)"
# block where bias = calibration slope (1.0 = well-calibrated).

# ── In-sample mode (NEW): shortcut to training_metrics ────────────────
cat("\n-- In-sample shortcut: predict(fit) ----------------------------\n")
in_logit <- predict(fit_logit)
cat(sprintf("  Eval scope    : %s\n", in_logit$eval_scope))
cat(sprintf("  Has truth     : %s\n", in_logit$has_truth))
cat(sprintf("  Training AUC  : %.4f\n", in_logit$metrics$AUC))
cat(sprintf("  Training bias : %.4f  (calibration slope)\n",
            in_logit$metrics$bias))

# ── Test-set prediction: NEW one-call API with y_new ──────────────────
cat("\n-- Test-set prediction with auto-metrics (NEW API) -------------\n")
pred_logit <- predict(
  fit_logit,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train),
  y_new         = y_test                       # ← triggers auto-metrics
)
print(pred_logit)
# pred_logit$prob          — fitted probabilities P(y=1)
# pred_logit$GEBV          — liability-scale GEBV
# pred_logit$metrics$bias  — calibration slope on observed scale
# pred_logit$metrics$AUC, $R2, $RMSE — auto-computed since y_new was supplied

# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Binary GBLUP with PROBIT link
# ════════════════════════════════════════════════════════════════════════════

cat("\n\n== MODEL 2: Binary trait with PROBIT link ==\n")
cat("(sigma2_e fixed at 1.0 on the standard normal liability scale)\n")

fit_probit <- masreml(
  y       = y_train,
  markers = list(snp_add = W_train),
  method  = "auto",
  trait   = "binary",
  link    = "probit",
  solver  = "auto",
  verbose = FALSE                     # silence banner for compactness
)

pred_probit <- predict(
  fit_probit,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train),
  y_new         = y_test
)

# ════════════════════════════════════════════════════════════════════════════
# Comparison: Logit vs Probit
# ════════════════════════════════════════════════════════════════════════════

cat("\n\n== Comparison: Logit vs Probit (test set) ==\n")
comp <- data.frame(
  Link              = c("logit", "probit"),
  h2_liability      = round(c(fit_logit$binary$h2_liability,
                              fit_probit$binary$h2_liability),  4),
  h2_observed       = round(c(fit_logit$binary$h2_observed,
                              fit_probit$binary$h2_observed),   4),
  AUC_train         = round(c(fit_logit$binary$auc,
                              fit_probit$binary$auc),           4),
  AUC_test          = round(c(pred_logit$metrics$AUC,
                              pred_probit$metrics$AUC),         4),
  R2_test           = round(c(pred_logit$metrics$R2,
                              pred_probit$metrics$R2),          4),
  RMSE_test         = round(c(pred_logit$metrics$RMSE,
                              pred_probit$metrics$RMSE),        4),
  calib_slope_test  = round(c(pred_logit$metrics$bias,          # ← key column
                              pred_probit$metrics$bias),        4)
)
print(comp, row.names = FALSE)

cat(sprintf("\nSimulated h2 (liability): %.4f\n", h2_liability_sim))
cat("\nInterpretation of `calib_slope_test`:\n")
cat("  = 1.0 : predicted probabilities well-calibrated\n")
cat("  < 1.0 : over-dispersion — predictions too extreme (over-confident)\n")
cat("  > 1.0 : under-dispersion — predictions too compressed toward base rate\n")

# ════════════════════════════════════════════════════════════════════════════
# Visualisation — three panels
# ════════════════════════════════════════════════════════════════════════════

# Helper: empirical ROC curve (base R, no external dependency)
.roc_curve <- function(y, prob) {
  thresholds <- sort(unique(prob), decreasing = TRUE)
  tpr <- fpr <- numeric(length(thresholds) + 2)
  tpr[1] <- fpr[1] <- 0
  for (i in seq_along(thresholds)) {
    pred_pos   <- prob >= thresholds[i]
    tpr[i + 1] <- sum(pred_pos & y == 1) / sum(y == 1)
    fpr[i + 1] <- sum(pred_pos & y == 0) / sum(y == 0)
  }
  tpr[length(tpr)] <- fpr[length(fpr)] <- 1
  list(fpr = fpr, tpr = tpr)
}

# Helper: decile-binned calibration curve
.calibration_curve <- function(y, prob, n_bins = 10) {
  qs    <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1),
                    na.rm = TRUE)
  qs[1] <- qs[1] - 1e-9   # ensure left-closed binning
  bin   <- cut(prob, breaks = qs, include.lowest = TRUE, labels = FALSE)
  agg   <- data.frame(bin = bin, y = y, p = prob)
  agg   <- aggregate(cbind(y = y, p = p) ~ bin, data = agg, FUN = mean)
  agg
}

par(mfrow = c(1, 3))

# Panel 1 — ROC curves
roc_l <- .roc_curve(y_test, pred_logit$prob)
roc_p <- .roc_curve(y_test, pred_probit$prob)
plot(roc_l$fpr, roc_l$tpr, type = "l", lwd = 2, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "ROC Curves (Test)",
     xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(roc_p$fpr, roc_p$tpr, lwd = 2, col = "firebrick")
abline(0, 1, lty = 2, col = "grey60")
legend("bottomright",
       legend = c(
         sprintf("Logit  (AUC=%.3f)", pred_logit$metrics$AUC),
         sprintf("Probit (AUC=%.3f)", pred_probit$metrics$AUC)
       ),
       col = c("steelblue", "firebrick"), lwd = 2, bty = "n", cex = 0.85)

# Panel 2 — Fitted probability distribution by disease status
boxplot(
  pred_logit$prob[y_test == 0],  pred_logit$prob[y_test == 1],
  pred_probit$prob[y_test == 0], pred_probit$prob[y_test == 1],
  names   = c("Logit\nControl", "Logit\nCase",
              "Probit\nControl", "Probit\nCase"),
  col     = c("lightblue", "lightcoral", "lightyellow", "lightsalmon"),
  main    = "Fitted P(Disease) by Status",
  ylab    = "P(y = 1)",
  outline = FALSE
)
abline(h = prev_obs, lty = 2, col = "grey50")
mtext(sprintf("Dashed line = observed prevalence (%.0f%%)", prev_obs * 100),
      side = 1, line = 4, cex = 0.75)

# Panel 3 — Calibration plot
# A well-calibrated model lies on the y=x diagonal. The fitted slope
# of the diagonal regression equals pred_*$metrics$bias.
cal_l <- .calibration_curve(y_test, pred_logit$prob)
cal_p <- .calibration_curve(y_test, pred_probit$prob)
plot(cal_l$p, cal_l$y, type = "b", lwd = 2, pch = 19, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Calibration Plot (Test)",
     xlab = "Predicted P(y = 1)", ylab = "Observed proportion")
lines(cal_p$p, cal_p$y, type = "b", lwd = 2, pch = 17, col = "firebrick")
abline(0, 1, lty = 2, col = "grey60")    # ideal: y = x (slope 1)
legend("topleft",
       legend = c(
         sprintf("Logit  (slope=%.3f)", pred_logit$metrics$bias),
         sprintf("Probit (slope=%.3f)", pred_probit$metrics$bias)
       ),
       col = c("steelblue", "firebrick"), lwd = 2, pch = c(19, 17),
       bty = "n", cex = 0.85)

par(mfrow = c(1, 1))

cat("\nDone.\n")
