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
#   4. Predicting fitted probabilities P(y=1) for the test set
#   5. Evaluating AUC and comparing logit vs probit links
#
# Reference: Dempster & Falconer (1950) Ann. Hum. Genet. 31:195-203
#            Da (2015) BMC Genet. 16:144
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

threshold   <- quantile(liability, probs = 1 - prevalence)
y_binary    <- as.integer(liability > threshold)
names(y_binary) <- rownames(W)

n_cases   <- sum(y_binary)
n_controls <- n - n_cases
prev_obs   <- mean(y_binary)
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
W_train   <- W[idx_train, ]
y_train   <- y_binary[idx_train]
W_test    <- W[-idx_train, ]
y_test    <- y_binary[-idx_train]

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

cat("\n")
summary(fit_logit)

# Binary-specific results stored in fit$binary
bin_logit <- fit_logit$binary
h2_a      <- fit_logit$varcomp$h2["snp_add"]

cat("\n-- Binary results (logit link, training set) --\n")
cat(sprintf("  h2 (liability scale) : %.4f\n", bin_logit$h2_liability))
cat(sprintf("  h2 (observed scale)  : %.4f\n", bin_logit$h2_observed))
cat(sprintf("  AUC (training)       : %.4f\n", bin_logit$auc))
cat(sprintf("  Prevalence           : %.4f\n", bin_logit$prevalence))
cat(sprintf("  Link function        : %s\n",   bin_logit$link))

# Test-set prediction — predict() inherits link from fit$binary$link
cat("\nPredicting test set (logit model)...\n")
pred_logit <- predict(
  fit_logit,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train)
)

# pred_logit$fitted contains P(y=1) on the probability scale
eval_logit <- evaluate_prediction(
  gebv        = pred_logit$total_gebv,
  y           = y_test,
  h2          = h2_a,
  fitted_prob = pred_logit$fitted
)
cat("Test-set evaluation (logit):\n")
print(eval_logit)

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
  solver  = "auto"
)

cat("\n")
summary(fit_probit)

bin_probit <- fit_probit$binary
h2_b       <- fit_probit$varcomp$h2["snp_add"]

cat("\n-- Binary results (probit link, training set) --\n")
cat(sprintf("  h2 (liability scale) : %.4f\n", bin_probit$h2_liability))
cat(sprintf("  h2 (observed scale)  : %.4f\n", bin_probit$h2_observed))
cat(sprintf("  AUC (training)       : %.4f\n", bin_probit$auc))

cat("\nPredicting test set (probit model)...\n")
pred_probit <- predict(
  fit_probit,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train)
)

eval_probit <- evaluate_prediction(
  gebv        = pred_probit$total_gebv,
  y           = y_test,
  h2          = h2_b,
  fitted_prob = pred_probit$fitted
)
cat("Test-set evaluation (probit):\n")
print(eval_probit)

# ════════════════════════════════════════════════════════════════════════════
# Comparison: Logit vs Probit
# ════════════════════════════════════════════════════════════════════════════

cat("\n\n== Comparison: Logit vs Probit ==\n")
comp <- data.frame(
  Link           = c("logit", "probit"),
  h2_liability   = round(c(bin_logit$h2_liability,  bin_probit$h2_liability),  4),
  h2_observed    = round(c(bin_logit$h2_observed,   bin_probit$h2_observed),   4),
  AUC_train      = round(c(bin_logit$auc,           bin_probit$auc),           4),
  AUC_test       = round(c(eval_logit$AUC,          eval_probit$AUC),          4),
  bias_test      = round(c(eval_logit$bias,         eval_probit$bias),         4),
  RMSE_test      = round(c(eval_logit$RMSE,         eval_probit$RMSE),         4)
)
print(comp, row.names = FALSE)
cat(sprintf("Simulated h2 (liability): %.4f\n", h2_liability_sim))

# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════

# Helper: empirical ROC curve (base R, no external dependency)
.roc_curve <- function(y, prob) {
  thresholds <- sort(unique(prob), decreasing = TRUE)
  tpr <- fpr <- numeric(length(thresholds) + 2)
  tpr[1] <- fpr[1] <- 0
  for (i in seq_along(thresholds)) {
    pred_pos  <- prob >= thresholds[i]
    tpr[i + 1] <- sum(pred_pos & y == 1) / sum(y == 1)
    fpr[i + 1] <- sum(pred_pos & y == 0) / sum(y == 0)
  }
  tpr[length(tpr)] <- fpr[length(fpr)] <- 1
  list(fpr = fpr, tpr = tpr)
}

par(mfrow = c(1, 2))

# Panel 1 — ROC curves
roc_l <- .roc_curve(y_test, pred_logit$fitted)
roc_p <- .roc_curve(y_test, pred_probit$fitted)

plot(roc_l$fpr, roc_l$tpr, type = "l", lwd = 2, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "ROC Curves (Test Set)",
     xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(roc_p$fpr, roc_p$tpr, lwd = 2, col = "firebrick")
abline(0, 1, lty = 2, col = "grey60")
legend("bottomright",
       legend = c(
         sprintf("Logit  (AUC=%.3f)", eval_logit$AUC),
         sprintf("Probit (AUC=%.3f)", eval_probit$AUC)
       ),
       col = c("steelblue", "firebrick"), lwd = 2, bty = "n", cex = 0.85)

# Panel 2 — Fitted probability distribution by disease status
prob_mat <- data.frame(
  logit  = pred_logit$fitted,
  probit = pred_probit$fitted,
  status = ifelse(y_test == 1, "Case (1)", "Control (0)")
)

boxplot(
  pred_logit$fitted[y_test == 0],  pred_logit$fitted[y_test == 1],
  pred_probit$fitted[y_test == 0], pred_probit$fitted[y_test == 1],
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

par(mfrow = c(1, 1))

cat("\nDone.\n")
