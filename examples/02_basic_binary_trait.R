# examples/02_basic_binary_trait.R
#
# REML-BLUP genomic prediction for a binary (0/1) trait using multi-allelic
# microhaplotype (MH) markers. The binary GLMM is solved via a single-step
# Laplace approximation on the liability scale.
#
# Uses the bundled demo dataset (load_data()): 10 full-sib families x 20
# offspring, within-family 80/20 split, QTL effects at the allele level
# (QTL@MH architecture). For SNP / SNP-vs-MH comparison see
# examples/03_marker_QTL_congruency_theory.R.
#
# Demonstrates:
#   1. Binary GBLUP with the default logit link on G_mh
#   2. predict() API: auto-compute metrics with `y_new` argument
#   3. $prob field (predicted probabilities P(y=1))
#   4. In-sample mode: predict(fit) returns training_metrics
#   5. Calibration slope (bias) on the observed scale: 1.0 = well-calibrated
#   6. Two-panel visualisation: ROC curve + calibration plot
#
# Reference: Dempster & Falconer (1950) Ann. Hum. Genet. 31:195-203
#            Da (2015) BMC Genet. 16:144 (multi-allelic W_ah coding)
#            Steyerberg (2010) Clinical Prediction Models -- calibration slope
# Requires : masreml, pROC

library(masreml)

# ── Load bundled demo data ──────────────────────────────────────────────────
d <- load_data()
y_binary <- as.integer(d$pheno$y_bin_qtl_mh)
names(y_binary) <- d$pheno$id
X <- model.matrix(~ sex - 1, data = d$pheno)
n <- nrow(d$mh)

cat(sprintf("  Prevalence: %.1f%% (cases=%d, controls=%d) | n=%d\n",
            mean(y_binary) * 100, sum(y_binary), n - sum(y_binary), n))

# ── Bundled train / test split ──────────────────────────────────────────────
idx_train <- d$train_idx
idx_test  <- d$test_idx
ids_train <- d$pheno$id[idx_train]
ids_test  <- d$pheno$id[idx_test]
y_train   <- y_binary[idx_train]
y_test    <- y_binary[idx_test]

cat(sprintf("  Training: n=%d (cases=%d) | Test: n=%d (cases=%d)\n\n",
            length(y_train), sum(y_train), length(y_test), sum(y_test)))

# ── Build the MH genomic relationship matrix (G_mh) ─────────────────────────
G_full  <- build_G_mh(d$mh, ids = d$pheno$id)
G_train <- G_full[ids_train, ids_train]

# ── Fit: binary GBLUP with logit link on G_train ────────────────────────────
fit <- masreml(
  y       = y_train,
  X       = X[idx_train, , drop = FALSE],
  G       = list(mh_add = G_train),
  method  = "auto",      # HE-regression for binary traits
  trait   = "binary",
  link    = "logit",
  solver  = "auto"
)

cat("\n")
summary(fit)

# ── In-sample shortcut: predict(fit) ────────────────────────────────────────
in_pred <- predict(fit)
cat(sprintf("\nIn-sample (training): AUC=%.3f | bias=%.3f (calibration slope)\n",
            in_pred$metrics$AUC, in_pred$metrics$bias))

# ── Test-set prediction (G_full route): y_new triggers auto-metrics ─────────
pred <- predict(
  fit,
  G_full    = list(mh_add = G_full),
  train_ids = ids_train,
  test_ids  = ids_test,
  X_new     = X[idx_test, , drop = FALSE],
  y_new     = y_test
)
print(pred)

cat("\nInterpretation of `bias` (calibration slope):\n")
cat("  = 1.0 : predicted probabilities well-calibrated\n")
cat("  < 1.0 : over-dispersion -- predictions too extreme (over-confident)\n")
cat("  > 1.0 : under-dispersion -- predictions too compressed toward base rate\n")

# ════════════════════════════════════════════════════════════════════════════
# Visualisation -- two panels (ROC + calibration)
# ════════════════════════════════════════════════════════════════════════════

# Helper: quantile-binned calibration curve. n_bins=5 chosen because n_test=40
# leaves only ~8 obs per bin at 5 bins; 10 bins would give ~4 obs per bin and
# a noisy curve. This is a plotting helper only -- the calibration slope
# itself is reported by predict()$metrics$bias on the observed scale.
.calibration_curve <- function(y, prob, n_bins = 5) {
  qs    <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1),
                    na.rm = TRUE)
  qs[1] <- qs[1] - 1e-9
  bin   <- cut(prob, breaks = qs, include.lowest = TRUE, labels = FALSE)
  agg   <- data.frame(bin = bin, y = y, p = prob)
  aggregate(cbind(y = y, p = p) ~ bin, data = agg, FUN = mean)
}

par(mfrow = c(1, 2))

# Panel 1 -- ROC curve
pROC::plot.roc(pROC::roc(y_test, pred$prob, quiet = TRUE),
               main = sprintf("ROC (Test, AUC=%.3f)", pred$metrics$AUC),
               col  = "steelblue", lwd = 2)

# Panel 2 -- Calibration plot: predicted probability vs observed proportion
cal <- .calibration_curve(y_test, pred$prob)
plot(cal$p, cal$y, type = "b", lwd = 2, pch = 19, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     main = sprintf("Calibration (slope=%.3f)", pred$metrics$bias),
     xlab = "Predicted P(y = 1)", ylab = "Observed proportion")
abline(0, 1, lty = 2, col = "grey60")

par(mfrow = c(1, 1))

cat(sprintf("\nG_mh dimensions: %d x %d (n x n; same shape as G_snp)\n",
            nrow(G_full), ncol(G_full)))
cat("\nDone.\n")
