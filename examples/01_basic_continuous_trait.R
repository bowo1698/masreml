# examples/01_basic_continuous_trait.R
#
# REML-BLUP genomic prediction for a continuous trait using multi-allelic
# microhaplotype (MH) markers. Uses the bundled demo dataset (load_data()):
# 10 full-sib families x 20 offspring, within-family 80/20 split, QTL
# effects placed at the allele-level W_mh columns (QTL@MH architecture).
#
# For the analogous SNP path and a direct SNP-vs-MH comparison study see
# examples/03_marker_QTL_congruency_theory.R.
#
# Reference: Da, Y. (2015) BMC Genet. 16:144 (multi-allelic W_ah coding)
# Requires : masreml  (devtools::install_github("bowo1698/masreml"))

library(masreml)

# ── Load bundled demo data ──────────────────────────────────────────────────
d <- load_data()
y <- d$pheno$y_cont_qtl_mh           # continuous trait, QTL@MH architecture
names(y) <- d$pheno$id               # masreml requires named y
X <- model.matrix(~ sex - 1, data = d$pheno)
tbv <- d$pheno$tbv_qtl_mh            # true breeding values for accuracy check
names(tbv) <- d$pheno$id

cat(sprintf("  n=%d | mean(y)=%.3f | sd(y)=%.3f\n",
            nrow(d$mh), mean(y), sd(y)))

# ── Bundled train / test split (within-family 80/20) ────────────────────────
idx_train <- d$train_idx
idx_test  <- d$test_idx
ids_train <- d$pheno$id[idx_train]
ids_test  <- d$pheno$id[idx_test]
y_train   <- y[idx_train]
y_test    <- y[idx_test]
tbv_test  <- tbv[idx_test]

cat(sprintf("  Training: n=%d | Test: n=%d (within-family split)\n\n",
            length(y_train), length(y_test)))

# ── Build the MH genomic relationship matrix (G_mh) ─────────────────────────
# build_G_mh() auto-detects the bundled haplotype matrix (n x 2*n_blocks).
# The full G_full enables leakage-safe test-set prediction via the G_full
# predict route; the train-only block G_train is what masreml() fits on.
G_full  <- build_G_mh(d$mh, ids = d$pheno$id)
G_train <- G_full[ids_train, ids_train]

# ── Fit: GBLUP with pre-built G_train ───────────────────────────────────────
fit <- masreml(
  y       = y_train,
  X       = X[idx_train, , drop = FALSE],
  G       = list(mh_add = G_train),
  method  = "auto",         # AI-REML for continuous traits
  solver  = "auto",         # Cholesky when n < 10,000
  trait   = "continuous"
)

cat("\n")
summary(fit)

# Variance components and heritability
vc <- varcomp(fit)
cat("\nVariance components:\n")
print(vc)
h2 <- fit$varcomp$h2["mh_add"]

# Training accuracy (in-sample)
acc_train <- compute_accuracy(gebv = fit$total_gebv, y = y_train, h2 = h2)
cat(sprintf("\nTraining accuracy: r=%.4f | slope=%.4f | r_MG=%.4f\n",
            acc_train$r, acc_train$slope, acc_train$r_MG))

# ── Test-set prediction (G_full route, leakage-safe) ────────────────────────
# We pass the full G alongside train_ids / test_ids; predict() pulls the
# cross-block G[test, train] internally and applies the appropriate BLUP
# formula. y_new triggers automatic test-set metrics.
pred <- predict(
  fit,
  G_full    = list(mh_add = G_full),
  train_ids = ids_train,
  test_ids  = ids_test,
  X_new     = X[idx_test, , drop = FALSE],
  y_new     = y_test
)

cat("\nTest-set evaluation (auto-computed by predict()):\n")
print(pred$metrics)

# evaluate_prediction() adds the TBV-based accuracy column (r_test_g) using
# the simulated truth. cor(GEBV, TBV) is the standard genomic-prediction
# accuracy in breeding literature -- not bounded by sqrt(h2) like
# cor(GEBV, y).
eval_full <- evaluate_prediction(
  gebv = pred$GEBV,
  y    = y_test,
  h2   = h2,
  tbv  = tbv_test[names(pred$GEBV)]
)
cat("\nEvaluation including TBV-based accuracy:\n")
print(eval_full)

# ── Visualisation: GEBV vs observed phenotype (test set) ────────────────────
plot(
  pred$GEBV, y_test,
  main = sprintf("GEBV vs Phenotype (r=%.3f, r_test_g=%.3f)",
                 eval_full$r_test_y, eval_full$r_test_g),
  xlab = "GEBV", ylab = "Phenotype",
  pch  = 16, col = rgb(0.2, 0.4, 0.8, 0.5)
)
abline(lm(y_test ~ pred$GEBV), col = "red", lwd = 2)
abline(0, 1, col = "grey50", lty = 2)
legend("topleft", legend = c("Regression", "1:1 line"),
       col = c("red", "grey50"), lty = c(1, 2), lwd = c(2, 1),
       bty = "n", cex = 0.8)

cat(sprintf("\nG_mh dimensions: %d x %d (n x n; same shape as G_snp)\n",
            nrow(G_full), ncol(G_full)))
cat("\nDone.\n")
