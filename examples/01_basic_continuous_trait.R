# examples/01_basic_continuous_trait.R
#
# REML-BLUP genomic prediction for a continuous trait using SNP additive markers.
#
# Demonstrates two modelling modes:
#   Mode A — raw marker matrix passed directly to masreml() (internal G-building)
#   Mode B — pre-built G matrix via build_G_snp(), with G_full for test prediction
#
# Reference: VanRaden (2008) J. Dairy Sci. 91:4414-4423
# Requires : masreml  (devtools::install_github("bowo1698/masreml"))

library(masreml)

set.seed(42)
n        <- 500   # total individuals
p_snp    <- 200   # number of SNP markers
p_causal <- 10    # number of causal SNPs

# ── Simulate SNP genotype matrix (0 / 1 / 2 allele counts) ──────────────────
cat("Simulating SNP genotype data...\n")
W <- matrix(
  rbinom(n * p_snp, size = 2, prob = 0.3),
  nrow = n, ncol = p_snp
)
rownames(W) <- paste0("ind", seq_len(n))
colnames(W) <- paste0("SNP", seq_len(p_snp))

# ── Simulate continuous phenotype ────────────────────────────────────────────
# y = additive genetic effects (p_causal SNPs) + environmental noise
cat("Simulating continuous phenotype...\n")
beta_causal <- rnorm(p_causal, mean = 0, sd = 0.3)
W_scaled    <- scale(W[, seq_len(p_causal)])
g_true      <- as.numeric(W_scaled %*% beta_causal)   # true breeding values
names(g_true) <- rownames(W)
e           <- rnorm(n, mean = 0, sd = 1)
y           <- g_true + e
names(y)    <- rownames(W)

h2_sim <- var(g_true) / var(y)
cat(sprintf(
  "  n=%d | mean=%.3f | SD=%.3f | simulated h2=%.3f\n",
  n, mean(y), sd(y), h2_sim
))

# ── Train / test split (80 / 20) ─────────────────────────────────────────────
set.seed(123)
idx_train <- sample(n, floor(0.8 * n))
W_train   <- W[idx_train, ]
y_train   <- y[idx_train]
W_test    <- W[-idx_train, ]
y_test    <- y[-idx_train]

cat(sprintf("  Training: n=%d | Test: n=%d\n\n", length(y_train), length(y_test)))

# ════════════════════════════════════════════════════════════════════════════
# MODE A — Raw markers passed directly; masreml() builds G internally
# ════════════════════════════════════════════════════════════════════════════

cat("== MODE A: Raw markers -> internal G-building ==\n")

fit_a <- masreml(
  y       = y_train,
  markers = list(snp_add = W_train),
  method  = "auto",       # AI-REML for continuous traits
  solver  = "auto",       # Cholesky when n < 10,000
  trait   = "continuous"
)

cat("\n")
summary(fit_a)

# Variance components table
vc_a <- varcomp(fit_a)
cat("\nVariance components (Mode A):\n")
print(vc_a)

# h2 is stored in the raw varcomp list, not in the varcomp() data.frame
h2_a <- fit_a$varcomp$h2["snp_add"]

# In-sample accuracy (training set)
acc_train_a <- compute_accuracy(
  gebv = fit_a$total_gebv,
  y    = y_train,
  h2   = h2_a
)
cat(sprintf(
  "\nTraining accuracy (Mode A): r=%.4f | slope=%.4f | r_MG=%.4f\n",
  acc_train_a$r, acc_train_a$slope, acc_train_a$r_MG
))

# Test-set prediction using markers_new + markers_train
# Allele frequencies are derived from markers_train only (no data leakage)
cat("Predicting test set (Mode A: markers_new mode)...\n")
pred_a <- predict(
  fit_a,
  markers_new   = list(snp_add = W_test),
  markers_train = list(snp_add = W_train)
)

eval_a <- evaluate_prediction(
  gebv = pred_a$GEBV,
  y    = y_test,
  h2   = h2_a,
  tbv  = g_true[names(pred_a$GEBV)]
)
cat("Test-set evaluation (Mode A):\n")
print(eval_a)

# ════════════════════════════════════════════════════════════════════════════
# MODE B — Pre-built G matrix via build_G_snp()
# ════════════════════════════════════════════════════════════════════════════

cat("\n\n== MODE B: Pre-built G matrix ==\n")

# Build training G; no ref_W needed when only training individuals are present
G_train <- build_G_snp(W_train)

fit_b <- masreml(
  y      = y_train,
  G      = list(snp_add = G_train),
  method = "auto",
  solver = "auto",
  trait  = "continuous"
)

cat("\n")
summary(fit_b)

vc_b <- varcomp(fit_b)
cat("\nVariance components (Mode B):\n")
print(vc_b)

h2_b <- fit_b$varcomp$h2["snp_add"]

# Build full G (train + test) using training allele frequencies (ref_W = W_train)
# ref_W prevents data leakage from test individuals into allele frequency estimates
cat("Building full G matrix (train + test) with training allele frequencies...\n")
G_full <- build_G_snp(W, ref_W = W_train)

cat("Predicting test set (Mode B: G_full mode)...\n")
pred_b <- predict(
  fit_b,
  G_full    = list(snp_add = G_full),
  train_ids = rownames(W_train),
  test_ids  = rownames(W_test)
)

eval_b <- evaluate_prediction(
  gebv = pred_b$GEBV,
  y    = y_test,
  h2   = h2_b,
  tbv  = g_true[names(pred_b$GEBV)]
)
cat("Test-set evaluation (Mode B):\n")
print(eval_b)

# ════════════════════════════════════════════════════════════════════════════
# Comparison: Mode A vs Mode B
# ════════════════════════════════════════════════════════════════════════════

cat("\n\n== Comparison: Mode A vs Mode B ==\n")
comp <- data.frame(
  Mode        = c("A (raw markers)", "B (pre-built G)"),
  h2_estimate = round(c(h2_a, h2_b), 4),
  r_test_y    = round(c(eval_a$r_test_y, eval_b$r_test_y), 4),
  r_test_g    = round(c(eval_a$r_test_g, eval_b$r_test_g), 4),
  bias        = round(c(eval_a$bias, eval_b$bias), 4),
  RMSE        = round(c(eval_a$RMSE, eval_b$RMSE), 4)
)
print(comp, row.names = FALSE)
cat(sprintf("Simulated h2: %.4f\n", h2_sim))

# ════════════════════════════════════════════════════════════════════════════
# Visualisation: GEBV vs observed phenotype (test set)
# ════════════════════════════════════════════════════════════════════════════

par(mfrow = c(1, 2))

plot(
  pred_a$GEBV, y_test,
  main = sprintf("Mode A: GEBV vs Phenotype (r=%.3f)", eval_a$r_test_y),
  xlab = "GEBV", ylab = "Phenotype",
  pch = 16, col = rgb(0.2, 0.4, 0.8, 0.5)
)
abline(lm(y_test ~ pred_a$GEBV), col = "red", lwd = 2)
abline(0, 1, col = "grey50", lty = 2)
legend("topleft", legend = c("Regression", "1:1 line"),
       col = c("red", "grey50"), lty = c(1, 2), lwd = c(2, 1),
       bty = "n", cex = 0.8)

plot(
  pred_b$GEBV, y_test,
  main = sprintf("Mode B: GEBV vs Phenotype (r=%.3f)", eval_b$r_test_y),
  xlab = "GEBV", ylab = "Phenotype",
  pch = 16, col = rgb(0.8, 0.3, 0.2, 0.5)
)
abline(lm(y_test ~ pred_b$GEBV), col = "red", lwd = 2)
abline(0, 1, col = "grey50", lty = 2)
legend("topleft", legend = c("Regression", "1:1 line"),
       col = c("red", "grey50"), lty = c(1, 2), lwd = c(2, 1),
       bty = "n", cex = 0.8)

par(mfrow = c(1, 1))

cat("\nDone.\n")
