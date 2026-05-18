# examples/05_forward_prediction.R
#
# Forward genomic prediction with GBLUP. Uses the multi-generation extension
# of the bundled demo dataset (load_data("small")$multigen):
#
#   Scenario A: train at gen-1,         predict gen-2
#   Scenario B: train at gen-1 + gen-2, predict gen-3
#
# Architecture fixed at QTL@MH. Two marker bases (SNP, MH) -> 4 fits.
# G is built leakage-safe (test G[test, train] sliced from a full G with
# training-set centring via ref_W / ref_mh). Reports r(GEBV, TBV) with
# bootstrap SE, rendered as a ggplot bar chart.
#
# Requirements: masreml, ggplot2, dplyr
# Usage: source("masreml/examples/05_forward_prediction.R")

suppressPackageStartupMessages({
  library(masreml); library(ggplot2); library(dplyr)
})

# Resolve the directory this script lives in, so plots save next to the
# script regardless of the caller's working directory. Works for both
# `Rscript path/to/file.R` and interactive `source("path/to/file.R")`.
script_dir <- local({
  fa <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) > 0) {
    return(dirname(normalizePath(sub("--file=", "", fa[1]))))
  }
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) dirname(normalizePath(ofile)) else getwd()
})

d <- load_data("small")$multigen
N_BOOT <- 200L

# Build training + prediction package for one (scenario, marker) cell.
# Architecture = QTL@MH; train/test ids span the relevant cohorts. Only the
# marker-specific full geno matrix is materialised — the unused alternative
# is never rbind()-ed.
build_pack <- function(scenario, marker) {
  if (scenario == "A") {
    train_pheno <- d$gen1$pheno
    test_cohort <- d$gen2_mh
    geno_list   <- list(snp = list(d$gen1$snp, d$gen2_mh$snp),
                        mh  = list(d$gen1$mh,  d$gen2_mh$mh))
  } else {
    train_pheno <- rbind(
      d$gen1$pheno[,    c("id", "sex", "y_cont_qtl_mh")],
      d$gen2_mh$pheno[, c("id", "sex", "y_cont_qtl_mh")])
    test_cohort <- d$gen3_mh
    geno_list <- list(
      snp = list(d$gen1$snp, d$gen2_mh$snp, d$gen3_mh$snp),
      mh  = list(d$gen1$mh,  d$gen2_mh$mh,  d$gen3_mh$mh))
  }
  ids_tr <- train_pheno$id
  ids_te <- test_cohort$pheno$id
  ids_all <- c(ids_tr, ids_te)

  y_tr <- setNames(train_pheno$y_cont_qtl_mh,        ids_tr)
  y_te <- setNames(test_cohort$pheno$y_cont_qtl_mh,  ids_te)
  X_tr <- model.matrix(~ train_pheno$sex - 1)
  X_te <- model.matrix(~ test_cohort$pheno$sex - 1)
  colnames(X_tr) <- colnames(X_te) <- c("F", "M")
  rownames(X_tr) <- ids_tr; rownames(X_te) <- ids_te

  if (marker == "snp") {
    snp_full <- do.call(rbind, geno_list$snp); rownames(snp_full) <- ids_all
    G_full <- build_G_snp(snp_full, ref_W = snp_full[ids_tr, ])
    G_key  <- "snp_add"
  } else {
    mh_full  <- do.call(rbind, geno_list$mh);  rownames(mh_full)  <- ids_all
    G_full <- build_G_mh(mh_full, ref_mh = mh_full[ids_tr, ], ids = ids_all)
    G_key  <- "mh_add"
  }

  list(y_tr = y_tr, X_tr = X_tr, X_te = X_te,
       ids_tr = ids_tr, ids_te = ids_te,
       G_full = G_full, G_key = G_key,
       tbv_te = setNames(test_cohort$pheno$tbv_qtl_mh_true, ids_te),
       y_te = y_te)
}

fit_predict <- function(p) {
  fit <- masreml(y = p$y_tr, X = p$X_tr,
                 G = setNames(list(p$G_full[p$ids_tr, p$ids_tr]), p$G_key),
                 method = "auto", solver = "auto", trait = "continuous")
  predict(fit,
          G_full    = setNames(list(p$G_full), p$G_key),
          train_ids = p$ids_tr, test_ids = p$ids_te,
          X_new     = p$X_te,    y_new    = p$y_te)
}

boot_se <- function(gebv, tbv, n_boot, seed) {
  set.seed(seed)
  rs <- replicate(n_boot, {
    idx <- sample(seq_along(tbv), replace = TRUE)
    suppressWarnings(cor(gebv[idx], tbv[idx]))
  })
  sd(rs, na.rm = TRUE)
}

combos <- expand.grid(scenario = c("A", "B"), marker = c("snp", "mh"),
                      stringsAsFactors = FALSE)
sink_path <- tempfile()
res <- combos
res$r_test_g <- NA_real_; res$se <- NA_real_
for (i in seq_len(nrow(combos))) {
  sink(sink_path)
  pack <- build_pack(combos$scenario[i], combos$marker[i])
  pred <- fit_predict(pack)
  sink()
  gebv_ord <- pred$GEBV
  tbv_ord  <- pack$tbv_te[names(gebv_ord)]
  res$r_test_g[i] <- cor(gebv_ord, tbv_ord)
  res$se[i]       <- boot_se(gebv_ord, tbv_ord, N_BOOT, seed = 1000L + i)
}
unlink(sink_path)

res_print <- res %>%
  mutate(scenario = ifelse(scenario == "A",
                           "A (gen1->gen2)", "B (gen1+2->gen3)")) %>%
  arrange(scenario, marker)
cat("\n=== Forward prediction r_test_g (GBLUP, QTL@MH architecture) ===\n\n")
print(res_print, row.names = FALSE)

# ── Bar plot with SE ──────────────────────────────────────────────────────
plot_df <- res %>%
  mutate(scenario = factor(scenario, levels = c("A", "B"),
                           labels = c("A: gen1 -> gen2",
                                      "B: gen1+gen2 -> gen3")),
         marker = toupper(marker))

# GBLUP forward prediction under the QTL@MH architecture: SNP vs MH markers,
# faceted by training scenario. Error bars are bootstrap SE (200 resamples on
# the test set, single seed).
p <- ggplot(plot_df, aes(marker, r_test_g, fill = marker)) +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = r_test_g - se, ymax = r_test_g + se),
                width = 0.2) +
  facet_wrap(~ scenario) +
  scale_fill_manual(values = c(SNP = "#4C72B0", MH = "#DD8452"),
                    guide = "none") +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "Marker", y = "r(GEBV, TBV)  ± bootstrap SE") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.x = element_blank())

print(p)
ggsave(file.path(script_dir, "05_forward_prediction.png"),
       p, width = 5, height = 4, dpi = 300, bg = "white")

invisible(list(results = res, plot = p))
