# 04_gwas.R — EMMAX GWAS + GWABLUP showcase using masreml.
#
# Demonstrates the GWAS surface in masreml:
#   * run_gwas() runs EMMAX (Kang et al. 2010) on top of a fitted masreml
#     object, returning per-marker (or per-block) likelihood ratios,
#     effect estimates, p-values, and a smoothed posterior probability
#     `pp` of non-zero effect.
#   * gwablup() reuses that `pp` as marker weights to build a
#     GWAS-assisted G matrix (Meuwissen et al. 2024) and refits the
#     mixed model — typically lifting heritability captured by the
#     genomic component when real QTL are present.
#
# Run with:
#   Rscript examples/04_gwas.R
#
# Uses load_data() (default size = "large": n = 200, 100 SNPs across 50
# MH blocks, 10 QTL per architecture). The demo ships d$map_snp and
# d$map_mh (5-chromosome synthetic layout with 100 kb intra-chr spacing;
# d$map_mh follows the maspipeline microhaplotype_coordinates.csv schema),
# so this example uses them directly for the Manhattan plots.

suppressPackageStartupMessages(library(masreml))

d        <- load_data()
ids      <- d$pheno$id
map_snp  <- d$map_snp
map_mh   <- d$map_mh
n_snp    <- nrow(map_snp)
n_blocks <- nrow(map_mh)
# Single-point position per block (midpoint of start_pos / end_pos) for
# CMplot consumption; map_mh keeps the start/end interval for downstream
# pipelines that need it.
mh_pos   <- as.integer((map_mh$start_pos + map_mh$end_pos) / 2)

# ----- (A) SNP path ---------------------------------------------------------
cat("\n== (A) SNP GWAS + GWABLUP ==\n")
y_snp <- setNames(d$pheno$y_cont_qtl_snp, ids)
W_snp <- d$snp
storage.mode(W_snp) <- "double"
rownames(W_snp)     <- ids

fit_snp_g  <- masreml(y       = y_snp,
                      markers = list(snp_add = W_snp),
                      method  = "auto",
                      trait   = "continuous")

gwas_snp   <- run_gwas(markers     = list(snp_add = W_snp),
                       y           = y_snp,
                       masreml_fit = fit_snp_g,
                       ref_markers = list(snp_add = W_snp))

fit_snp_wa <- gwablup(y           = y_snp,
                      markers     = list(snp_add = W_snp),
                      gwas_result = gwas_snp,
                      ref_markers = list(snp_add = W_snp),
                      trait       = "continuous")

cat("Top 10 SNPs by posterior probability (pp):\n")
top10_snp_idx <- order(-gwas_snp$pp)[seq_len(min(10L, n_snp))]
print(data.frame(SNP   = map_snp$SNP[top10_snp_idx],
                 CHROM = map_snp$CHROM[top10_snp_idx],
                 POS   = map_snp$POS[top10_snp_idx],
                 pval  = signif(gwas_snp$pval[top10_snp_idx], 3),
                 pp    = round(gwas_snp$pp[top10_snp_idx], 3)),
      row.names = FALSE)

# ----- (B) Microhaplotype path ---------------------------------------------
cat("\n== (B) MH GWAS + GWABLUP ==\n")
y_mh <- setNames(d$pheno$y_cont_qtl_mh, ids)
mh   <- d$mh
rownames(mh) <- ids

fit_mh_g  <- masreml(y       = y_mh,
                     markers = list(mh_add = mh),
                     method  = "auto",
                     trait   = "continuous")

gwas_mh   <- run_gwas(markers     = list(mh_add = mh),
                      y           = y_mh,
                      masreml_fit = fit_mh_g,
                      ref_markers = list(mh_add = mh))

fit_mh_wa <- gwablup(y           = y_mh,
                     markers     = list(mh_add = mh),
                     gwas_result = gwas_mh,
                     ref_markers = list(mh_add = mh),
                     trait       = "continuous")

cat("Top 10 MH blocks by posterior probability (pp):\n")
top10_mh_idx <- order(-gwas_mh$pp)[seq_len(min(10L, n_blocks))]
print(data.frame(Block = map_mh$block_id[top10_mh_idx],
                 CHR   = map_mh$chr[top10_mh_idx],
                 POS   = mh_pos[top10_mh_idx],
                 pval  = signif(gwas_mh$pval[top10_mh_idx], 3),
                 pp    = round(gwas_mh$pp[top10_mh_idx], 3)),
      row.names = FALSE)

# ----- (C) QTL recovery sanity ---------------------------------------------
# masreml's run_gwas(mh_add = ...) returns one pp per block, so we map
# allele-level QTL indices to block ids via d$allele_freq$haplotype
# (the canonical W_ah column -> block lookup that ships with the demo).
cat("\n== (C) QTL recovery sanity ==\n")
cat(sprintf(
  "  SNP path: median pp at QTL = %.3f  vs  non-QTL = %.3f\n",
  median(gwas_snp$pp[d$qtl$snp_idx]),
  median(gwas_snp$pp[-d$qtl$snp_idx])
))

qtl_block_names <- unique(d$allele_freq$haplotype[d$qtl$mh_idx])
qtl_blocks      <- as.integer(sub("block_", "", qtl_block_names))
cat(sprintf(
  "  MH  path: median pp at QTL = %.3f  vs  non-QTL = %.3f\n",
  median(gwas_mh$pp[qtl_blocks]),
  median(gwas_mh$pp[-qtl_blocks])
))

cat(sprintf(
  "  SNP h2: GBLUP = %.3f | GWABLUP = %.3f\n",
  as.numeric(fit_snp_g$varcomp$h2["snp_add"]),
  as.numeric(fit_snp_wa$varcomp$h2["snp_add"])
))
cat(sprintf(
  "  MH  h2: GBLUP = %.3f | GWABLUP = %.3f\n",
  as.numeric(fit_mh_g$varcomp$h2["mh_add"]),
  as.numeric(fit_mh_wa$varcomp$h2["mh_add"])
))

# ----- (D) Manhattan plots via CMplot --------------------------------------
# Two metrics per path:
#   pval — raw EMMAX evidence; plotted as -log10(p) with a Bonferroni
#          line at 0.05 / n_tests.
#   pp   — smoothed posterior probability used as gwablup() weight;
#          plotted as -log10(1 - pp) with a 0.5 reference line.
#          Cap at 1 - 1e-3 to keep
#          -log10(0) = Inf out of the plot.

if (!requireNamespace("CMplot", quietly = TRUE)) {
  stop("Section (D) requires the CMplot package. Install with ",
       "install.packages(\"CMplot\").")
}

out_dir  <- getwd()
pp_cap   <- 1 - 1e-3
pval_cap <- 1e-12

snp_pval_df <- data.frame(SNP  = map_snp$SNP,
                          Chr  = map_snp$CHROM,
                          Pos  = map_snp$POS,
                          pval = pmax(gwas_snp$pval, pval_cap),
                          check.names = FALSE)
snp_pp_df   <- data.frame(SNP  = map_snp$SNP,
                          Chr  = map_snp$CHROM,
                          Pos  = map_snp$POS,
                          `1-pp` = 1 - pmin(gwas_snp$pp, pp_cap),
                          check.names = FALSE)
mh_pval_df  <- data.frame(Block = map_mh$block_id,
                          Chr   = map_mh$chr,
                          Pos   = mh_pos,
                          pval  = pmax(gwas_mh$pval, pval_cap),
                          check.names = FALSE)
mh_pp_df    <- data.frame(Block = map_mh$block_id,
                          Chr   = map_mh$chr,
                          Pos   = mh_pos,
                          `1-pp` = 1 - pmin(gwas_mh$pp, pp_cap),
                          check.names = FALSE)

thr_snp_p <- 0.05 / n_snp
thr_mh_p  <- 0.05 / n_blocks

nonempty <- function(x) if (length(x) > 0L) as.character(x) else NULL
snp_pval_hl <- nonempty(map_snp$SNP[gwas_snp$pval < thr_snp_p])
snp_pp_hl   <- nonempty(map_snp$SNP[gwas_snp$pp   > 0.5])
mh_pval_hl  <- nonempty(map_mh$block_id[gwas_mh$pval  < thr_mh_p])
mh_pp_hl    <- nonempty(map_mh$block_id[gwas_mh$pp    > 0.5])

cmplot_args <- list(
  type          = "h",
  plot.type     = "m",
  LOG10         = TRUE,
  threshold.lty = 2,
  threshold.lwd = 1,
  threshold.col = "red",
  amplify       = FALSE,
  highlight.col = NULL,
  file          = "jpg",
  dpi           = 150,
  file.output   = TRUE,
  verbose       = FALSE
)

do.call(CMplot::CMplot, c(list(snp_pval_df,
  ylab      = expression(-log[10](italic(p))),
  threshold = thr_snp_p,
  file.name = "snp_pval",
  highlight = snp_pval_hl, highlight.text = snp_pval_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(snp_pp_df,
  ylab      = expression(-log[10](1 - italic(pp))),
  threshold = 0.5,
  file.name = "snp_pp",
  highlight = snp_pp_hl, highlight.text = snp_pp_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(mh_pval_df,
  ylab      = expression(-log[10](italic(p))),
  threshold = thr_mh_p,
  file.name = "mh_pval",
  highlight = mh_pval_hl, highlight.text = mh_pval_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(mh_pp_df,
  ylab      = expression(-log[10](1 - italic(pp))),
  threshold = 0.5,
  file.name = "mh_pp",
  highlight = mh_pp_hl, highlight.text = mh_pp_hl), cmplot_args))

cat("  Rect_Manhtn.{snp_pval,snp_pp,mh_pval,mh_pp}.jpg\n")

cat("\n04_gwas.R completed.\n")
