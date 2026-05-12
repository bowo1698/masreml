#' Load the bundled demo dataset
#'
#' Returns a small, deterministic genomic dataset bundled with the package,
#' intended for examples, vignettes, and tests.
#'
#' Two scales are bundled: \code{size = "large"} (default; n=200, p=100,
#' n_qtl=10) and \code{size = "small"} (n=100, p=50, n_qtl=5) that run
#' during \code{R CMD check --examples}.
#' Both have the same family structure (10 full-sib families) and the
#' same 12-field shape, so consumer code does not have to branch on size.
#'
#' The data simulates a small breeding-style population: 10 full-sib
#' families (each from one sire-dam founder pair). The founders are kept in
#' the pedigree only (no genotype, no phenotype). A within-family 80/20
#' train/test split is bundled so every test individual has its full-sibs in
#' training -- this is the regime where genomic prediction methods deliver
#' high accuracy.
#'
#' @section On the microhaplotype representation in \code{d$mh}:
#' In production, microhaplotype genotypes are produced by the
#' \pkg{maspipeline} preprocessing pipeline
#' (\code{genomic_prediction/maspipeline/}): phasing with Beagle, conversion
#' to phased haplotype alleles, LD- or window-based block discovery, then
#' per-strand block encoding. The on-disk output is one
#' tab-separated file per chromosome with header
#' \preformatted{
#' ID         hap_1_1 hap_1_1 hap_1_2 hap_1_2 hap_2_1 hap_2_1 hap_2_2 hap_2_2 ...
#' IND00001        1       1       2       2       1       1       2       2 ...
#' IND00002        1       1       1       1       2       2       1       1 ...
#' }
#' Each block contributes 4 columns: \code{hap_<b>_1 hap_<b>_1 hap_<b>_2 hap_<b>_2}
#' (the per-strand block name repeated twice, with per-SNP allele codes 1 or
#' 2 inside). Block coordinates and allele frequencies live in companion
#' CSVs (e.g.
#' \code{mh_info_ld_haploblock_G0/stats/microhaplotype_coordinates.csv}).
#'
#' The \code{d$mh} stored here is a compact in-memory equivalent for
#' demos: instead of per-SNP allele codes, each strand of each block is
#' collapsed into a single integer haplotype id via a base-3 polynomial
#' encoding of its SNP alleles. Shape becomes \eqn{n \times (2 \cdot n_{blocks})}
#' (two columns per block, not four). \code{\link{build_G_mh}()} auto-detects
#' this matrix layout and re-encodes the sparse integer ids internally; users
#' do not need to convert back to the per-SNP file representation.
#'
#' @param size character, either \code{"large"} (default) or \code{"small"}.
#'
#' @return A list with the following elements (dimensions shown for
#'   \code{size = "large"} / \code{size = "small"}):
#' \describe{
#'   \item{\code{snp}}{Integer matrix \eqn{n \times p} of biallelic SNP
#'     dosages (values \code{0/1/2}). Rownames are individual IDs
#'     \code{IND001..INDn}; colnames are \code{SNP001..SNPp}. Passes
#'     directly into \code{\link{masreml}(markers = list(snp_add = snp))}
#'     and \code{\link{build_G_snp}()}. Dimensions: 200 x 100 (large) /
#'     100 x 50 (small).}
#'   \item{\code{mh}}{Integer matrix \eqn{n \times (2 \cdot n_{blocks})} of
#'     microhaplotype allele codes. Columns alternate strand 1 / strand 2
#'     per block. The \code{attr(mh, "block_id")} attribute maps each column
#'     to its block. Consumable directly by
#'     \code{\link{build_G_mh}(mh_list = mh)} via the haplotype-matrix
#'     auto-detection path. Dimensions: 200 x 100 with 50 blocks (large) /
#'     100 x 50 with 25 blocks (small).}
#'   \item{\code{allele_freq}}{List with parallel vectors
#'     \code{haplotype}, \code{allele}, \code{freq} -- the training-style
#'     allele frequency table used by \pkg{masbayes}'s
#'     \code{construct_wah_matrix()}. Provided for cross-package
#'     interoperability; \code{build_G_mh()} does not need it.}
#'   \item{\code{pheno}}{Data frame with \code{n} rows. Columns: \code{id},
#'     \code{sex} (factor F/M, balanced 50/50),
#'     \code{y_cont_qtl_snp}, \code{y_cont_qtl_mh} (continuous traits under
#'     two QTL architectures), \code{y_bin_qtl_snp}, \code{y_bin_qtl_mh}
#'     (binary traits, threshold at median), \code{tbv_qtl_snp},
#'     \code{tbv_qtl_mh} (true breeding values).}
#'   \item{\code{pedigree}}{Data frame with 220 (large) or 120 (small) rows:
#'     10 sire founders + 10 dam founders (all NA parents) plus the
#'     offspring with their sire/dam recorded. Passes directly into
#'     \code{\link{build_A_ped}()}. Columns: \code{id}, \code{sire},
#'     \code{dam}.}
#'   \item{\code{qtl}}{List with \code{snp_idx}, \code{mh_idx},
#'     \code{effects_snp}, \code{effects_mh} (each length 10 for large, 5
#'     for small; effects drawn from \code{rnorm}, unit-normalised).}
#'   \item{\code{meta}}{List with \code{n}, \code{n_snp}, \code{n_blocks},
#'     \code{n_snp_per_block}, \code{n_qtl}, \code{n_families},
#'     \code{n_per_family}, \code{h2_target}, \code{sex_beta_snp},
#'     \code{sex_beta_mh}, \code{seed}, \code{split_seed}, \code{size}.}
#'   \item{\code{family_id}}{Character vector length \code{n}, values
#'     \code{fam_01..fam_10}.}
#'   \item{\code{train_idx}}{Integer vector -- row indices into
#'     \code{snp}/\code{mh}/\code{pheno} for the training set. Length 160
#'     (large) / 80 (small).}
#'   \item{\code{test_idx}}{Integer vector -- row indices for the test set.
#'     Length 40 (large) / 20 (small). Every test individual has its
#'     full-sibs in \code{train_idx}.}
#'   \item{\code{map_snp}}{Data frame with one row per SNP. Columns:
#'     \code{SNP} (matches \code{colnames(snp)}), \code{CHROM} (integer
#'     chromosome id, 1..5), \code{POS} (integer base-pair position).
#'     Synthetic 5-chromosome layout with 100 kb intra-chromosome spacing;
#'     deterministic, no RNG. Passes directly into GWAS / Manhattan-plot
#'     consumers (e.g. \pkg{CMplot}). Dimensions: 100 x 3 (large) /
#'     50 x 3 (small).}
#'   \item{\code{map_mh}}{Data frame with one row per microhaplotype block.
#'     Columns: \code{block_id} (matches \code{unique(attr(mh, "block_id"))}),
#'     \code{chr}, \code{start_pos}, \code{end_pos}, \code{n_snps}. Schema
#'     mirrors the \pkg{maspipeline} output
#'     \code{microhaplotype_coordinates.csv} so production pipelines can
#'     consume \code{d$map_mh} without translation. Dimensions: 50 x 5
#'     (large) / 25 x 5 (small).}
#' }
#'
#'
#' @examples
#' d <- load_data()
#' str(d, max.level = 1)
#' dim(d$snp)
#' head(d$pheno)
#'
#' # Smaller dataset for fast examples / unit tests:
#' d_small <- load_data("small")
#' dim(d_small$snp)
#'
#' @export
load_data <- function(size = c("large", "small")) {
  size  <- match.arg(size)
  fname <- if (size == "large") "demo_data.rds" else "demo_data_small.rds"
  path  <- system.file("extdata", fname, package = "masreml")
  if (!nzchar(path)) {
    stop(
      sprintf("%s not found in masreml inst/extdata. ", fname),
      "If you are developing masreml, run `Rscript tools/make_demo_data.R` ",
      "from the genomic_prediction/ project root and reinstall the package."
    )
  }
  readRDS(path)
}
