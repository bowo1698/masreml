#' GWAS-Assisted Best Linear Unbiased Prediction (GWABLUP)
#'
#' Genomic prediction using GWAS-weighted relationship matrix.
#' Implements Meuwissen et al. (2024) GWABLUP: weights all markers
#' by their posterior probability of having non-zero effects,
#' derived from EMMAX GWAS results.
#'
#' User workflow:
#' \enumerate{
#'   \item Fit standard GBLUP: \code{fit <- masreml(y, markers)}
#'   \item Run GWAS: \code{gwas <- run_gwas(markers, y, fit)}
#'   \item Inspect GWAS results (optional): \code{summary(gwas)}
#'   \item Run GWABLUP: \code{fit_wa <- gwablup(y, markers, gwas)}
#' }
#'
#' @param y numeric named vector of phenotypes (length n)
#' @param markers list of raw marker inputs:
#'   \itemize{
#'     \item \code{snp_add}: matrix (n x m), raw genotype 0/1/2
#'     \item \code{mh_add}: list of chr data.frames
#'   }
#' @param gwas_result object of class \code{"gwas_result"} from
#'   \code{run_gwas()}. Must be run with the same markers and individuals.
#' @param X fixed effects design matrix (n x c). If NULL, intercept only
#' @param method character, REML algorithm (default "auto").
#'   Passed to \code{masreml()}
#' @param solver character, EBV solver (default "auto").
#'   Passed to \code{masreml()}
#' @param max_iter integer, maximum REML iterations (default 100)
#' @param tol numeric, convergence tolerance (default 1e-6)
#' @param n_threads integer, number of threads (default: all cores)
#' @param ref_markers list of raw marker data for training individuals only,
#'   same format as \code{markers}. If provided, allele frequencies for
#'   building G_wa are computed from \code{ref_markers} instead of
#'   \code{markers}. Use in train/test context to avoid data leakage.
#'   If NULL (default), allele frequencies computed from \code{markers}.
#' @param trait character, trait distribution: \code{"continuous"} (default)
#'   or \code{"binary"}. Passed to \code{masreml()}.
#' @param link character, link function for binary trait: \code{"logit"}
#'   (default) or \code{"probit"}. Passed to \code{masreml()}.
#'
#' @return Object of class \code{c("gwablup", "masreml")} —
#'   identical structure to \code{masreml()} output, with
#'   additional \code{$gwas} slot containing the gwas_result.
#'   All S3 methods from masreml (summary, print, plot) work directly.
#'
#' @seealso \code{\link{masreml}}, \code{\link{run_gwas}},
#'   \code{\link{compute_accuracy}}
#' 
#' @references
#' Meuwissen et al. (2024) GWABLUP: genome-wide association
#' assisted BLUP. \emph{Genet Sel Evol.} 56:17.
#'
#' @examples
#' \dontrun{
#' # ── SNP GWABLUP ───────────────────────────────────────────
#' fit    <- masreml(y, markers = list(snp_add = W))
#' gwas   <- run_gwas(markers = list(snp_add = W), y = y, masreml_fit = fit)
#' fit_wa <- gwablup(y, markers = list(snp_add = W), gwas_result = gwas)
#' summary(fit_wa)
#'
#' # ── MH GWABLUP ────────────────────────────────────────────
#' fit    <- masreml(y, markers = list(mh_add = mh_list))
#' gwas   <- run_gwas(markers = list(mh_add = mh_list), y = y, masreml_fit = fit)
#' fit_wa <- gwablup(y, markers = list(mh_add = mh_list), gwas_result = gwas)
#' summary(fit_wa)
#'
#' # ── Compare GBLUP vs GWABLUP ──────────────────────────────
#' fit_gblup   <- masreml(y, markers = list(snp_add = W))
#' gwas        <- run_gwas(list(snp_add = W), y, fit_gblup)
#' fit_gwablup <- gwablup(y, list(snp_add = W), gwas)
#' cor(fit_gblup$total_gebv, fit_gwablup$total_gebv)
#'
#' # ── Train/test workflow (no data leakage) ─────────────────
#' fit_tr  <- masreml(y_train, markers = list(snp_add = W_train))
#' gwas_tr <- run_gwas(
#'   markers     = list(snp_add = W_train),
#'   y           = y_train,
#'   masreml_fit = fit_tr,
#'   ref_markers = list(snp_add = W_train)
#' )
#' fit_wa_tr <- gwablup(
#'   y           = y_train,
#'   markers     = list(snp_add = W_train),
#'   gwas_result = gwas_tr,
#'   ref_markers = list(snp_add = W_train)
#' )
#'
#' # ── Binary trait GWABLUP ──────────────────────────────────
#' fit_tr_bin <- masreml(y_bin_train, markers = list(snp_add = W_train),
#'                       trait = "binary")
#' gwas_bin   <- run_gwas(list(snp_add = W_train), y_bin_train, fit_tr_bin,
#'                        ref_markers = list(snp_add = W_train))
#' fit_wa_bin <- gwablup(y_bin_train, list(snp_add = W_train), gwas_bin,
#'                       trait = "binary", ref_markers = list(snp_add = W_train))
#' G_full <- build_G_snp(W_all, ref_W = W_train)
#' pred   <- predict(fit_wa_tr,
#'                   G_full    = list(snp_add = G_full),
#'                   train_ids = rownames(W_train),
#'                   test_ids  = rownames(W_test))
#' }
#'
#' @export
gwablup <- function(
    y,
    markers,
    gwas_result,
    X            = NULL,
    method       = "auto",
    solver       = "auto",
    max_iter     = 100L,
    tol          = 1e-6,
    n_threads    = NULL,
    min_weight   = 1e-4,
    ref_markers  = NULL,
    trait        = "continuous",
    link         = "logit"
) {
  # ── Input validation ──────────────────────────────────────
  if (!is.numeric(y)) {
    stop("y must be a numeric vector.")
  }

  if (!inherits(gwas_result, "gwas_result")) {
    stop("gwas_result must be output from run_gwas().")
  }

  if (is.null(gwas_result$pp)) {
    stop("gwas_result$pp is NULL. Re-run run_gwas().")
  }

  has_snp <- !is.null(markers$snp_add)
  has_mh  <- !is.null(markers$mh_add)

  if (!has_snp && !has_mh) {
    stop("markers must contain snp_add or mh_add.")
  }

  # Marker type consistency check
  if (has_snp && gwas_result$marker_type != "snp") {
    stop("markers$snp_add provided but gwas_result is from MH GWAS.")
  }
  if (has_mh && gwas_result$marker_type != "mh") {
    stop("markers$mh_add provided but gwas_result is from SNP GWAS.")
  }

  ids <- names(y)

  # ── Build G_wa weighted matrix ────────────────────────────
  message("Building weighted G matrix (G_wa)...")
  g_list_wa <- .build_g_matrices_weighted(
    markers     = markers,
    gwas_result = gwas_result,
    ids         = ids,
    min_weight  = min_weight,
    ref_W       = ref_markers$snp_add,
    ref_mh      = ref_markers$mh_add
  )

  # ── Run masreml dengan G_wa ───────────────────────────────
  message("Running REML-BLUP with G_wa [GWABLUP]...")
  fit_wa <- masreml(
    y         = y,
    X         = X,
    G         = g_list_wa,
    method    = method,
    solver    = solver,
    max_iter  = max_iter,
    tol       = tol,
    trait     = trait,
    link      = link,
    n_threads = n_threads
  )

  # ── Attach gwas_result + upgrade class ───────────────────
  fit_wa$gwas  <- gwas_result
  fit_wa$call  <- match.call()
  class(fit_wa) <- c("gwablup", "masreml")

  fit_wa
}

# ============================================================
# S3 methods
# ============================================================

#' @export
print.gwablup <- function(x, ...) {
  cat("GWABLUP — GWAS-Assisted Genomic Prediction\n")
  cat(sprintf("  Marker type : %s\n", x$gwas$marker_type))
  cat(sprintf("  N markers   : %d\n", x$gwas$n_markers))
  cat(sprintf("  N PP > 0.5  : %d\n", sum(x$gwas$pp > 0.5)))
  cat(sprintf("  N PP > 0.9  : %d\n", sum(x$gwas$pp > 0.9)))
  cat("\n")
  # Delegate to masreml print for variance components + GEBV summary
  NextMethod()
}

#' @export
summary.gwablup <- function(object, ...) {
  cat("GWABLUP Summary\n")
  cat("════════════════════════════════════\n")
  cat("GWAS weights:\n")
  cat(sprintf("  Marker type : %s\n", object$gwas$marker_type))
  cat(sprintf("  N markers   : %d\n", object$gwas$n_markers))
  cat(sprintf("  Prior pi    : %.4f\n", object$gwas$pi))
  cat(sprintf("  Window      : %d\n",  object$gwas$window))
  cat(sprintf("  N PP > 0.5  : %d (%.1f%%)\n",
    sum(object$gwas$pp > 0.5),
    100 * mean(object$gwas$pp > 0.5)
  ))
  cat(sprintf("  N PP > 0.9  : %d (%.1f%%)\n",
    sum(object$gwas$pp > 0.9),
    100 * mean(object$gwas$pp > 0.9)
  ))
  cat("\nGenomic prediction:\n")
  # Delegate to masreml summary for variance components
  NextMethod()
}