#' Run GWAS for SNP or Microhaplotype Markers
#'
#' Performs genome-wide association study (GWAS) for SNP or microhaplotype
#' (MH) markers. Uses EMMAX (Efficient Mixed-Model Association eXpedited)
#' to control for population structure and relatedness, preventing spurious
#' associations. Results can be inspected for QTL detection or passed
#' directly to \code{gwablup()} for GWAS-assisted genomic prediction.
#'
#' @param markers list of raw marker inputs. Provide one of:
#'   \itemize{
#'     \item \code{snp_add}: numeric matrix (n x m), genotype coded 0/1/2
#'     \item \code{mh_add}: list of data.frames, one per chromosome,
#'       with paired haplotype columns per block
#'   }
#' @param y numeric named vector of phenotypes (length n). Names must
#'   match row names of marker matrix.
#' @param masreml_fit fitted \code{masreml} object from \code{masreml()}.
#'   Used to account for population structure in the GWAS model.
#'   Must be fitted with the same markers and individuals.
#' @param X fixed effects design matrix (n x c). If NULL, intercept only.
#' @param pi numeric, prior probability that a marker has a non-zero
#'   effect (default 0.001). Lower values make posterior probabilities
#'   more conservative.
#' @param window integer, number of adjacent markers used in moving
#'   average smoothing of likelihood ratios (default 5). Larger values
#'   produce smoother posterior probabilities.
#' @param ref_markers list of raw marker data for training individuals only,
#'   in the same format as \code{markers}. If provided, allele frequencies
#'   for building G_u (used in EMMAX) are computed from \code{ref_markers}
#'   instead of \code{markers}. Use this when running GWAS in a train/test
#'   context to avoid data leakage from test individuals into the
#'   population structure correction matrix.
#'   Supported elements: \code{snp_add} (matrix) or \code{mh_add} (list).
#'   If NULL (default), allele frequencies are computed from \code{markers}.
#'
#' @return Object of class \code{"gwas_result"} with elements:
#'   \itemize{
#'     \item \code{lr}: log-likelihood ratio per marker/block
#'     \item \code{beta}: effect estimate per marker/block
#'     \item \code{se}: standard error per marker/block
#'     \item \code{pval}: p-value per marker/block
#'     \item \code{smoothed_lr}: smoothed LR after moving average
#'     \item \code{pp}: posterior probability of non-zero effect per
#'       marker/block — use this as input to \code{gwablup()}
#'     \item \code{marker_type}: \code{"snp"} or \code{"mh"}
#'     \item \code{n_markers}: number of markers or MH blocks
#'     \item \code{pi}: prior probability used
#'     \item \code{window}: smoothing window size used
#'   }
#'
#' @seealso \code{\link{masreml}}, \code{\link{gwablup}}
#'
#' @references
#' Kang et al. (2010) Variance component model to account for
#' sample structure in GWAS. \emph{Nat Genet.} 42:348-354.
#'
#' Meuwissen et al. (2024) GWABLUP: genome-wide association
#' assisted BLUP. \emph{Genet Sel Evol.} 56:17.
#'
#' @examples
#' \dontrun{
#' # ── Standard workflow (all individuals) ───────────────────
#' # Step 1: fit standard GBLUP
#' fit <- masreml(y, markers = list(snp_add = W))
#'
#' # Step 2: run GWAS — inspect QTL signals
#' gwas <- run_gwas(
#'   markers     = list(snp_add = W),
#'   y           = y,
#'   masreml_fit = fit
#' )
#' summary(gwas)
#'
#' # Step 3: run GWABLUP using GWAS weights
#' fit_wa <- gwablup(y, markers = list(snp_add = W), gwas_result = gwas)
#' summary(fit_wa)
#'
#' # ── Train/test workflow (no data leakage) ─────────────────
#' # Step 1: fit GBLUP on training only
#' fit_tr <- masreml(y_train, markers = list(snp_add = W_train))
#'
#' # Step 2: run GWAS on training — ref_markers = training markers
#' gwas_tr <- run_gwas(
#'   markers     = list(snp_add = W_train),
#'   y           = y_train,
#'   masreml_fit = fit_tr,
#'   ref_markers = list(snp_add = W_train)
#' )
#'
#' # Step 3: GWABLUP on training with training-based freq
#' fit_wa_tr <- gwablup(
#'   y           = y_train,
#'   markers     = list(snp_add = W_train),
#'   gwas_result = gwas_tr,
#'   ref_markers = list(snp_add = W_train)
#' )
#'
#' # Step 4: predict test set
#' G_full <- build_G_snp(W_all, ref_W = W_train)
#' pred   <- predict(fit_wa_tr,
#'                   G_full    = list(snp_add = G_full),
#'                   train_ids = rownames(W_train),
#'                   test_ids  = rownames(W_test))
#' }
#'
#' @export
run_gwas <- function(
    markers,
    y,
    masreml_fit,
    X            = NULL,
    pi           = 0.001,
    window       = 5L,
    ref_markers  = NULL
) {
  # ── Input validation ──────────────────────────────────────
  if (!inherits(masreml_fit, "masreml")) {
    stop("masreml_fit must be a fitted masreml object.")
  }

  if (!is.numeric(y)) {
    stop("y must be a numeric vector.")
  }

  if (pi <= 0 || pi >= 1) {
    stop("pi must be in (0, 1).")
  }

  window <- as.integer(window)
  if (window < 1L) {
    stop("window must be >= 1.")
  }

  # ── Extract variance components from masreml_fit ──────────
  sigma2   <- masreml_fit$varcomp$sigma2
  n_random <- length(sigma2) - 1L
  sigma2_g <- sigma2[1]           # first random component
  sigma2_e <- sigma2[n_random + 1]  # residual

  ids <- names(y)
  n   <- length(y)

  # Prepare X — intercept only if NULL
  X <- .prepare_X(X, n)

  # ── Determine marker type ─────────────────────────────────
  has_snp <- !is.null(markers$snp_add)
  has_mh  <- !is.null(markers$mh_add)

  if (!has_snp && !has_mh) {
    stop("markers must contain snp_add or mh_add.")
  }
  if (has_snp && has_mh) {
    stop("Provide either snp_add or mh_add, not both, for GWAS.")
  }

  marker_type <- if (has_snp) "snp" else "mh"

  # ── Build G_u unweighted — reuse from masreml_fit ─────────
  g_u <- .extract_g_u(masreml_fit, markers, ids, ref_markers = ref_markers)

  # ── Run EMMAX di Rust ──────────────────────────────────────
  message(sprintf(
    "Running EMMAX GWAS [marker_type=%s, n=%d, pi=%.4f, window=%d]...",
    marker_type, n, pi, window
  ))

  if (marker_type == "snp") {
    w <- .validate_snp_matrix(markers$snp_add, "snp_add", ids)
    raw_result <- r_run_emmax_snp(
      w        = w,
      y        = as.double(y),
      x        = X,
      sigma2_g = sigma2_g,
      sigma2_e = sigma2_e,
      g_u      = g_u
    )
    n_markers <- ncol(w)

  } else {
    # MH: parse to get W_αh flat matrix + block_sizes
    parsed_mh <- .prepare_mh_for_gwas(markers$mh_add, ids)
    raw_result <- r_run_emmax_mh(
        hap1      = parsed_mh$hap1,
        hap2      = parsed_mh$hap2,
        n_alleles = parsed_mh$n_alleles,
        y         = as.double(y),
        x         = X,
        sigma2_g  = sigma2_g,
        sigma2_e  = sigma2_e,
        g_u       = g_u
    )
    n_markers <- length(parsed_mh$block_sizes)
  }

  # ── Smooth LR + compute PP in Rust ────────────────────────
  smooth_result <- r_smooth_and_pp(
    lr     = raw_result$lr,
    window = window,
    pi     = pi
  )

  # ── Assemble output ────────────────────────────────────────
  structure(
    list(
      lr          = raw_result$lr,
      beta        = raw_result$beta,
      se          = raw_result$se,
      pval        = raw_result$pval,
      smoothed_lr = smooth_result$smoothed_lr,
      pp          = smooth_result$pp,
      marker_type = marker_type,
      n_markers   = n_markers,
      pi          = pi,
      window      = window
    ),
    class = "gwas_result"
  )
}

# ============================================================
# Internal helpers
# ============================================================

#' Extract G_u matrix from masreml_fit
#' If not available, rebuild from markers
#' @noRd
.extract_g_u <- function(masreml_fit, markers, ids, ref_markers = NULL) {
  if (!is.null(markers$snp_add)) {
    w <- .validate_snp_matrix(markers$snp_add, "snp_add", ids)
    allele_freq <- if (!is.null(ref_markers$snp_add)) {
      ref_W <- .validate_snp_matrix(ref_markers$snp_add, "snp_add_ref")
      colMeans(ref_W) / 2
    } else NULL
    return(.build_g_snp(w, type = "additive", weights = NULL,
                        allele_freq = allele_freq))
  }
  if (!is.null(markers$mh_add)) {
    return(.build_g_mh(markers$mh_add, ids, weights = NULL,
                       ref_mh = ref_markers$mh_add))
  }
  stop("Cannot extract G_u: no marker data available.")
}

#' Prepare MH data untuk GWAS input
#' Returns W_αh flat matrix + block_sizes
#' @noRd
.prepare_mh_for_gwas <- function(mh_list, ids = NULL) {

  # Parse every chr
  parsed <- lapply(seq_along(mh_list), function(k) {
    df <- mh_list[[k]]
    chr_label <- if (!is.null(names(mh_list))) {
      names(mh_list)[k]
    } else {
      paste0("chr", k)
    }
    .parse_mh_chr(df, chr_label)
  })

  # Align IDs
  chr_ids <- parsed[[1]]$ids
  if (!is.null(ids)) {
    idx <- match(ids, chr_ids)
    parsed <- lapply(parsed, function(p) {
      list(
        ids       = p$ids[idx],
        hap1      = p$hap1[idx, , drop = FALSE],
        hap2      = p$hap2[idx, , drop = FALSE],
        n_alleles = p$n_alleles
      )
    })
  }

  # Stack hap1/hap2 column-wise
  hap1_all      <- do.call(cbind, lapply(parsed, `[[`, "hap1"))
  hap2_all      <- do.call(cbind, lapply(parsed, `[[`, "hap2"))
  n_alleles_all <- unlist(lapply(parsed, `[[`, "n_alleles"))

  # block_sizes = n_alleles - 1 per locus (kolom W_αh per blok)
  block_sizes <- pmax(n_alleles_all - 1L, 0L)

  # Build W_αh flat matrix via Rust
  list(
    hap1        = hap1_all,
    hap2        = hap2_all,
    n_alleles   = as.integer(n_alleles_all),
    block_sizes = as.integer(block_sizes)
  )
}

# ============================================================
# S3 methods
# ============================================================

#' @export
print.gwas_result <- function(x, ...) {
  cat("GWAS Result\n")
  cat(sprintf("  Marker type : %s\n", x$marker_type))
  cat(sprintf("  N markers   : %d\n", x$n_markers))
  cat(sprintf("  Prior (pi)  : %.4f\n", x$pi))
  cat(sprintf("  Window      : %d\n", x$window))
  cat(sprintf("  Top LR      : %.2f\n", max(x$lr)))
  cat(sprintf("  N PP > 0.5  : %d\n", sum(x$pp > 0.5)))
  cat(sprintf("  N PP > 0.9  : %d\n", sum(x$pp > 0.9)))
  invisible(x)
}

#' @export
summary.gwas_result <- function(object, ...) {
  cat("GWAS Summary\n")
  cat(sprintf("  Marker type      : %s\n", object$marker_type))
  cat(sprintf("  N markers/blocks : %d\n", object$n_markers))
  cat(sprintf("  Prior pi         : %.4f\n", object$pi))
  cat(sprintf("  Window size      : %d\n", object$window))
  cat("\nLR distribution:\n")
  print(quantile(object$lr, probs = c(0, 0.25, 0.5, 0.75, 0.95, 0.99, 1)))
  cat("\nPosterior probability (PP) distribution:\n")
  print(quantile(object$pp, probs = c(0, 0.25, 0.5, 0.75, 0.95, 0.99, 1)))
  cat(sprintf("\nN PP > 0.5 : %d (%.1f%%)\n",
    sum(object$pp > 0.5),
    100 * mean(object$pp > 0.5)
  ))
  cat(sprintf("N PP > 0.9 : %d (%.1f%%)\n",
    sum(object$pp > 0.9),
    100 * mean(object$pp > 0.9)
  ))
  invisible(object)
}