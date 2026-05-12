#' Universal REML-BLUP for SNP and Microhaplotype Genomic Prediction
#'
#' BLUP-based model for genomic prediction using SNP or multi-allelic
#' markers. Estimates variance components via REML and predicts genomic
#' estimated breeding values (GEBV) via BLUP. Supports SNP additive,
#' SNP dominance, and multi-allelic additive relationship matrices, for both
#' continuous and binary traits. For GWAS-assisted prediction, see
#' \code{\link{run_gwas}} and \code{\link{gwablup}}.
#'
#' @param y numeric vector of phenotypes (length n). Named vector
#'   recommended for ID alignment. For binary trait, must contain
#'   only 0 and 1.
#' @param X fixed effects design matrix (n x c). If NULL, intercept
#'   only is used.
#' @param markers list of raw marker inputs (optional):
#'   \itemize{
#'     \item \code{snp_add}: matrix (n x m), raw genotype 0/1/2
#'     \item \code{snp_dom}: matrix (n x m), raw genotype 0/1/2
#'     \item \code{mh_add}: list of data.frames, one per chromosome.
#'       First column = individual ID, remaining columns = paired
#'       haplotype allele codes per block (strand1, strand2, strand1, ...)
#'   }
#' @param G list of pre-built relationship matrices (optional):
#'   \itemize{
#'     \item \code{snp_add}: numeric matrix (n x n), SNP additive G
#'     \item \code{snp_dom}: numeric matrix (n x n), SNP dominance D
#'     \item \code{mh_add}: numeric matrix (n x n), multi-allelic additive Agh
#'     \item \code{pedigree}: numeric matrix (n x n), pedigree A
#'   }
#' @param method character, REML algorithm for variance component estimation:
#'   \itemize{
#'     \item \code{"auto"} (default): AI-REML for continuous,
#'       HE regression for binary
#'     \item \code{"HE"}: Haseman-Elston regression (fast, one-step)
#'     \item \code{"AI"}: Average Information REML (accurate, iterative)
#'     \item \code{"EM"}: Expectation-Maximization REML (stable, slower)
#'     \item \code{"HI"}: HE-initialized AI-REML
#'   }
#'   For binary trait, \code{"HE"} is recommended for speed.
#'   Any method can be used but \code{"AI"} may be slow on working response.
#' @param solver character, EBV solver:
#'   \itemize{
#'     \item \code{"auto"} (default): Cholesky for n < 10,000, PCG otherwise
#'     \item \code{"cholesky"}: direct Cholesky factorization
#'     \item \code{"pcg"}: preconditioned conjugate gradient (large n)
#'   }
#' @param max_iter integer, maximum REML iterations (default 100)
#' @param tol numeric, convergence tolerance (default 1e-6)
#' @param n_threads integer, number of threads (default: all physical cores)
#' @param trait character, trait distribution:
#'   \itemize{
#'     \item \code{"continuous"} (default): Gaussian REML-BLUP
#'     \item \code{"binary"}: 0/1 phenotype, single-step Laplace
#'       approximation on liability scale
#'   }
#' @param link character, link function for binary trait:
#'   \itemize{
#'     \item \code{"logit"} (default): logistic link, sigma2_e = pi^2/3
#'     \item \code{"probit"}: probit link, sigma2_e = 1
#'   }
#' @param verbose logical, if \code{TRUE} (default) prints a brief
#'   post-fit summary banner (algorithm, runtime, convergence, variance
#'   components, h2, GEBV summary) when \code{masreml()} returns. The
#'   existing progress messages (\code{"Building..."}, \code{"Running
#'   REML..."}, \code{"Solving EBV..."}) are unaffected by this flag.
#' @param save_rds logical, if \code{TRUE} the fit is also written to
#'   disk as an RDS file. Default \code{FALSE} \emph{(differs from
#'   \code{masbayes::run_bayesr()})} because \code{masreml()} is often
#'   called inside \code{\link{cv_masreml}} loops where auto-saving
#'   would clobber per-fold output. Set to \code{TRUE} for one-off
#'   full-data fits you want to persist.
#' @param save_path optional explicit RDS path. If \code{NULL} (default),
#'   defaults to \code{"results_masreml.Rds"} in the current working
#'   directory. Ignored unless \code{save_rds = TRUE}.
#'
#' @details
#' \strong{Statistical model.}
#' The general mixed linear model is:
#' \deqn{y = Xb + \sum_{k} Z_k u_k + e}
#' with independent random genetic effects
#' \eqn{u_k \sim \mathcal{N}(0,\,G_k \sigma_{g_k}^2)} and residuals
#' \eqn{e \sim \mathcal{N}(0,\,I \sigma_e^2)}, where \eqn{G_k} is the
#' genomic relationship matrix for component \eqn{k}
#' (\code{snp_add}, \code{snp_dom}, \code{mh_add}, or \code{pedigree}).
#' For a single additive microhaplotype component this reduces to:
#' \deqn{y = Xb + Z a_h + e, \quad a_h \sim \mathcal{N}(0,\,A_{gh}\,\sigma_{a_h}^2)}
#' where \eqn{A_{gh} = W_{ah} W_{ah}^\top / k_{ah}} and
#' \eqn{k_{ah} = \mathrm{tr}(W_{ah} W_{ah}^\top) / n}
#' (Da, 2015; see \code{\link{build_G_mh}}).
#' Variance components are estimated by REML; GEBVs are obtained by BLUP.
#'
#' \strong{Auto-summary.} On completion, \code{masreml()} prints a brief
#' banner mirroring \code{masbayes::run_bayesr()}/
#' \code{masbayes::run_bayesa()}. Set \code{verbose = FALSE} to suppress.
#' The full \code{\link{summary.masreml}} method is unchanged and is
#' still the canonical detailed report.
#'
#' \strong{Runtime field.} The returned object now contains a
#' \code{runtime} field (elapsed seconds for REML + BLUP), populated
#' even when \code{verbose = FALSE}.
#'
#' @return Object of class \code{"masreml"} with elements:
#'   \itemize{
#'     \item \code{gebv}: named list of GEBV vectors per component
#'     \item \code{total_gebv}: total GEBV (sum across components)
#'     \item \code{fixed_effects}: fixed effect estimates
#'     \item \code{varcomp}: list with \code{sigma2} and \code{h2}
#'     \item \code{loglik}: restricted log-likelihood
#'     \item \code{algorithm}: REML algorithm used
#'     \item \code{solver}: EBV solver used
#'     \item \code{converged}: logical, convergence status
#'     \item \code{n_iter}: number of REML iterations
#'     \item \code{n}: number of individuals
#'     \item \code{runtime}: elapsed seconds for REML + BLUP
#'     \item \code{rds_path}: path of saved RDS file, or \code{NULL}
#'       when \code{save_rds = FALSE}
#'     \item \code{training_metrics}: list of training-set fit metrics.
#'       Continuous: \code{R2}, \code{RMSE}, \code{accuracy} (Pearson
#'       \emph{r}), \code{bias} (slope of \eqn{y \sim \hat{y}}). Binary:
#'       same fields plus \code{AUC}, computed on the observed
#'       (probability) scale (\code{y_01} vs \code{mu_hat}) so that
#'       \code{bias} is the \emph{calibration slope} (1.0 = perfectly
#'       calibrated) and \code{RMSE\^2} approximates the Brier score.
#'       \code{AUC} is rank-invariant under inverse-link transformation.
#'     \item \code{call}: matched call
#'   }
#'   For binary trait (\code{trait = "binary"}), additional
#'   \code{binary} slot contains:
#'   \itemize{
#'     \item \code{link}: link function used
#'     \item \code{prevalence}: proportion of y = 1
#'     \item \code{h2_liability}: h2 on liability scale
#'     \item \code{h2_observed}: h2 on observed (0/1) scale
#'       via Dempster-Falconer transformation
#'     \item \code{auc}: Area Under ROC Curve (Wilcoxon-MWW)
#'     \item \code{fitted}: fitted probabilities P(y = 1)
#'     \item \code{converged}: always TRUE for single-step
#'   }
#'
#' @seealso \code{\link{run_gwas}}, \code{\link{gwablup}},
#'   \code{\link{cv_masreml}}, \code{\link{compute_accuracy}}
#' 
#' @references
#' VanRaden (2008) Efficient methods to compute genomic predictions.
#' \emph{J. Dairy Sci.} 91:4414-4423.
#'
#' Da (2015) Multi-allelic haplotype model for genomic prediction.
#' \emph{BMC Genet.} 16:144.
#'
#' Dempster & Falconer (1950) Interpretation of high and low liability.
#' \emph{Ann. Hum. Genet.} 31:195-203.
#'
#' Johnson & Thompson (1995) Restricted maximum likelihood estimation
#' of variance components. \emph{J. Dairy Sci.} 78:449-456.
#'
#' @examples
#' \dontrun{
#' d        <- load_data("small")
#' y        <- d$pheno$y_cont_qtl_snp;  names(y) <- d$pheno$id
#' y_binary <- as.integer(d$pheno$y_bin_qtl_snp); names(y_binary) <- d$pheno$id
#'
#' # ── Continuous trait ─────────────────────────────────────
#' # SNP additive GBLUP (markers list, masreml builds G internally)
#' fit <- masreml(y = y, markers = list(snp_add = d$snp))
#' summary(fit)
#'
#' # multi-allelic additive only (d$mh is auto-detected as a hap-block matrix)
#' fit <- masreml(y = y, markers = list(mh_add = d$mh))
#'
#' # Combined SNP additive + dominance
#' fit <- masreml(
#'   y       = y,
#'   markers = list(snp_add = d$snp, snp_dom = d$snp),
#'   method  = "AI"
#' )
#'
#' # Pre-built G matrix
#' G   <- build_G_snp(d$snp)
#' fit <- masreml(y = y, G = list(snp_add = G))
#'
#' # ── Binary trait ─────────────────────────────────────────
#' # Binary GBLUP (logit link, default HE)
#' fit <- masreml(y = y_binary, markers = list(snp_add = d$snp),
#'                trait = "binary")
#' summary(fit)
#'
#' # Binary with probit link
#' fit <- masreml(y = y_binary, markers = list(snp_add = d$snp),
#'                trait = "binary", link = "probit")
#' }
#'
#' @export
masreml <- function(
    y,
    X          = NULL,
    markers    = NULL,
    G          = NULL,
    method     = "auto",
    solver     = "auto",
    max_iter   = 100L,
    tol        = 1e-6,
    n_threads  = NULL,
    trait      = "continuous",   # "continuous" or "binary"
    link       = "logit",        # "logit" or "probit", binary only
    verbose    = TRUE,
    save_rds   = FALSE,
    save_path  = NULL
) {
  call <- match.call()

  # ── Input validation ────────────────────────────────────────
  y <- .validate_y(y)
  n <- length(y)
  ids <- names(y)

  X <- .prepare_X(X, n)

  if (is.null(markers) && is.null(G)) {
    stop("Provide at least one of 'markers' or 'G'.")
  }

  method <- match.arg(method, c("auto", "HE", "AI", "EM", "HI"))
  solver <- match.arg(solver, c("auto", "cholesky", "pcg"))
  trait  <- match.arg(trait,  c("continuous", "binary"))
  link   <- match.arg(link,   c("logit", "probit"))
  max_iter  <- as.integer(max_iter)
  n_threads <- .resolve_threads(n_threads)

  # ── Build G matrices ────────────────────────────────────────
  message("Building relationship matrices...")
  g_list <- .build_g_matrices(markers = markers, G = G, ids = ids)

  # ── Route binary trait ──────────────────────────────────────
  if (trait == "binary") {
    message("Binary trait detected — running GLMM via Laplace approximation...")
    method_binary <- if (method == "auto") "HE" else method
    return(.masreml_binary(
      y         = y,
      X         = X,
      g_list    = g_list,
      link      = link,
      method    = method_binary,
      solver    = solver,
      max_iter  = max_iter,
      tol       = tol,
      n_threads = n_threads,
      ids       = ids,
      call      = call,
      verbose   = verbose,
      save_rds  = save_rds,
      save_path = save_path
    ))
  }

  # ── Resolve solver ──────────────────────────────────────────
  solver_used <- if (solver == "auto") {
    if (n < 10000L) "cholesky" else "pcg"
  } else {
    solver
  }

  # ── Run REML + BLUP (timed) ─────────────────────────────────
  message(sprintf(
    "Running REML [method=%s, n=%d, n_components=%d]...",
    method, n, length(g_list)
  ))

  timing <- system.time({
    reml_result <- .run_reml(
      y         = y,
      X         = X,
      g_list    = g_list,
      method    = method,
      max_iter  = max_iter,
      tol       = tol,
      n_threads = n_threads
    )

    if (!is.null(reml_result$error) && !is.na(reml_result$error)) {
      stop(sprintf("REML failed: %s", reml_result$error))
    }

    message(sprintf("Solving EBV [solver=%s]...", solver_used))
    blup_result <- .solve_blup(
      y          = y,
      X          = X,
      g_list     = g_list,
      sigma2     = reml_result$sigma2,
      solver     = solver_used,
      max_iter   = max_iter,
      tol        = tol
    )
  })

  # ── Assemble output ─────────────────────────────────────────
  fit <- .assemble_output(
    y           = y,
    ids         = ids,
    g_names     = names(g_list),
    reml_result = reml_result,
    blup_result = blup_result,
    solver_used = solver_used,
    call        = call
  )

  # Training fitted values: y_hat = X * b_fixed + total_gebv
  y_hat <- as.numeric(X %*% blup_result$fixed_effects) +
    as.numeric(blup_result$total_gebv)
  fit$training_metrics <- .compute_training_metrics(y, y_hat)

  fit$runtime  <- as.numeric(timing["elapsed"])
  fit$rds_path <- .maybe_save_rds_masreml(fit, save_rds, save_path)

  if (isTRUE(verbose)) .print_run_summary(fit)

  fit
}

# ============================================================
# Internal helpers
# ============================================================

#' @noRd
.validate_y <- function(y) {
  if (!is.numeric(y)) stop("'y' must be a numeric vector.")
  if (anyNA(y)) stop("'y' contains NA. Remove or impute missing phenotypes.")
  if (length(y) < 2) stop("'y' must have at least 2 observations.")
  y
}

#' @noRd
.prepare_X <- function(X, n) {
  if (is.null(X)) {
    X <- matrix(1.0, nrow = n, ncol = 1)
    colnames(X) <- "(Intercept)"
    return(X)
  }
  if (is.data.frame(X)) X <- as.matrix(X)
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("'X' must be a numeric matrix or data.frame.")
  }
  if (nrow(X) != n) {
    stop(sprintf("'X' nrows (%d) != length(y) (%d).", nrow(X), n))
  }
  if (anyNA(X)) stop("'X' contains NA values.")
  X
}

#' @noRd
.resolve_threads <- function(n_threads) {
  if (is.null(n_threads)) {
    max(1L, parallel::detectCores(logical = FALSE))
  } else {
    as.integer(max(1L, n_threads))
  }
}

#' @noRd
.assemble_output <- function(
    y, ids, g_names, reml_result, blup_result, solver_used, call
) {
  n <- length(y)

  # Named variance components
  sigma2_names <- c(g_names, "residual")
  sigma2 <- setNames(reml_result$sigma2, sigma2_names)
  h2     <- setNames(reml_result$h2, g_names)

  # Named EBV per component
  gebv_list <- setNames(blup_result$gebv_per_component, g_names)
  if (!is.null(ids)) {
    gebv_list <- lapply(gebv_list, function(v) {
      names(v) <- ids; v
    })
  }

  # Total GEBV
  total_gebv <- blup_result$total_gebv
  if (!is.null(ids)) names(total_gebv) <- ids

  # Fixed effects
  fixed_effects <- blup_result$fixed_effects

  structure(
    list(
      gebv          = gebv_list,
      total_gebv    = total_gebv,
      fixed_effects = fixed_effects,
      varcomp       = list(
        sigma2 = sigma2,
        h2     = h2
      ),
      loglik        = reml_result$loglik,
      algorithm     = reml_result$algorithm,
      solver        = solver_used,
      converged     = reml_result$converged,
      n_iter        = reml_result$n_iter,
      n             = n,
      call          = call
    ),
    class = "masreml"
  )
}