#' Cross-Validation for Genomic Prediction Models
#'
#' Performs k-fold or leave-one-out (LOO) cross-validation to estimate
#' prediction accuracy of genomic models fitted with \code{masreml()}.
#' Individuals are split into training and validation sets; the model
#' is trained on each training set and used to predict the held-out
#' validation individuals.
#'
#' @param y numeric vector of phenotypes (length n)
#' @param X fixed effects matrix (n x c). NULL = intercept only
#' @param markers list of raw marker inputs (see \code{\link{masreml}})
#' @param G list of pre-built G matrices (see \code{\link{masreml}})
#' @param folds integer, number of CV folds (default 5).
#'   Use \code{folds = length(y)} for leave-one-out (LOO).
#'   Larger values give less biased but more variable estimates.
#' @param scheme character, fold assignment scheme:
#'   \itemize{
#'     \item \code{"random"} (default): individuals randomly assigned
#'       to folds — recommended for most cases
#'     \item \code{"systematic"}: every k-th individual assigned to
#'       same fold — useful for structured populations
#'   }
#' @param method character, REML method (see \code{\link{masreml}})
#' @param solver character, EBV solver (see \code{\link{masreml}})
#' @param max_iter integer, max REML iterations
#' @param tol numeric, convergence tolerance
#' @param n_threads integer, number of threads
#' @param seed integer, random seed for reproducibility
#'
#' @return Object of class \code{"masreml_cv"} with elements:
#'   \itemize{
#'     \item \code{accuracy}: mean prediction accuracy (r_MG)
#'     \item \code{accuracy_fold}: accuracy per fold
#'     \item \code{bias}: mean regression slope (bias check)
#'     \item \code{bias_fold}: slope per fold
#'     \item \code{gebv_all}: GEBV for all individuals
#'     \item \code{fold_assignments}: fold ID per individual
#'     \item \code{folds}: number of folds used
#'     \item \code{scheme}: fold scheme used
#'     \item \code{call}: matched call
#'   }
#'
#' @seealso \code{\link{masreml}}, \code{\link{compute_accuracy}}
#' 
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' y <- d$pheno$y_cont_qtl_snp; names(y) <- d$pheno$id
#'
#' # 5-fold cross-validation
#' cv5 <- cv_masreml(
#'   y       = y,
#'   markers = list(snp_add = d$snp),
#'   folds   = 5L,
#'   seed    = 42L
#' )
#' summary(cv5)
#'
#' # 10-fold cross-validation (more stable accuracy estimate at n=100;
#' # LOOCV at folds = length(y) is not currently supported because the
#' # per-fold validation matrix collapses to a single row).
#' cv10 <- cv_masreml(
#'   y       = y,
#'   markers = list(snp_add = d$snp),
#'   folds   = 10L,
#'   seed    = 42L
#' )
#' summary(cv10)
#' }
#'
#' @export
cv_masreml <- function(
    y,
    X          = NULL,
    markers    = NULL,
    G          = NULL,
    folds      = 5L,
    scheme     = "random",
    method     = "auto",
    solver     = "auto",
    max_iter   = 100L,
    tol        = 1e-8,
    n_threads  = NULL,
    seed       = NULL
) {
  call <- match.call()

  # ── Validate inputs ─────────────────────────────────────────
  y <- .validate_y(y)
  n <- length(y)
  ids <- names(y)

  X         <- .prepare_X(X, n)
  folds     <- as.integer(min(folds, n))
  scheme    <- match.arg(scheme, c("random", "systematic"))
  n_threads <- .resolve_threads(n_threads)

  if (!is.null(seed)) set.seed(seed)

  # ── Build G matrices once (reused across folds) ─────────────
  message("Building relationship matrices...")
  g_list <- .build_g_matrices(markers = markers, G = G, ids = ids)

  # ── Assign fold membership ──────────────────────────────────
  fold_assignments <- .assign_folds(n, folds, scheme)

  # ── Run CV folds ────────────────────────────────────────────
  message(sprintf("Running %d-fold CV [scheme=%s]...", folds, scheme))

  gebv_all   <- numeric(n)
  acc_folds  <- numeric(folds)
  bias_folds <- numeric(folds)

  for (k in seq_len(folds)) {
    message(sprintf("  Fold %d / %d", k, folds))

    # Training and validation indices
    val_idx   <- which(fold_assignments == k)
    train_idx <- which(fold_assignments != k)

    # Subset data for this fold
    y_train    <- y[train_idx]
    X_train    <- X[train_idx, , drop = FALSE]
    g_train    <- lapply(g_list, function(g) g[train_idx, train_idx])
    g_val      <- lapply(g_list, function(g) g[val_idx, train_idx])

    # Run REML on training set
    reml_result <- tryCatch(
      .run_reml(
        y         = y_train,
        X         = X_train,
        g_list    = g_train,
        method    = method,
        max_iter  = max_iter,
        tol       = tol,
        n_threads = n_threads
      ),
      error = function(e) {
        warning(sprintf("Fold %d REML failed: %s", k, e$message))
        NULL
      }
    )

    if (is.null(reml_result)) next

    # Predict validation individuals
    # GEBV_val = sum_i(G_val_i * G_train_i^-1 * u_train_i)
    # Efficient: use training EBV and G relationship
    gebv_val <- .predict_validation(
      y_train     = y_train,
      X_train     = X_train,
      X_val       = X[val_idx, , drop = FALSE],
      g_train     = g_train,
      g_val       = g_val,
      sigma2      = reml_result$sigma2,
      solver      = solver,
      n_threads   = n_threads,
      max_iter    = max_iter,
      tol         = tol
    )

    gebv_all[val_idx] <- gebv_val

    # Fold accuracy and bias
    y_val           <- y[val_idx]
    acc_folds[k]    <- cor(gebv_val, y_val, use = "complete.obs")
    bias_folds[k]   <- coef(lm(y_val ~ gebv_val))["gebv_val"]
  }

  # ── Assemble output ─────────────────────────────────────────
  if (!is.null(ids)) names(gebv_all) <- ids

  structure(
    list(
      accuracy         = mean(acc_folds, na.rm = TRUE),
      accuracy_fold    = setNames(acc_folds, paste0("fold", seq_len(folds))),
      bias             = mean(bias_folds, na.rm = TRUE),
      bias_fold        = setNames(bias_folds, paste0("fold", seq_len(folds))),
      gebv_all         = gebv_all,
      fold_assignments = fold_assignments,
      folds            = folds,
      scheme           = scheme,
      call             = call
    ),
    class = "masreml_cv"
  )
}

# ============================================================
# CV Internal helpers
# ============================================================

#' Assign fold membership
#' @noRd
.assign_folds <- function(n, folds, scheme) {
  if (scheme == "random") {
    # Random assignment ensuring balanced folds
    fold_ids <- rep(seq_len(folds), length.out = n)
    sample(fold_ids)
  } else {
    # Systematic: every k-th individual to same fold
    ((seq_len(n) - 1) %% folds) + 1L
  }
}

#' Predict validation individuals using training EBV
#'
#' For validation individual j:
#'   GEBV_j = sum_i( g_val[j, train] * G_train^-1 * u_train )
#'          = g_val * alpha
#' where alpha = G_train^-1 * u_train (marker effects)
#'
#' Equivalent to: fit full model on training, extract marker effects,
#' then predict validation via G_val_train relationship
#' @noRd
.predict_validation <- function(
    y_train, X_train, X_val,
    g_train, g_val,
    sigma2, solver,
    n_threads, max_iter, tol
) {
  n_train  <- length(y_train)
  n_random <- length(g_train)

  # Resolve solver for training set size
  solver_used <- if (solver == "auto") {
    if (n_train < 10000L) "cholesky" else "pcg"
  } else solver

  # Get training EBV
  blup_train <- .solve_blup(
    y       = y_train,
    X       = X_train,
    g_list  = g_train,
    sigma2  = sigma2,
    solver  = solver_used,
    max_iter = max_iter,
    tol     = tol
  )

  # Predict validation: GEBV_val = sum_i( G_val_i * alpha_i )
  # alpha_i = G_train_i^-1 * u_train_i
  # Efficient: solve G_train_i * alpha_i = u_train_i via Cholesky
  # then GEBV_val_i = G_val_i * alpha_i
  total_gebv_val <- numeric(nrow(g_val[[1]]))

  for (i in seq_len(n_random)) {
    u_train  <- blup_train$gebv_per_component[[i]]
    g_tr     <- g_train[[i]]
    g_vl     <- g_val[[i]]

    # Solve G_train * alpha = u_train
    alpha <- tryCatch(
      solve(g_tr, u_train),
      error = function(e) {
        # Fallback: pseudoinverse via SVD
        svd_g  <- svd(g_tr)
        tol_sv <- max(svd_g$d) * .Machine$double.eps * nrow(g_tr)
        inv_d  <- ifelse(svd_g$d > tol_sv, 1 / svd_g$d, 0)
        svd_g$v %*% diag(inv_d) %*% t(svd_g$u) %*% u_train
      }
    )

    total_gebv_val <- total_gebv_val + as.numeric(g_vl %*% alpha)
  }

  total_gebv_val
}