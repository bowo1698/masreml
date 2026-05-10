# ============================================================
# Binary trait GLMM via Laplace approximation (single-step)
# Gaussian approximation on liability scale
# ============================================================

#' @noRd
.masreml_binary <- function(
    y, X, g_list, link,
    method, solver, max_iter, tol, n_threads,
    ids, call,
    verbose   = TRUE,
    save_rds  = FALSE,
    save_path = NULL
) {
  n <- length(y)

  # ── Validate binary input ───────────────────────────────────
  if (!all(y %in% c(0, 1))) {
    stop("For trait='binary', y must contain only 0 and 1.")
  }

  prevalence <- mean(y)
  if (prevalence == 0 || prevalence == 1) {
    stop("Binary y must have both 0 and 1 observations.")
  }

  # ── Working response from prevalence ────────────────────────
  lf      <- .link_functions(link)
  mu_init <- rep(pmax(pmin(prevalence, 0.99), 0.01), n)
  eta_init <- lf$fun(mu_init)
  W_init   <- pmax(lf$deriv(mu_init), 1e-6)
  z_init   <- eta_init + (y - mu_init) / W_init
  names(z_init) <- ids

  # ── Sigma2 estimate of working response ───────────────────
  message(sprintf(
    "Estimating variance components [method=%s, n=%d]...", method, n
  ))

  solver_used <- if (solver == "auto") {
    if (n < 10000L) "cholesky" else "pcg"
  } else solver

  timing <- system.time({
    reml_init <- .run_reml(
      y         = z_init,
      X         = X,
      g_list    = g_list,
      method    = method,
      max_iter  = max_iter,
      tol       = tol,
      n_threads = n_threads
    )

    if (!is.null(reml_init$error) && !is.na(reml_init$error)) {
      stop(sprintf("REML failed: %s", reml_init$error))
    }

    # sigma2_e to theoretical value
    sigma2_e_fixed <- if (link == "logit") pi^2 / 3 else 1.0
    sigma2_e_reml  <- reml_init$sigma2[length(reml_init$sigma2)]
    scale_factor   <- sigma2_e_fixed / sigma2_e_reml
    sigma2_u_reml  <- reml_init$sigma2[seq_len(length(g_list))]
    sigma2_u_scaled <- sigma2_u_reml * scale_factor
    sigma2_u_max    <- var(z_init) - sigma2_e_fixed  # max possible sigma2_u
    sigma2_u_scaled <- pmax(pmin(sigma2_u_scaled, sigma2_u_max), 1e-4)
    sigma2_fixed    <- c(sigma2_u_scaled, sigma2_e_fixed)

    # ── Solve BLUP once on working response ─────────────
    message("Solving BLUP on liability scale...")

    blup_result <- .solve_blup(
      y        = z_init,
      X        = X,
      g_list   = g_list,
      sigma2   = sigma2_fixed,
      solver   = solver_used,
      max_iter = max_iter,
      tol      = tol
    )
  })

  # ── Compute fitted probabilities ─────────────────────────────
  # eta = X*b + u (linear predictor on liability scale)
  u_hat  <- as.numeric(blup_result$total_gebv)
  b_hat  <- as.numeric(blup_result$fixed_effects)
  eta_hat <- as.numeric(X %*% b_hat) + u_hat
  mu_hat  <- lf$inv(eta_hat)
  mu_hat  <- pmax(pmin(mu_hat, 1 - 1e-6), 1e-6)

  if (!is.null(ids)) {
    names(u_hat)  <- ids
    names(mu_hat) <- ids
  }

  # ── Compute binary metrics ───────────────────────────────────
  sigma2_u     <- sigma2_fixed[seq_len(length(g_list))]
  h2_liability <- sum(sigma2_u) / (sum(sigma2_u) + sigma2_e_fixed)
  h2_observed  <- .h2_liability_to_observed(h2_liability, prevalence, link)
  auc          <- .compute_auc(y, mu_hat)

  # ── Assemble binary slot ────────────────────────────────────
  binary_info <- list(
    link           = link,
    prevalence     = prevalence,
    h2_liability   = h2_liability,
    h2_observed    = h2_observed,
    auc            = auc,
    fitted         = mu_hat,
    n_iter         = 1L,
    converged      = TRUE
  )

  # ── Assemble main output ─────────────────────────────────────
  g_names      <- names(g_list)
  sigma2_names <- c(g_names, "residual")
  sigma2       <- setNames(sigma2_fixed, sigma2_names)
  h2           <- setNames(sigma2_u / sum(sigma2_fixed), g_names)

  gebv_list <- setNames(blup_result$gebv_per_component, g_names)
  gebv_list <- lapply(gebv_list, function(v) {
    if (!is.null(ids)) names(v) <- ids
    v
  })

  total_gebv <- u_hat
  if (!is.null(ids)) names(total_gebv) <- ids

  fit <- structure(
    list(
      gebv          = gebv_list,
      total_gebv    = total_gebv,
      fixed_effects = b_hat,
      varcomp       = list(sigma2 = sigma2, h2 = h2),
      loglik        = reml_init$loglik,
      algorithm     = paste0("Laplace-1step (", reml_init$algorithm, ")"),
      solver        = solver_used,
      converged     = TRUE,
      n_iter        = 1L,
      n             = n,
      trait         = "binary",
      binary        = binary_info,
      call          = call,
      runtime       = as.numeric(timing["elapsed"])
    ),
    class = "masreml"
  )

  fit$rds_path <- .maybe_save_rds_masreml(fit, save_rds, save_path)

  if (isTRUE(verbose)) .print_run_summary(fit)

  fit
}