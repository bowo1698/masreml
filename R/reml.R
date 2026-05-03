#' REML wrapper: routes to Rust adaptive REML
#'
#' Internal function called by masreml().
#' Handles input preparation and output parsing.
#'
#' @param y numeric vector (n)
#' @param X numeric matrix (n x c)
#' @param g_list named list of G matrices (n x n each)
#' @param method character, one of "auto","HE","AI","EM","HI"
#' @param max_iter integer
#' @param tol numeric
#' @param n_threads integer
#' @return list: sigma2, h2, loglik, algorithm, converged, n_iter
#' @noRd
.run_reml <- function(y, X, g_list, method, max_iter, tol, n_threads) {

  # Convert g_list to R list format expected by Rust
  g_rlist <- .g_list_to_rlist(g_list)

  # Call Rust adaptive REML
  result <- r_run_reml(
    y         = as.double(y),
    x         = X,
    g_list    = g_rlist,
    method    = method,
    max_iter  = as.integer(max_iter),
    tol       = as.double(tol),
    n_threads = as.integer(n_threads)
  )

  # Check for Rust-side error
  if (!is.null(result$error) && !is.na(result$error)) {
    stop(sprintf("REML error: %s", result$error))
  }

  # Parse and return
  list(
    sigma2    = as.numeric(result$sigma2),
    h2        = as.numeric(result$h2),
    loglik    = as.numeric(result$loglik),
    algorithm = as.character(result$algorithm),
    converged = as.logical(result$converged),
    n_iter    = as.integer(result$n_iter),
    error     = NULL
  )
}

#' Convert named G list to format for Rust
#' Rust expects: list of named numeric matrices
#' @noRd
.g_list_to_rlist <- function(g_list) {
  lapply(g_list, function(g) {
    storage.mode(g) <- "double"
    g
  })
}

#' Estimate initial variance components via simple method-of-moments
#' Used as fallback if HE regression fails
#' @noRd
.init_sigma2_mom <- function(y, n_random, prop_genetic = 0.3) {
  var_y <- var(y)
  s2_genetic <- var_y * prop_genetic / n_random
  s2_e <- var_y * (1 - prop_genetic)
  c(rep(s2_genetic, n_random), s2_e)
}

#' Print REML convergence summary
#' @noRd
.print_reml_summary <- function(result, g_names) {
  cat("\n── REML Results ──────────────────────────────\n")
  cat(sprintf("  Algorithm : %s\n", result$algorithm))
  cat(sprintf("  Iterations: %d\n", result$n_iter))
  cat(sprintf("  Converged : %s\n", result$converged))
  cat(sprintf("  Log-lik   : %.4f\n", result$loglik))
  cat("\n  Variance Components:\n")
  sigma2_names <- c(g_names, "residual")
  for (i in seq_along(result$sigma2)) {
    cat(sprintf("    %-12s: %.6f\n", sigma2_names[i], result$sigma2[i]))
  }
  cat("\n  Heritability (h²):\n")
  for (i in seq_along(result$h2)) {
    cat(sprintf("    %-12s: %.4f\n", g_names[i], result$h2[i]))
  }
  cat("──────────────────────────────────────────────\n\n")
}