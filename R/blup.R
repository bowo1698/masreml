#' EBV solver wrapper: routes to Rust Cholesky or PCG
#'
#' Internal function called by masreml().
#'
#' @param y numeric vector (n)
#' @param X numeric matrix (n x c)
#' @param g_list named list of G matrices
#' @param sigma2 numeric vector of variance components
#'   [sigma2_1, ..., sigma2_k, sigma2_e]
#' @param solver character, "cholesky" or "pcg"
#' @param max_iter integer (for PCG)
#' @param tol numeric (for PCG)
#' @return list: gebv_per_component, total_gebv, fixed_effects
#' @noRd
.solve_blup <- function(y, X, g_list, sigma2, solver, max_iter, tol) {
  g_rlist <- .g_list_to_rlist(g_list)
  
  result <- r_solve_ebv(
    y        = as.double(y),
    x        = X,
    g_list   = g_rlist,
    sigma2   = as.double(sigma2),
    solver   = solver,
    max_iter = as.integer(max_iter),
    tol      = as.double(tol)
  )
  
  .parse_blup_result(result, g_rlist)
}

#' Parse Rust solver output into R list
#' @noRd
.parse_blup_result <- function(result, g_rlist) {
  n_random <- length(g_rlist)
  g_names  <- names(g_rlist)

  # Extract per-component EBV
  # Rust returns: fixed_effects, total_gebv, solver, n_iter
  # + one vector per component named by g_names
  gebv_per_component <- vector("list", n_random)
  names(gebv_per_component) <- g_names

  for (nm in g_names) {
    if (!is.null(result[[nm]])) {
      gebv_per_component[[nm]] <- as.numeric(result[[nm]])
    }
  }

  list(
    gebv_per_component = gebv_per_component,
    total_gebv         = as.numeric(result$total_gebv),
    fixed_effects      = as.numeric(result$fixed_effects),
    solver             = as.character(result$solver),
    n_iter             = as.integer(result$n_iter)
  )
}

#' Compute prediction accuracy (correlation GEBV vs phenotype)
#' @noRd
.compute_accuracy <- function(gebv, y) {
  if (length(gebv) != length(y)) {
    stop("GEBV and y must have same length for accuracy computation.")
  }
  cor(gebv, y, use = "complete.obs")
}

#' Compute regression slope (bias check)
#' Convention: lm(observed ~ predicted)
#' Slope = 1 → unbiased
#' @noRd
.compute_bias <- function(gebv, y) {
  fit <- lm(y ~ gebv)
  coef(fit)["gebv"]
}