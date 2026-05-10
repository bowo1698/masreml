#' Print Genomic Prediction Results
#'
#' Compact display of a fitted \code{masreml} object showing variance
#' components, heritability, and GEBV summary.
#'
#' @param x object of class \code{"masreml"} from \code{masreml()} or
#'   \code{gwablup()}.
#' @param ... ignored.
#' @seealso \code{\link{summary.masreml}}, \code{\link{masreml}}
#' @noRd
print.masreml <- function(x, ...) {
  cat("\n── masreml: Genomic Prediction Results ───────────────\n")
  cat(sprintf("  Call      : %s\n", deparse(x$call)))
  cat(sprintf("  n         : %d individuals\n", x$n))
  cat(sprintf("  Components: %s\n", paste(names(x$varcomp$sigma2), collapse = ", ")))
  cat(sprintf("  Algorithm : %s\n", x$algorithm))
  cat(sprintf("  Solver    : %s\n", x$solver))
  cat(sprintf("  Converged : %s (iter = %d)\n", x$converged, x$n_iter))
  cat(sprintf("  Log-lik   : %.4f\n\n", x$loglik))

  cat("  Variance Components:\n")
  for (nm in names(x$varcomp$sigma2)) {
    cat(sprintf("    %-14s: %.6f\n", nm, x$varcomp$sigma2[nm]))
  }

  cat("\n  Heritability (h²):\n")
  for (nm in names(x$varcomp$h2)) {
    cat(sprintf("    %-14s: %.4f\n", nm, x$varcomp$h2[nm]))
  }

  cat("\n  GEBV summary (total):\n")
  g <- x$total_gebv
  cat(sprintf(
    "    Min=%.4f  Mean=%.4f  Max=%.4f  SD=%.4f\n",
    min(g), mean(g), max(g), sd(g)
  ))

  # Binary info
  if (!is.null(x$binary)) {
    cat(sprintf("\n  Trait     : binary (%s link)\n", x$binary$link))
    cat(sprintf("  AUC       : %.4f\n", x$binary$auc))
    cat(sprintf("  h2 (obs)  : %.4f\n", x$binary$h2_observed))
  }
  cat("──────────────────────────────────────────────────────\n\n")

  invisible(x)
}

#' Summarize Genomic Prediction Results
#'
#' Detailed summary of a fitted \code{masreml} object including model
#' information, variance components table, heritability, and GEBV
#' distribution per component. For binary traits, also reports AUC
#' and heritability on liability and observed scales.
#'
#' @param object object of class \code{"masreml"} from \code{masreml()}
#'   or \code{gwablup()}.
#' @param ... ignored.
#' @seealso \code{\link{print.masreml}}, \code{\link{varcomp}},
#'   \code{\link{masreml}}
#' @export
summary.masreml <- function(object, ...) {
  x <- object
  cat("\n══ masreml Summary ═══════════════════════════════════\n\n")

  # Model info
  cat("Model:\n")
  cat(sprintf("  %-16s: %s\n", "Call", deparse(x$call)))
  cat(sprintf("  %-16s: %d\n", "Individuals", x$n))
  cat(sprintf("  %-16s: %s\n", "Components",
    paste(names(x$varcomp$sigma2), collapse = ", ")))
  cat(sprintf("  %-16s: %.4f\n", "Log-likelihood", x$loglik))
  cat(sprintf("  %-16s: %s\n", "Algorithm", x$algorithm))
  cat(sprintf("  %-16s: %s\n", "Solver", x$solver))
  cat(sprintf("  %-16s: %s (iterations = %d)\n",
    "Converged", x$converged, x$n_iter))

  # Variance components table
  cat("\nVariance Components:\n")
  sigma2_vec <- x$varcomp$sigma2
  sigma2_p   <- sum(sigma2_vec)
  h2_full    <- c(x$varcomp$h2, residual = NA_real_)

  vc_df <- data.frame(
    Component  = names(sigma2_vec),
    Sigma2     = round(sigma2_vec, 6),
    H2         = round(h2_full, 4),
    Proportion = round(sigma2_vec / sigma2_p, 4),
    row.names  = NULL
  )
  print(vc_df, row.names = FALSE)

  # Training performance (mirror of masbayes summary's "Training fit" block)
  if (!is.null(x$training_metrics)) {
    m <- x$training_metrics
    is_binary <- !is.null(x$binary)
    header <- if (is_binary)
      "\nTraining Performance (observed/probability scale):\n"
    else
      "\nTraining Performance:\n"
    cat(header)
    fmt <- function(v) {
      if (is.null(v) || !is.finite(v)) "NA"
      else formatC(v, digits = 4, format = "g")
    }
    if (is_binary) {
      cat(sprintf("  %-16s: %s\n", "AUC",            fmt(m$AUC)))
    } else {
      cat(sprintf("  %-16s: %s\n", "accuracy (r)",   fmt(m$accuracy)))
    }
    cat(sprintf("  %-16s: %s\n",   "R^2",            fmt(m$R2)))
    cat(sprintf("  %-16s: %s\n",   "RMSE",           fmt(m$RMSE)))
    cat(sprintf("  %-16s: %s\n",   "bias (slope)",   fmt(m$bias)))
    if (is_binary)
      cat(sprintf("  %-16s: %s\n", "accuracy (r)",   fmt(m$accuracy)))
  }

  # GEBV summary per component
  cat("\nGEBV Summary:\n")
  all_gebv <- c(x$gebv, list(total = x$total_gebv))
  gebv_summary <- do.call(rbind, lapply(names(all_gebv), function(nm) {
    g <- all_gebv[[nm]]
    data.frame(
      Component = nm,
      Min       = round(min(g), 4),
      Mean      = round(mean(g), 4),
      Max       = round(max(g), 4),
      SD        = round(sd(g), 4),
      row.names = NULL
    )
  }))
  print(gebv_summary, row.names = FALSE)

  # Binary trait metrics
  if (!is.null(x$binary)) {
    cat("\nBinary Trait (GLMM Laplace):\n")
    cat(sprintf("  %-16s: %s\n",   "Link",           x$binary$link))
    cat(sprintf("  %-16s: %.4f\n", "Prevalence",     x$binary$prevalence))
    cat(sprintf("  %-16s: %.4f\n", "AUC",            x$binary$auc))
    cat(sprintf("  %-16s: %.4f\n", "h2 (liability)", x$binary$h2_liability))
    cat(sprintf("  %-16s: %.4f\n", "h2 (observed)",  x$binary$h2_observed))
    cat(sprintf("  %-16s: %d\n",   "IRLS iter",      x$binary$n_irls))
    cat(sprintf("  %-16s: %s\n",   "IRLS converged", x$binary$irls_converged))

    cat("\nFitted Probabilities:\n")
    fp <- x$binary$fitted
    cat(sprintf(
      "  Min=%.4f  Mean=%.4f  Max=%.4f  SD=%.4f\n",
      min(fp), mean(fp), max(fp), sd(fp)
    ))
  }

  cat("\n══════════════════════════════════════════════════════\n\n")
  invisible(x)
}

# ============================================================
# Auto-summary printed at the end of masreml() / .masreml_binary()
# Mirrors masbayes::print_run_summary() but adapted for REML-BLUP.
# Continuous and binary traits share a skeleton; the binary path
# adds a Binary Trait block and reports liability/observed h2.
# ============================================================

#' @noRd
.print_run_summary <- function(x) {
  is_binary <- !is.null(x$binary)
  trait_tag <- if (is_binary)
    sprintf(" [binary trait, %s link]", x$binary$link)
  else
    " [continuous trait]"

  rds_line <- if (!is.null(x$rds_path))
    sprintf(" RDS saved to    : %s\n", x$rds_path) else ""

  n_random <- length(x$varcomp$h2)
  n_fixed  <- if (!is.null(x$fixed_effects)) length(x$fixed_effects) else 0L
  comp_line <- sprintf("%s  (%d random + %d fixed)",
                       paste(names(x$varcomp$h2), collapse = ", "),
                       n_random, n_fixed)

  cat("\n============================================================\n")
  cat(sprintf(" masreml — REML-BLUP (algorithm: %s)%s\n",
              x$algorithm, trait_tag))
  cat("============================================================\n")
  cat(sprintf(" Observations    : n = %d\n", x$n))
  cat(sprintf(" Components      : %s\n", comp_line))
  if (!is.null(x$runtime))
    cat(sprintf(" Runtime         : %.2f seconds\n", x$runtime))
  cat(sprintf(" Solver          : %s\n", x$solver))
  cat(rds_line)

  conv_str <- if (isTRUE(x$converged))
    sprintf("TRUE  (iterations = %d, log-lik = %.4f)",
            x$n_iter, x$loglik)
  else
    sprintf("FALSE (iterations = %d, did not converge — inspect fit$loglik)",
            x$n_iter)
  cat(sprintf("\n REML convergence: %s\n", conv_str))

  # Variance components table
  sigma2_vec <- x$varcomp$sigma2
  sigma2_p   <- sum(sigma2_vec)
  prop_vec   <- sigma2_vec / sigma2_p
  h2_vec     <- x$varcomp$h2

  cat("\n Variance Components\n")
  cat(" ----------------------------------------\n")
  cat(sprintf("   %-14s %10s %8s %8s\n",
              "Component", "sigma2", "%var", "h2"))
  for (nm in names(sigma2_vec)) {
    h2_show <- if (nm %in% names(h2_vec))
      sprintf("%8.4f", h2_vec[[nm]]) else "       —"
    extra <- if (is_binary && nm == "residual") {
      if (identical(x$binary$link, "logit"))
        " (fixed: pi^2/3)"
      else
        " (fixed: 1.0)"
    } else {
      ""
    }
    cat(sprintf("   %-14s %10.4f %8.4f %s%s\n",
                nm, sigma2_vec[[nm]], prop_vec[[nm]], h2_show, extra))
  }
  cat(" ----------------------------------------\n")

  if (is_binary) {
    cat(sprintf("   Total h2 (liability): %.4f  |  (observed): %.4f\n",
                x$binary$h2_liability, x$binary$h2_observed))
  } else {
    cat(sprintf("   Total h2 (genetic) : %.4f\n", sum(h2_vec)))
  }

  # Binary trait extra block
  if (is_binary) {
    cat("\n Binary Trait (Laplace)\n")
    cat(" ----------------------------------------\n")
    cat(sprintf("   Prevalence    : %.4f\n", x$binary$prevalence))
    cat(sprintf("   AUC           : %.4f\n", x$binary$auc))
    fp <- x$binary$fitted
    cat(sprintf("   Fitted P(y=1) : min=%.4f  mean=%.4f  max=%.4f\n",
                min(fp), mean(fp), max(fp)))
  }

  # GEBV one-liner
  g <- x$total_gebv
  gebv_label <- if (is_binary) "GEBV (liability)" else "GEBV (total)    "
  cat(sprintf("\n %s: min=%.4f  mean=%.4f  max=%.4f  sd=%.4f\n",
              gebv_label, min(g), mean(g), max(g), sd(g)))

  cat("\n Run summary(fit) for full report.\n")
  cat("============================================================\n\n")
  invisible(NULL)
}

#' Compute Gaussian-style training metrics: R2, RMSE, accuracy (Pearson r),
#' and bias (regression slope of y on y_hat). Mirrors
#' masbayes::compute_metrics_gaussian. AUC is computed separately by the
#' binary path because it requires the observed-scale (0/1) response.
#' @noRd
.compute_training_metrics <- function(y, y_hat) {
  stopifnot(length(y) == length(y_hat))
  if (length(y) < 2L || stats::sd(y_hat) < .Machine$double.eps) {
    return(list(
      R2       = NA_real_,
      RMSE     = sqrt(mean((y - y_hat)^2)),
      accuracy = NA_real_,
      bias     = NA_real_
    ))
  }
  resid    <- y - y_hat
  rmse     <- sqrt(mean(resid^2))
  accuracy <- as.numeric(stats::cor(y, y_hat))
  list(
    R2       = accuracy^2,
    RMSE     = rmse,
    accuracy = accuracy,
    bias     = as.numeric(stats::coef(stats::lm(y ~ y_hat))[2L])
  )
}

#' @noRd
.default_rds_path_masreml <- function() {
  file.path(getwd(), "results_masreml.Rds")
}

#' @noRd
.maybe_save_rds_masreml <- function(fit, save_rds, save_path) {
  if (!isTRUE(save_rds)) return(NULL)
  path <- if (is.null(save_path)) .default_rds_path_masreml() else save_path
  tryCatch({
    saveRDS(fit, file = path)
    path
  }, error = function(e) {
    warning(sprintf("Failed to save RDS to '%s': %s", path, e$message))
    NULL
  })
}

#' Print Cross-Validation Results
#'
#' Compact display of cross-validation results from \code{cv_masreml()},
#' showing mean accuracy, bias, and per-fold breakdown.
#'
#' @param x object of class \code{"masreml_cv"} from \code{cv_masreml()}.
#' @param ... ignored.
#' @seealso \code{\link{summary.masreml_cv}}, \code{\link{cv_masreml}}
#' @export
print.masreml_cv <- function(x, ...) {
  cat("\n── masreml CV Results ────────────────────────────────\n")
  cat(sprintf("  Folds     : %d\n", x$folds))
  cat(sprintf("  Scheme    : %s\n", x$scheme))
  cat(sprintf("  Accuracy  : %.4f (mean r)\n", x$accuracy))
  cat(sprintf("  Bias      : %.4f (mean slope)\n", x$bias))
  cat("\n  Per-fold accuracy:\n")
  for (k in seq_along(x$accuracy_fold)) {
    cat(sprintf("    Fold %-3d: r = %.4f, slope = %.4f\n",
      k, x$accuracy_fold[k], x$bias_fold[k]))
  }
  cat("──────────────────────────────────────────────────────\n\n")
  invisible(x)
}

#' Summarize Cross-Validation Results
#'
#' Detailed summary of cross-validation results from \code{cv_masreml()},
#' including accuracy statistics across folds, regression slope (bias),
#' and GEBV distribution for all individuals.
#'
#' @param object object of class \code{"masreml_cv"} from
#'   \code{cv_masreml()}.
#' @param ... ignored.
#' @seealso \code{\link{print.masreml_cv}}, \code{\link{cv_masreml}}
#' @export
summary.masreml_cv <- function(object, ...) {
  x <- object
  cat("\n══ masreml Cross-Validation Summary ══════════════════\n\n")

  cat("Settings:\n")
  cat(sprintf("  %-16s: %d\n", "Folds", x$folds))
  cat(sprintf("  %-16s: %s\n", "Scheme", x$scheme))

  cat("\nPrediction Accuracy (r):\n")
  cat(sprintf("  %-16s: %.4f\n", "Mean", x$accuracy))
  cat(sprintf("  %-16s: %.4f\n", "SD", sd(x$accuracy_fold, na.rm = TRUE)))
  cat(sprintf("  %-16s: %.4f\n", "Min", min(x$accuracy_fold, na.rm = TRUE)))
  cat(sprintf("  %-16s: %.4f\n", "Max", max(x$accuracy_fold, na.rm = TRUE)))

  cat("\nRegression Slope (bias, expect = 1):\n")
  cat(sprintf("  %-16s: %.4f\n", "Mean", x$bias))
  cat(sprintf("  %-16s: %.4f\n", "SD", sd(x$bias_fold, na.rm = TRUE)))

  cat("\nPer-fold Results:\n")
  fold_df <- data.frame(
    Fold     = seq_along(x$accuracy_fold),
    Accuracy = round(x$accuracy_fold, 4),
    Slope    = round(x$bias_fold, 4),
    row.names = NULL
  )
  print(fold_df, row.names = FALSE)

  cat("\nGEBV Summary (all individuals):\n")
  g <- x$gebv_all
  cat(sprintf(
    "  Min=%.4f  Mean=%.4f  Max=%.4f  SD=%.4f\n",
    min(g), mean(g), max(g), sd(g)
  ))

  # Binary trait metrics (jika ada)
  if (!is.null(x$binary)) {
    cat("\nBinary Trait (GLMM Laplace):\n")
    cat(sprintf("  %-16s: %s\n",   "Link",          x$binary$link))
    cat(sprintf("  %-16s: %.4f\n", "Prevalence",    x$binary$prevalence))
    cat(sprintf("  %-16s: %.4f\n", "AUC",           x$binary$auc))
    cat(sprintf("  %-16s: %.4f\n", "h2 (liability)", x$binary$h2_liability))
    cat(sprintf("  %-16s: %.4f\n", "h2 (observed)", x$binary$h2_observed))
    cat(sprintf("  %-16s: %d\n",   "IRLS iter",     x$binary$n_irls))
    cat(sprintf("  %-16s: %s\n",   "IRLS converged", x$binary$irls_converged))

    cat("\nFitted Probabilities:\n")
    fp <- x$binary$fitted
    cat(sprintf(
      "  Min=%.4f  Mean=%.4f  Max=%.4f  SD=%.4f\n",
      min(fp), mean(fp), max(fp), sd(fp)
    ))
  }

  cat("\n══════════════════════════════════════════════════════\n\n")
  invisible(x)
}