#' Print method for masreml objects
#'
#' @param x object of class \code{"masreml"}
#' @param ... ignored
#' @export
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

  # Binary info (jika ada)
  if (!is.null(x$binary)) {
    cat(sprintf("\n  Trait     : binary (%s link)\n", x$binary$link))
    cat(sprintf("  AUC       : %.4f\n", x$binary$auc))
    cat(sprintf("  h2 (obs)  : %.4f\n", x$binary$h2_observed))
  }
  cat("──────────────────────────────────────────────────────\n\n")

  invisible(x)
}

#' Summary method for masreml objects
#'
#' @param object object of class \code{"masreml"}
#' @param ... ignored
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

  # Binary trait metrics (jika ada)
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

#' Print method for masreml_cv objects
#'
#' @param x object of class \code{"masreml_cv"}
#' @param ... ignored
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

#' Summary method for masreml_cv objects
#'
#' @param object object of class \code{"masreml_cv"}
#' @param ... ignored
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

#' Plot method for masreml objects
#'
#' @param x object of class \code{"masreml"}
#' @param type character: "gebv" (default), "varcomp", "both"
#' @param ... additional arguments passed to plot
#' @export
plot.masreml <- function(x, type = "gebv", ...) {
  type <- match.arg(type, c("gebv", "varcomp", "both"))

  if (type %in% c("gebv", "both")) {
    # GEBV distribution plot
    g <- x$total_gebv
    hist(
      g,
      main  = "GEBV Distribution",
      xlab  = "Total GEBV",
      col   = "#56B4E9",
      border = "white",
      breaks = 30,
      ...
    )
    abline(v = mean(g), col = "#E69F00", lwd = 2, lty = 2)
  }

  if (type %in% c("varcomp", "both")) {
    # Variance components bar plot
    sigma2 <- x$varcomp$sigma2
    sigma2_p <- sum(sigma2)
    prop <- sigma2 / sigma2_p

    # Okabe-Ito palette
    oi_colors <- c(
      "#E69F00", "#56B4E9", "#009E73",
      "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    )

    barplot(
      prop,
      names.arg = names(prop),
      main      = "Variance Components (proportion)",
      ylab      = "Proportion of total variance",
      col       = oi_colors[seq_along(prop)],
      border    = "white",
      ylim      = c(0, 1),
      ...
    )
    abline(h = 0, col = "grey50")
  }

  invisible(x)
}

#' Plot method for masreml_cv objects
#'
#' @param x object of class \code{"masreml_cv"}
#' @param ... additional arguments passed to plot
#' @export
plot.masreml_cv <- function(x, ...) {
  
  # Per-fold accuracy barplot
  acc <- x$accuracy_fold
  oi_colors <- c(
    "#56B4E9", "#E69F00", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
  )

  barplot(
    acc,
    names.arg = paste0("F", seq_along(acc)),
    main      = sprintf("CV Accuracy per Fold (mean = %.4f)", x$accuracy),
    ylab      = "Prediction accuracy (r)",
    col       = oi_colors[seq_along(acc) %% length(oi_colors) + 1],
    border    = "white",
    ylim      = c(0, 1),
    ...
  )
  abline(h = x$accuracy, col = "#D55E00", lwd = 2, lty = 2)

  invisible(x)
}