#' Utility functions for masreml
#' @keywords internal

# ============================================================
# Pedigree builder (R-side wrapper)
# ============================================================

#' Build pedigree A matrix (Henderson 1976)
#'
#' @param pedigree data.frame with columns:
#'   \code{id}, \code{sire}, \code{dam}
#'   IDs must be integers or convertible to integer (1-based).
#'   Use 0 for unknown parents.
#' @return numeric matrix (n x n), additive relationship matrix A
#'
#' @examples
#' \dontrun{
#' ped <- data.frame(
#'   id   = 1:5,
#'   sire = c(0, 0, 1, 1, 3),
#'   dam  = c(0, 0, 2, 2, 4)
#' )
#' A <- build_A_ped(ped)
#' }
#'
#' @export
build_A_ped <- function(pedigree) {
  required_cols <- c("id", "sire", "dam")
  if (!all(required_cols %in% names(pedigree))) {
    stop(sprintf(
      "Pedigree must have columns: %s",
      paste(required_cols, collapse = ", ")
    ))
  }

  n    <- nrow(pedigree)
  sire <- as.integer(pedigree$sire)
  dam  <- as.integer(pedigree$dam)
  ids  <- as.character(pedigree$id)

  # Validate: sire/dam indices must be 0 to n
  if (any(sire < 0 | sire > n) || any(dam < 0 | dam > n)) {
    stop("Sire/dam indices must be between 0 (unknown) and n.")
  }

  A <- r_build_a_ped(sire = sire, dam = dam, n = as.integer(n))
  rownames(A) <- ids
  colnames(A) <- ids
  A
}

# ============================================================
# G matrix builders (public API)
# ============================================================

#' Build SNP additive G matrix (VanRaden 2008)
#'
#' @param W numeric matrix (n x m), raw genotype 0/1/2
#' @return numeric matrix (n x n), SNP additive G
#'
#' @export
build_G_snp <- function(W) {
  W <- .validate_snp_matrix(W, "snp_add")
  G <- r_build_g_snp_add(W)
  rownames(G) <- rownames(W)
  colnames(G) <- rownames(W)
  G
}

#' Build SNP dominance D matrix (Da et al. 2014)
#'
#' @param W numeric matrix (n x m), raw genotype 0/1/2
#' @return numeric matrix (n x n), SNP dominance D
#'
#' @export
build_D_snp <- function(W) {
  W <- .validate_snp_matrix(W, "snp_dom")
  G <- r_build_g_snp_dom(W)
  rownames(G) <- rownames(W)
  colnames(G) <- rownames(W)
  G
}

#' Build MH additive Agh matrix (Da 2015)
#'
#' @param mh_list list of data.frames, one per chromosome.
#'   Each data.frame: first col = ID, then paired strand columns
#'   per haplotype block (strand1, strand2, strand1, strand2, ...)
#' @return numeric matrix (n x n), MH additive Agh
#'
#' @export
build_G_mh <- function(mh_list) {
  .build_g_mh(mh_list)
}

# ============================================================
# Accuracy and diagnostics
# ============================================================

#' Compute prediction accuracy and bias
#'
#' @param gebv numeric vector, estimated breeding values
#' @param y numeric vector, observed phenotypes
#' @param h2 numeric, heritability estimate (for r_MG calculation)
#' @return data.frame with accuracy (r), bias (slope), r_MG
#'
#' @export
compute_accuracy <- function(gebv, y, h2 = NULL) {
  if (length(gebv) != length(y)) {
    stop("gebv and y must have the same length.")
  }

  # r = cor(GEBV, y)
  r <- cor(gebv, y, use = "complete.obs")

  # Regression slope: lm(observed ~ predicted)
  fit   <- lm(y ~ gebv)
  slope <- coef(fit)["gebv"]

  # r_MG = r / sqrt(h2) if h2 provided
  r_MG <- if (!is.null(h2) && h2 > 0) r / sqrt(h2) else NA_real_

  data.frame(
    r     = round(r, 4),
    slope = round(slope, 4),
    r_MG  = round(r_MG, 4),
    row.names = NULL
  )
}

#' Check if matrix is positive definite
#'
#' @param G numeric matrix
#' @param label character, name for error messages
#' @return logical
#'
#' @export
is_positive_definite <- function(G, label = "G") {
  tryCatch({
    chol(G)
    TRUE
  }, error = function(e) {
    message(sprintf("[%s] Not positive definite: %s", label, e$message))
    FALSE
  })
}

#' Jitter G matrix diagonal to ensure positive definiteness
#'
#' @param G numeric matrix
#' @param eps numeric, small value added to diagonal (default 1e-6)
#' @return numeric matrix
#'
#' @export
jitter_diagonal <- function(G, eps = 1e-6) {
  diag(G) <- diag(G) + eps
  G
}

# ============================================================
# Format converters
# ============================================================

#' Convert masreml result to data.frame of GEBVs
#'
#' @param x object of class \code{"masreml"}
#' @param include_components logical, include per-component EBV columns
#' @return data.frame
#'
#' @export
as.data.frame.masreml <- function(x, include_components = FALSE, ...) {
  df <- data.frame(
    id         = names(x$total_gebv),
    total_gebv = x$total_gebv,
    row.names  = NULL,
    stringsAsFactors = FALSE
  )

  if (include_components) {
    for (nm in names(x$gebv)) {
      df[[paste0("gebv_", nm)]] <- x$gebv[[nm]]
    }
  }

  df
}

#' Extract variance components as data.frame
#'
#' @param x object of class \code{"masreml"}
#' @return data.frame with Component, Sigma2, H2, Proportion
#'
#' @export
varcomp <- function(x) {
  UseMethod("varcomp")
}

#' @export
varcomp.masreml <- function(x) {
  sigma2   <- x$varcomp$sigma2
  sigma2_p <- sum(sigma2)
  h2_full  <- c(x$varcomp$h2, residual = NA_real_)

  data.frame(
    Component  = names(sigma2),
    Sigma2     = round(sigma2, 6),
    H2         = round(h2_full, 4),
    Proportion = round(sigma2 / sigma2_p, 4),
    row.names  = NULL
  )
}

# ============================================================
# Package-level settings
# ============================================================

#' Set global number of threads for masreml
#'
#' @param n integer, number of threads
#' @export
set_masreml_threads <- function(n) {
  options(masreml.threads = as.integer(max(1L, n)))
  invisible(n)
}

#' Get current thread count setting
#' @noRd
.get_masreml_threads <- function() {
  opt <- getOption("masreml.threads", default = NULL)
  if (is.null(opt)) {
    parallel::detectCores(logical = FALSE)
  } else {
    opt
  }
}

# ============================================================
# Link functions for binary trait (GLMM)
# ============================================================

#' @noRd
.link_functions <- function(link) {
  switch(link,
    "logit" = list(
      # g(mu) = log(mu/(1-mu))
      fun   = function(mu) log(mu / (1 - mu)),
      # g^-1(eta) = 1/(1+exp(-eta))
      inv   = function(eta) 1 / (1 + exp(-eta)),
      # dmu/deta = mu*(1-mu)  [variance function for IRLS weight]
      deriv = function(mu) mu * (1 - mu)
    ),
    "probit" = list(
      # g(mu) = Phi^-1(mu)
      fun   = function(mu) qnorm(mu),
      # g^-1(eta) = Phi(eta)
      inv   = function(eta) pnorm(eta),
      # dmu/deta = phi(eta) = phi(Phi^-1(mu))
      deriv = function(mu) dnorm(qnorm(mu))
    ),
    stop(sprintf("Unknown link function: '%s'. Use 'logit' or 'probit'.", link))
  )
}

#' @noRd
.h2_liability_to_observed <- function(h2_liability, prevalence, link = "logit") {
  # Dempster & Falconer (1950) / Robertson & Lerner (1949)
  # Transforms h2 from liability scale to observed (0/1) scale
  #
  # h2_obs = h2_liab * z^2 / (p * q)
  # where z = ordinate of standard normal at truncation point t
  #       p = prevalence, q = 1-p

  p <- prevalence
  q <- 1 - p

  if (link == "probit") {
    # Truncation point
    t <- qnorm(1 - p)
    z <- dnorm(t)
    h2_obs <- h2_liability * z^2 / (p * q)
  } else {
    # Logit scale: use approximate transformation
    # Via probit-logit rescaling (Lee & Nelder 2001 approximation)
    # sigma2_logit = pi^2/3, sigma2_probit = 1
    # h2_probit ≈ h2_logit * (pi^2/3) / (pi^2/3 + 1) * correction
    h2_probit <- h2_liability * (pi^2 / 3) / (pi^2 / 3 + 1)
    t <- qnorm(1 - p)
    z <- dnorm(t)
    h2_obs <- h2_probit * z^2 / (p * q)
  }

  # Clamp to [0, 1]
  pmax(pmin(h2_obs, 1), 0)
}

#' @noRd
.compute_auc <- function(y, fitted_prob) {
  # Wilcoxon-Mann-Whitney AUC
  # AUC = P(score_positive > score_negative)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)

  if (n1 == 0 || n0 == 0) return(NA_real_)

  scores_1 <- fitted_prob[y == 1]
  scores_0 <- fitted_prob[y == 0]

  # Count concordant pairs
  auc <- mean(outer(scores_1, scores_0, ">")) +
         0.5 * mean(outer(scores_1, scores_0, "=="))

  round(auc, 4)
}