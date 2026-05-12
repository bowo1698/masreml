# ============================================================
# Pedigree builder (R-side wrapper)
# ============================================================

#' Build Pedigree-Based Additive Relationship Matrix
#'
#' Constructs the numerator relationship matrix (A) from pedigree data
#' following Henderson (1976). The A matrix captures additive genetic
#' relationships based on known ancestry, and can be used in
#' \code{masreml()} for pedigree-based BLUP (PBLUP) or combined with
#' genomic relationship matrices.
#'
#' @param pedigree data.frame with columns \code{id}, \code{sire},
#'   and \code{dam}. Rows must be ordered such that parents appear
#'   before offspring. IDs must be integers or convertible to integer
#'   (1-based). Use 0 for unknown parents. Example format:
#'   \preformatted{
#'   id sire dam
#'    1    0   0
#'    2    0   0
#'    3    1   2
#'    4    1   2
#'    5    3   4
#'   }
#'
#' @return numeric matrix (n x n) of additive relationships. Diagonal
#'   elements equal 1 + inbreeding coefficient. Off-diagonal elements
#'   equal twice the coefficient of kinship between pairs.
#'
#' @seealso \code{\link{build_G_snp}}, \code{\link{masreml}}
#'
#' @references
#' Henderson (1976) A simple method for computing the inverse of a
#' numerator relationship matrix. \emph{Biometrics} 32:69-83.
#'
#' @examples
#' \dontrun{
#' # build_A_ped() expects integer indices: id = 1..n, sire/dam = integers
#' # into id (0 = unknown founder). The bundled pedigree uses character IDs
#' # with NA for founder parents, so we convert before calling.
#' d      <- load_data("small")
#' ped    <- d$pedigree
#' id_map <- setNames(seq_along(ped$id), ped$id)
#' ped_int <- data.frame(
#'   id   = id_map[ped$id],
#'   sire = ifelse(is.na(ped$sire), 0L, id_map[ped$sire]),
#'   dam  = ifelse(is.na(ped$dam),  0L, id_map[ped$dam])
#' )
#' A <- build_A_ped(ped_int)
#' dim(A)
#' # Diagonal ~ 1 + inbreeding coefficient
#' round(summary(diag(A)), 3)
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

#' Build SNP Additive Genomic Relationship Matrix
#'
#' Constructs the SNP additive genomic relationship matrix (G) following
#' VanRaden (2008) Method 1. The resulting matrix captures additive genetic
#' relationships among individuals based on SNP marker data, and can be
#' used directly in \code{masreml()} or \code{gwablup()} via the \code{G}
#' argument.
#'
#' @details
#' \strong{SNP additive GRM (VanRaden 2008).}
#' The centred genotype matrix \eqn{W} is first constructed by:
#' \deqn{W_{ij} = X_{ij} - 2p_j}
#' where \eqn{X_{ij} \in \{0,1,2\}} is the allele dosage of individual
#' \eqn{i} at SNP \eqn{j} and \eqn{p_j} is the allele frequency.
#' The additive GRM is then:
#' \deqn{G = \frac{WW^\top}{2\sum_j p_j(1-p_j)}}
#' The denominator \eqn{2\sum_j p_j(1-p_j)} scales \eqn{G} so that
#' diagonal elements average approximately 1 (i.e., similar to the
#' pedigree-based numerator relationship matrix).
#'
#' @param W numeric matrix (n x m) of raw genotype codes, where n is the
#'   number of individuals and m is the number of SNP markers. Values must
#'   be 0, 1, or 2 representing the number of copies of the reference
#'   allele. Row names are used as individual IDs. Example format:
#'   \preformatted{
#'        SNP1 SNP2 SNP3
#'   ind1    0    1    2
#'   ind2    1    0    1
#'   ind3    2    1    0
#'   }
#'
#' @return numeric matrix (n x n) of genomic relationships. Diagonal
#'   elements approximate 1 + inbreeding coefficient. Off-diagonal
#'   elements represent genomic relationships between pairs of individuals.
#'
#' @seealso \code{\link{build_G_mh}}, \code{\link{build_D_snp}},
#'   \code{\link{masreml}}
#'
#' @references
#' VanRaden, P. M. (2008) Efficient methods to compute genomic predictions.
#' \emph{J. Dairy Sci.} 91:4414-4423. \doi{10.3168/jds.2007-0980}
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' G <- build_G_snp(d$snp)
#'
#' # Diagonal should be ~1 + inbreeding coefficient
#' round(summary(diag(G)), 3)
#'
#' # Train-only G with leakage-safe full-set G for prediction
#' G_train <- build_G_snp(d$snp[d$train_idx, ])
#' G_full  <- build_G_snp(d$snp, ref_W = d$snp[d$train_idx, ])
#' }
#'
#' @export
build_G_snp <- function(W, ref_W = NULL) {
  W <- .validate_snp_matrix(W, "snp_add")
  allele_freq <- if (!is.null(ref_W)) {
    ref_W <- .validate_snp_matrix(ref_W, "snp_add_ref")
    colMeans(ref_W) / 2
  } else {
    NULL
  }
  G <- r_build_g_snp_add(W, weights = NULL, allele_freq = allele_freq)
  rownames(G) <- rownames(W)
  colnames(G) <- rownames(W)
  G
}

#' Build SNP Dominance Relationship Matrix
#'
#' Constructs the SNP dominance relationship matrix (D) following
#' Da (2015). Captures non-additive (dominance) genetic
#' relationships among individuals. Can be used alongside the additive
#' G matrix in \code{masreml()} to partition genetic variance into
#' additive and dominance components.
#'
#' @details
#' \strong{SNP dominance GRM (Da, 2015).}
#' For individual \eqn{i} at SNP \eqn{j} with allele frequency \eqn{p_j}
#' and \eqn{q_j = 1 - p_j}, the dominance-coded value is:
#' \describe{
#'   \item{Genotype \eqn{aa} (dosage = 0):}{\eqn{W_{D,ij} = -2p_j^2}}
#'   \item{Genotype \eqn{Aa} (dosage = 1):}{\eqn{W_{D,ij} = 2p_j q_j}}
#'   \item{Genotype \eqn{AA} (dosage = 2):}{\eqn{W_{D,ij} = -2q_j^2}}
#' }
#' The dominance GRM is then:
#' \deqn{D = \frac{W_D W_D^\top}{\sum_j (2p_j q_j)^2}}
#'
#' @param W numeric matrix (n x m) of raw genotype codes, where n is the
#'   number of individuals and m is the number of SNP markers. Values must
#'   be 0, 1, or 2. Same format as \code{\link{build_G_snp}}.
#'
#' @return numeric matrix (n x n) of dominance relationships.
#'
#' @seealso \code{\link{build_G_snp}}, \code{\link{masreml}}
#'
#' @references
#' Da, Y. (2015) Multi-allelic haplotype model based on genetic partition for
#' genomic prediction and variance component estimation using SNP markers.
#' \emph{BMC Genetics} 16:144. \doi{10.1186/s12863-015-0301-1}
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' D <- build_D_snp(d$snp)
#' dim(D)
#' # Off-diagonal mean ~ 0 under random mating; diagonal carries the
#' # dominance "self" contributions.
#' round(summary(diag(D)), 3)
#' }
#'
#' @export
build_D_snp <- function(W) {
  W <- .validate_snp_matrix(W, "snp_dom")
  G <- r_build_g_snp_dom(W)
  rownames(G) <- rownames(W)
  colnames(G) <- rownames(W)
  G
}

#' Build Multi-allelic Additive Genomic Relationship Matrix
#'
#' Constructs the multi-allelic additive genomic relationship matrix
#' following Da (2015). Uses multi-allelic haplotype block coding
#' (\eqn{W_{ah}}) that captures haplotype diversity beyond what bi-allelic
#' SNPs can represent. The resulting matrix can be used in \code{masreml()}
#' or \code{gwablup()} via the \code{G} argument.
#'
#' @details
#' \strong{Multi-allelic additive GRM (Da, 2015).}
#' For individual \eqn{i} with phased alleles \eqn{(A_i, A_j)} at haplotype
#' block \eqn{h}, the column corresponding to non-baseline allele \eqn{k}
#' (with population frequency \eqn{p_k}) is coded as:
#' \describe{
#'   \item{Homozygous (\eqn{A_i = A_j = k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = -2(1 - p_k)}}
#'   \item{Heterozygous (\eqn{A_i = k} or \eqn{A_j = k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = -(1 - 2p_k)}}
#'   \item{Absent (\eqn{A_i \neq k} and \eqn{A_j \neq k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = 2p_k}}
#' }
#' Rare alleles receive larger absolute deviations; common alleles receive
#' smaller deviations. The population column-mean is zero by construction,
#' ensuring the additive effects sum to zero across individuals.
#' 
#' The additive multi-allelic GRM is then:
#' \deqn{A_{gh} = \frac{W_{ah} W_{ah}^\top}{k_{ah}}, \quad
#'   k_{ah} = \frac{\mathrm{tr}(W_{ah} W_{ah}^\top)}{n}}
#' where \eqn{n} is the number of individuals and \eqn{k_{ah}} scales the
#' matrix so that diagonal elements average approximately 1.
#'
#' @param mh_list list of data.frames (one per chromosome) or integer matrix
#'   (n x n_blocks*2). Two input modes are supported:
#'   \itemize{
#'     \item \strong{List of data.frames}: First column = individual IDs,
#'       remaining columns = paired haplotype allele codes per block
#'       (strand1_block1, strand2_block1, ...). Allele codes must be
#'       0-based sequential integers.
#'     \item \strong{Haplotype matrix}: integer matrix (n x n_blocks*2),
#'       columns alternate strand1/strand2 per block. Allele codes can be
#'       any integer (sparse, non-sequential) — re-encoded internally to
#'       sequential 0-based using training reference (\code{ref_mh}).
#'   }
#'   Example matrix format (50 blocks → 100 columns):
#'   \preformatted{
#'   hap_block_all  # n x (n_blocks*2), cols: s1_b1, s2_b1, s1_b2, s2_b2, ...
#'   }
#'
#' @param ref_mh reference haplotype data for training-based allele frequencies.
#'   Same format as \code{mh_list}. If provided, allele frequencies and
#'   dropped allele (most frequent) are estimated from \code{ref_mh} only,
#'   avoiding data leakage from test individuals. If NULL, frequencies are
#'   computed from all individuals in \code{mh_list}.
#' @param ids character vector of individual IDs for alignment. Required
#'   when \code{mh_list} is a haplotype matrix without rownames.
#'
#' @return numeric matrix (n x n) of microhaplotype-based genomic
#'   relationships. Same interpretation as SNP additive G matrix but
#'   based on haplotype block diversity.
#'
#' @seealso \code{\link{build_G_snp}}, \code{\link{masreml}},
#'   \code{\link{run_gwas}}
#'
#' @references
#' Da, Y. (2015) Multi-allelic haplotype model based on genetic partition for
#' genomic prediction and variance component estimation using SNP markers.
#' \emph{BMC Genetics} 16:144. \doi{10.1186/s12863-015-0301-1}
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' # d$mh is a haplotype matrix consumable directly via auto-detection.
#' G_mh <- build_G_mh(d$mh)
#' dim(G_mh)
#' # Diagonal ~ 1 + inbreeding coefficient
#' round(summary(diag(G_mh)), 3)
#'
#' # Train-only G with leakage-safe full-set G for prediction
#' G_mh_full <- build_G_mh(
#'   mh_list = d$mh,
#'   ref_mh  = d$mh[d$train_idx, ],
#'   ids     = d$pheno$id
#' )
#' dim(G_mh_full)
#' }
#'
#' @export
build_G_mh <- function(mh_list, ref_mh = NULL, ids = NULL) {
  .build_g_mh(mh_list, ids = ids, ref_mh = ref_mh)
}

# ============================================================
# Accuracy and diagnostics
# ============================================================

#' Compute Genomic Prediction Accuracy and Bias
#'
#' Evaluates prediction accuracy and bias of genomic estimated breeding
#' values (GEBV) against observed phenotypes. Returns three metrics
#' commonly used.
#'
#' @param gebv numeric vector of genomic estimated breeding values,
#'   typically \code{fit$total_gebv} from a \code{masreml()} or
#'   \code{gwablup()} object.
#' @param y numeric vector of observed phenotypes (same length as
#'   \code{gebv}).
#' @param h2 numeric, heritability estimate for computing the
#'   marker-genetic correlation (r_MG). If NULL, r_MG is returned
#'   as NA. Typically \code{fit$varcomp$h2} from the fitted model.
#'
#' @return data.frame with columns:
#'   \itemize{
#'     \item \code{r}: Pearson correlation between GEBV and phenotype.
#'       Ranges from -1 to 1; higher is better.
#'     \item \code{slope}: regression coefficient of phenotype on GEBV
#'       (\code{lm(y ~ gebv)}). Value of 1 indicates unbiased prediction;
#'       < 1 indicates inflation; > 1 indicates deflation.
#'     \item \code{r_MG}: marker-genetic correlation, computed as
#'       \code{r / sqrt(h2)}. Approximates correlation between GEBV
#'       and true breeding value.
#'   }
#'
#' @seealso \code{\link{masreml}}, \code{\link{gwablup}},
#'   \code{\link{cv_masreml}}
#'
#' @examples
#' \dontrun{
#' d   <- load_data("small")
#' y   <- d$pheno$y_cont_qtl_snp; names(y) <- d$pheno$id
#' fit <- masreml(y, markers = list(snp_add = d$snp))
#'
#' acc <- compute_accuracy(
#'   gebv = fit$total_gebv,
#'   y    = y,
#'   h2   = fit$varcomp$h2["snp_add"]
#' )
#' print(acc)
#' }
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
#' @noRd
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
#' @noRd
jitter_diagonal <- function(G, eps = 1e-6) {
  diag(G) <- diag(G) + eps
  G
}

# ============================================================
# Format converters
# ============================================================

#' Convert masreml Result to Data Frame
#'
#' Extracts genomic estimated breeding values (GEBV) from a fitted
#' \code{masreml} object into a data.frame, suitable for export or
#' downstream analysis.
#'
#' @param x object of class \code{"masreml"} from \code{masreml()} or
#'   \code{gwablup()}.
#' @param include_components logical, if \code{TRUE} includes per-component
#'   GEBV columns (e.g. \code{gebv_snp_add}, \code{gebv_mh_add}) in
#'   addition to \code{total_gebv}. Default \code{FALSE}.
#' @param ... additional arguments (ignored).
#'
#' @return data.frame with columns:
#'   \itemize{
#'     \item \code{id}: individual IDs
#'     \item \code{total_gebv}: total GEBV (sum across all components)
#'     \item \code{gebv_*}: per-component GEBV columns, if
#'       \code{include_components = TRUE}
#'   }
#'
#' @seealso \code{\link{masreml}}, \code{\link{gwablup}}
#'
#' @examples
#' \dontrun{
#' d   <- load_data("small")
#' y   <- d$pheno$y_cont_qtl_snp; names(y) <- d$pheno$id
#' fit <- masreml(y, markers = list(snp_add = d$snp))
#' df  <- as.data.frame(fit)
#' head(df)
#' }
#'
#' @noRd
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

#' Extract Variance Components
#'
#' Extracts variance component estimates from a fitted \code{masreml}
#' object into a tidy data.frame, showing heritability and proportion
#' of total variance explained by each component.
#'
#' @param x object of class \code{"masreml"} from \code{masreml()} or
#'   \code{gwablup()}.
#'
#' @return data.frame with columns:
#'   \itemize{
#'     \item \code{Component}: name of variance component
#'       (e.g. \code{snp_add}, \code{mh_add}, \code{residual})
#'     \item \code{Sigma2}: estimated variance component
#'     \item \code{H2}: heritability (\code{sigma2 / sigma2_total});
#'       \code{NA} for residual component
#'     \item \code{Proportion}: proportion of total phenotypic variance
#'   }
#'
#' @seealso \code{\link{masreml}}, \code{\link{summary.masreml}}
#'
#' @examples
#' \dontrun{
#' d   <- load_data("small")
#' y   <- d$pheno$y_cont_qtl_snp; names(y) <- d$pheno$id
#' fit <- masreml(y, markers = list(snp_add = d$snp))
#' varcomp(fit)
#' }
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

#' Set Global Number of Threads
#'
#' Sets the number of parallel threads used by masreml for matrix
#' computations. By default, masreml uses all available physical cores.
#' Use this function to limit CPU usage, particularly on shared HPC
#' systems where oversubscription should be avoided.
#'
#' @param n integer, number of threads to use. Must be >= 1.
#'   Values less than 1 are silently coerced to 1.
#'
#' @return invisibly returns \code{n}.
#'
#' @seealso \code{\link{masreml}}
#'
#' @examples
#' \dontrun{
#' # Use 4 threads
#' set_masreml_threads(4)
#'
#' # Use single thread (useful for debugging)
#' set_masreml_threads(1)
#'
#' # Reset to all available cores
#' set_masreml_threads(parallel::detectCores(logical = FALSE))
#' }
#'
#' @noRd
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