#' Predict GEBV for New Individuals Using a Fitted masreml Model
#'
#' Computes genomic estimated breeding values (GEBV) for test individuals
#' using the relationship between test and training sets. Two modes are
#' supported: pre-built G matrix (full or cross-block) and raw marker data.
#'
#' @param object fitted \code{masreml} object from \code{masreml()}
#' @param G_full numeric matrix (n_total x n_total), pre-built genomic
#'   relationship matrix including both training and test individuals.
#'   Row/column names must include all training IDs and test IDs.
#'   Either \code{G_full} or \code{markers_new} must be provided.
#'   For multi-component models, provide a named list of G matrices,
#'   one per component (names must match \code{names(object$gebv)}).
#' @param markers_new list of raw marker data for test individuals only.
#'   Used to compute cross-G matrix \code{G[test, train]} directly
#'   from marker data. Supported elements:
#'   \itemize{
#'     \item \code{snp_add}: integer matrix (n_test x m), raw genotype 0/1/2
#'     \item \code{mh_add}: list of data.frames in same format as
#'       \code{build_G_mh()} input (one per chromosome, first col = ID)
#'   }
#'   Allele frequencies and scaling factors are computed from training
#'   markers stored via \code{markers_train} argument.
#' @param markers_train list of raw marker data for training individuals,
#'   required when \code{markers_new} is provided. Same format as
#'   \code{markers_new}. Used to compute consistent allele frequencies
#'   and scaling (k) for cross-G computation.
#' @param train_ids character vector of training individual IDs, in the
#'   same order as used when fitting \code{object}. Required when using
#'   \code{G_full} mode to subset the training block.
#' @param test_ids character vector of test individual IDs. Required for
#'   \code{G_full} mode. Inferred from \code{markers_new} row names if
#'   not provided.
#' @param link character, link function for binary trait prediction:
#'   \code{"logit"} (default) or \code{"probit"}. Inherited from
#'   \code{object$binary$link} if available.
#' @param jitter numeric, small value added to diagonal of
#'   \code{G[train,train]} for numerical stability (default 1e-6).
#' @param ... ignored
#'
#' @return list of class \code{"masreml_pred"} with elements:
#'   \itemize{
#'     \item \code{total_gebv}: named numeric vector of total GEBV
#'       for test individuals (sum across components)
#'     \item \code{gebv}: named list of per-component GEBV vectors
#'     \item \code{fitted}: fitted probabilities P(y=1) for binary trait,
#'       NULL for continuous
#'     \item \code{train_ids}: character vector of training IDs used
#'     \item \code{test_ids}: character vector of test IDs predicted
#'     \item \code{n_train}: number of training individuals
#'     \item \code{n_test}: number of test individuals
#'   }
#'
#' @details
#' The prediction formula per component k is:
#' \deqn{\hat{u}_{test,k} = G_{te,tr,k} \cdot (G_{tr,tr,k} + \lambda_k I)^{-1} \cdot \hat{u}_{train,k}}
#' where \eqn{\lambda_k = \sigma^2_e / \sigma^2_{g,k}}.
#'
#' For \code{markers_new} mode (SNP), the cross-G is:
#' \deqn{G_{te,tr} = \frac{W_{te} W_{tr}'}{k}}
#' where \eqn{W} matrices are centered using training allele frequencies
#' and \eqn{k = \sum 2p_j(1-p_j)} computed from training data.
#'
#' For binary trait, prediction is on the liability scale. Fitted
#' probabilities are obtained by applying the inverse link function
#' to \eqn{\eta_{test} = \mu_{fixed} + \hat{u}_{test}}.
#'
#' @seealso \code{\link{masreml}}, \code{\link{build_G_snp}},
#'   \code{\link{build_G_mh}}
#'
#' @examples
#' \dontrun{
#' # ── Mode A: G_full pre-built ──────────────────────────────
#' G_full <- build_G_snp(W_all)   # n_total x n_total
#' fit    <- masreml(y_train, G = list(snp_add = G_train))
#'
#' pred <- predict(fit,
#'   G_full     = list(snp_add = G_full),
#'   train_ids  = rownames(W_train),
#'   test_ids   = rownames(W_test))
#'
#' cor(pred$total_gebv, tbv_test)   # accuracy
#'
#' # ── Mode B: markers_new ───────────────────────────────────
#' fit  <- masreml(y_train, markers = list(snp_add = W_train))
#'
#' pred <- predict(fit,
#'   markers_new   = list(snp_add = W_test),
#'   markers_train = list(snp_add = W_train))
#'
#' cor(pred$total_gebv, tbv_test)
#' }
#'
#' @export
predict.masreml <- function(
    object,
    G_full        = NULL,
    markers_new   = NULL,
    markers_train = NULL,
    train_ids     = NULL,
    test_ids      = NULL,
    link          = NULL,
    jitter        = 1e-6,
    ...
) {
  # ── Inherit link from binary fit ──────────────────────────
  if (is.null(link)) {
    link <- if (!is.null(object$binary)) object$binary$link else "logit"
  }
  link <- match.arg(link, c("logit", "probit"))

  is_binary <- !is.null(object$binary)

  # ── Extract training info from fitted object ──────────────
  u_train_list  <- object$gebv             # named list per component
  g_names       <- names(u_train_list)
  sigma2        <- object$varcomp$sigma2   # named: g1,...,gk, residual
  sigma2_e      <- sigma2["residual"]
  sigma2_g_list <- sigma2[g_names]        # per component

  mu_fixed <- object$fixed_effects[1]     # intercept (mu_hat)

  # ── Determine mode ────────────────────────────────────────
  use_G_full      <- !is.null(G_full)
  use_markers_new <- !is.null(markers_new)

  if (!use_G_full && !use_markers_new) {
    stop("Provide either 'G_full' or 'markers_new' + 'markers_train'.")
  }
  if (use_G_full && use_markers_new) {
    stop("Provide only one of 'G_full' or 'markers_new', not both.")
  }
  if (use_markers_new && is.null(markers_train)) {
    stop("'markers_train' is required when using 'markers_new' mode.")
  }

  # ── Validate train_ids ────────────────────────────────────
  if (use_G_full && is.null(train_ids)) {
    # Try to infer from fitted object GEBV names
    train_ids <- names(u_train_list[[1]])
    if (is.null(train_ids)) {
      stop("'train_ids' must be provided when using G_full mode.")
    }
  }

  # ── Mode A: G_full pre-built ──────────────────────────────
  if (use_G_full) {

    # Normalise: accept single matrix or named list
    if (is.matrix(G_full)) {
      # single matrix: replicate for all components
      G_full <- setNames(
        rep(list(G_full), length(g_names)),
        g_names
      )
    }
    if (!is.list(G_full)) {
      stop("'G_full' must be a matrix or named list of matrices.")
    }
    if (!all(g_names %in% names(G_full))) {
      stop(sprintf(
        "'G_full' must have entries for all components: %s",
        paste(g_names, collapse = ", ")
      ))
    }

    # Infer test_ids from G_full if not provided
    if (is.null(test_ids)) {
      all_ids  <- rownames(G_full[[g_names[1]]])
      test_ids <- setdiff(all_ids, train_ids)
      if (length(test_ids) == 0) {
        stop("Cannot infer test_ids from G_full: all IDs match train_ids.")
      }
    }

    # Validate IDs present in G_full
    for (nm in g_names) {
      g_ids <- rownames(G_full[[nm]])
      if (is.null(g_ids)) {
        stop(sprintf("G_full[['%s']] has no rownames.", nm))
      }
      missing_tr <- train_ids[!train_ids %in% g_ids]
      missing_te <- test_ids[!test_ids %in% g_ids]
      if (length(missing_tr) > 0) {
        stop(sprintf(
          "G_full[['%s']]: missing train_ids: %s",
          nm, paste(head(missing_tr, 5), collapse = ", ")
        ))
      }
      if (length(missing_te) > 0) {
        stop(sprintf(
          "G_full[['%s']]: missing test_ids: %s",
          nm, paste(head(missing_te, 5), collapse = ", ")
        ))
      }
    }

    # Predict per component
    gebv_test_list <- .predict_from_G_full(
      G_full        = G_full,
      g_names       = g_names,
      u_train_list  = u_train_list,
      train_ids     = train_ids,
      test_ids      = test_ids,
      sigma2_g_list = sigma2_g_list,
      sigma2_e      = sigma2_e,
      jitter        = jitter
    )
  }

  # ── Mode B: markers_new ───────────────────────────────────
  if (use_markers_new) {
    gebv_test_list <- .predict_from_markers(
      markers_new   = markers_new,
      markers_train = markers_train,
      g_names       = g_names,
      u_train_list  = u_train_list,
      sigma2_g_list = sigma2_g_list,
      sigma2_e      = sigma2_e,
      jitter        = jitter
    )
    # Infer test_ids
    test_ids <- names(gebv_test_list[[g_names[1]]])
  }

  # ── Total GEBV (sum across components) ───────────────────
  total_gebv_test <- Reduce("+", gebv_test_list)
  names(total_gebv_test) <- test_ids

  # ── Binary: fitted probabilities ─────────────────────────
  fitted_prob <- NULL
  if (is_binary) {
    lf      <- .link_functions(link)
    eta_te  <- mu_fixed + total_gebv_test
    fitted_prob <- pmax(pmin(lf$inv(eta_te), 1 - 1e-6), 1e-6)
    names(fitted_prob) <- test_ids
  }

  structure(
    list(
      total_gebv = total_gebv_test,
      gebv       = gebv_test_list,
      fitted     = fitted_prob,
      train_ids  = train_ids,
      test_ids   = test_ids,
      n_train    = length(train_ids),
      n_test     = length(test_ids)
    ),
    class = "masreml_pred"
  )
}

#' Evaluate Out-of-Sample Genomic Prediction
#'
#' Computes prediction accuracy metrics for test individuals. Designed
#' to complement \code{predict.masreml()} for out-of-sample evaluation.
#' Works with any GEBV source (masreml, masbayes, or other models).
#'
#' @param gebv numeric vector of predicted GEBV for test individuals
#' @param y numeric vector of observed phenotypes for test individuals
#' @param h2 numeric, heritability from fitted model for r_MG computation.
#'   Typically \code{fit$varcomp$h2} (single component) or a specific
#'   component. If NULL, r_MG is returned as NA.
#' @param tbv numeric vector of true breeding values (simulation only).
#'   If provided, \code{r_MG_true = cor(gebv, tbv)} is computed directly
#'   without requiring h2. If NULL, r_MG_true is NA.
#' @param fitted_prob numeric vector of fitted probabilities P(y=1) for
#'   binary trait, typically \code{pred$fitted} from \code{predict.masreml()}.
#'   If NULL, AUC is returned as NA.
#'
#' @return data.frame with columns:
#'   \itemize{
#'     \item \code{r_test_y}: cor(gebv, y) — predictive ability
#'     \item \code{r_test_g}: cor(gebv, tbv) — accuracy vs true BV, NA if tbv NULL
#'     \item \code{bias}: regression slope lm(y ~ gebv), expect 1 = unbiased
#'     \item \code{r_MG}: r_test_y / sqrt(h2), NA if h2 NULL
#'     \item \code{AUC}: area under ROC curve, NA if fitted_prob NULL
#'     \item \code{RMSE}: sqrt(mean((gebv - y)^2))
#'   }
#'
#' @seealso \code{\link{predict.masreml}}, \code{\link{compute_accuracy}}
#'
#' @examples
#' \dontrun{
#' pred <- predict(fit, G_full = list(g = G_full),
#'                 train_ids = train_ids, test_ids = test_ids)
#'
#' # Continuous trait
#' evaluate_prediction(
#'   gebv = pred$total_gebv,
#'   y    = y_test,
#'   h2   = fit$varcomp$h2["g"],
#'   tbv  = tbv_test
#' )
#'
#' # Binary trait
#' evaluate_prediction(
#'   gebv        = pred$total_gebv,
#'   y           = y_test,
#'   h2          = fit$varcomp$h2["g"],
#'   fitted_prob = pred$fitted
#' )
#' }
#'
#' @export
evaluate_prediction <- function(gebv, y, h2 = NULL, tbv = NULL,
                                fitted_prob = NULL) {
  if (length(gebv) != length(y)) {
    stop("'gebv' and 'y' must have same length.")
  }
  if (!is.null(tbv) && length(tbv) != length(gebv)) {
    stop("'tbv' must have same length as 'gebv'.")
  }
  if (!is.null(fitted_prob) && length(fitted_prob) != length(gebv)) {
    stop("'fitted_prob' must have same length as 'gebv'.")
  }

  r_test_y <- cor(gebv, y, use = "complete.obs")
  r_test_g <- if (!is.null(tbv)) cor(gebv, tbv, use = "complete.obs") else NA_real_
  bias     <- tryCatch(coef(lm(y ~ gebv))[2], error = function(e) NA_real_)
  r_MG     <- if (!is.null(h2) && !is.na(h2) && h2 > 0)
                r_test_y / sqrt(h2)
              else NA_real_
  auc      <- if (!is.null(fitted_prob)) {
              tryCatch({
                n1 <- sum(y == 1); n0 <- sum(y == 0)
                if (n1 == 0 || n0 == 0) NA_real_
                else {
                  s1 <- fitted_prob[y == 1]; s0 <- fitted_prob[y == 0]
                  mean(outer(s1, s0, ">")) + 0.5 * mean(outer(s1, s0, "=="))
                }
              }, error = function(e) NA_real_)
            } else NA_real_
  rmse     <- sqrt(mean((gebv - y)^2, na.rm = TRUE))

  data.frame(
    r_test_y = round(r_test_y, 4),
    r_test_g = round(r_test_g, 4),
    bias     = round(bias, 4),
    r_MG     = round(r_MG, 4),
    AUC      = round(auc, 4),
    RMSE     = round(rmse, 4),
    row.names = NULL
  )
}

# ============================================================
# Internal helpers
# ============================================================

#' @noRd
.predict_from_G_full <- function(
    G_full, g_names, u_train_list, train_ids, test_ids,
    sigma2_g_list, sigma2_e, jitter
) {
  lapply(setNames(g_names, g_names), function(nm) {
    G    <- G_full[[nm]]
    u_tr <- u_train_list[[nm]][train_ids]
    sg   <- sigma2_g_list[nm]
    lambda <- sigma2_e / sg

    G_tr_tr <- G[train_ids, train_ids, drop = FALSE]
    G_te_tr <- G[test_ids,  train_ids, drop = FALSE]

    # Ridge regularisation for numerical stability
    diag(G_tr_tr) <- diag(G_tr_tr) + jitter

    # Solve: (G_tr_tr + lambda*I) x = u_tr
    # => x = solve(G_tr_tr + lambda*I, u_tr)
    M <- G_tr_tr + diag(lambda, nrow(G_tr_tr))
    x <- tryCatch(
      solve(M, u_tr),
      error = function(e) {
        # Fallback: increase jitter and retry
        M2 <- G_tr_tr + diag(lambda + 1e-4, nrow(G_tr_tr))
        solve(M2, u_tr)
      }
    )

    u_te <- as.vector(G_te_tr %*% x)
    names(u_te) <- test_ids
    u_te
  })
}

#' @noRd
.predict_from_markers <- function(
    markers_new, markers_train, g_names,
    u_train_list, sigma2_g_list, sigma2_e, jitter
) {
  result <- list()

  # ── SNP additive ──────────────────────────────────────────
  if ("snp_add" %in% g_names) {
    if (is.null(markers_new$snp_add) || is.null(markers_train$snp_add)) {
      stop("'markers_new$snp_add' and 'markers_train$snp_add' required for snp_add component.")
    }

    W_tr <- markers_train$snp_add
    W_te <- markers_new$snp_add

    # Validate and convert
    if (is.data.frame(W_tr)) W_tr <- as.matrix(W_tr[, -1])
    if (is.data.frame(W_te)) W_te <- as.matrix(W_te[, -1])
    storage.mode(W_tr) <- "double"
    storage.mode(W_te) <- "double"

    if (ncol(W_tr) != ncol(W_te)) {
      stop(sprintf(
        "snp_add: markers_train has %d SNPs, markers_new has %d SNPs.",
        ncol(W_tr), ncol(W_te)
      ))
    }

    test_ids_snp <- rownames(W_te)
    if (is.null(test_ids_snp)) {
      test_ids_snp <- paste0("test_", seq_len(nrow(W_te)))
    }

    # Allele freq from training, centering both
    p_tr <- colMeans(W_tr) / 2
    k    <- 2 * sum(p_tr * (1 - p_tr))

    W_tr_c <- sweep(W_tr, 2, 2 * p_tr, "-")
    W_te_c <- sweep(W_te, 2, 2 * p_tr, "-")

    # G_tr_tr and cross-G
    n_tr    <- nrow(W_tr_c)
    G_tr_tr <- tcrossprod(W_tr_c) / k
    G_te_tr <- tcrossprod(W_te_c, W_tr_c) / k

    u_tr   <- u_train_list[["snp_add"]]
    sg     <- sigma2_g_list["snp_add"]
    lambda <- sigma2_e / sg

    diag(G_tr_tr) <- diag(G_tr_tr) + jitter
    M  <- G_tr_tr + diag(lambda, n_tr)
    x  <- tryCatch(
      solve(M, u_tr),
      error = function(e) solve(M + diag(1e-4, n_tr), u_tr)
    )

    u_te <- as.vector(G_te_tr %*% x)
    names(u_te) <- test_ids_snp
    result[["snp_add"]] <- u_te
  }

  # ── MH additive ──────────────────────────────────────────
  if ("mh_add" %in% g_names) {
    if (is.null(markers_new$mh_add) || is.null(markers_train$mh_add)) {
      stop("'markers_new$mh_add' and 'markers_train$mh_add' required for mh_add component.")
    }

    # Parse hap matrices for train and test
    parsed_tr <- .parse_mh_all_chr(markers_train$mh_add)
    parsed_te <- .parse_mh_all_chr(markers_new$mh_add)

    hap1_tr <- parsed_tr$hap1
    hap2_tr <- parsed_tr$hap2
    hap1_te <- parsed_te$hap1
    hap2_te <- parsed_te$hap2
    test_ids_mh <- parsed_te$ids

    n_blocks <- ncol(hap1_tr)
    if (ncol(hap1_te) != n_blocks) {
      stop(sprintf(
        "mh_add: markers_train has %d blocks, markers_new has %d blocks.",
        n_blocks, ncol(hap1_te)
      ))
    }

    # Build W_ah matrices for train and test using training allele freqs
    W_tr_mh <- .build_wah_from_hap(hap1_tr, hap2_tr, ref_hap1 = NULL, ref_hap2 = NULL)
    W_te_mh <- .build_wah_from_hap(hap1_te, hap2_te,
                                    ref_hap1 = hap1_tr, ref_hap2 = hap2_tr)

    n_tr    <- nrow(W_tr_mh)
    k_mh    <- ncol(W_tr_mh)  # already normalised inside .build_wah_from_hap
    G_tr_tr <- tcrossprod(W_tr_mh) / k_mh
    G_te_tr <- tcrossprod(W_te_mh, W_tr_mh) / k_mh

    u_tr   <- u_train_list[["mh_add"]]
    sg     <- sigma2_g_list["mh_add"]
    lambda <- sigma2_e / sg

    diag(G_tr_tr) <- diag(G_tr_tr) + jitter
    M  <- G_tr_tr + diag(lambda, n_tr)
    x  <- tryCatch(
      solve(M, u_tr),
      error = function(e) solve(M + diag(1e-4, n_tr), u_tr)
    )

    u_te <- as.vector(G_te_tr %*% x)
    names(u_te) <- test_ids_mh
    result[["mh_add"]] <- u_te
  }

  if (length(result) == 0) {
    stop("No recognised marker components found. Check names of markers_new.")
  }

  result
}

#' Parse MH hap list into stacked hap1/hap2 matrices
#' @noRd
.parse_mh_all_chr <- function(mh_list) {
  parsed <- lapply(seq_along(mh_list), function(k) {
    df  <- mh_list[[k]]
    chr <- if (!is.null(names(mh_list))) names(mh_list)[k] else paste0("chr", k)
    .parse_mh_chr(df, chr)
  })
  ids      <- parsed[[1]]$ids
  hap1_all <- do.call(cbind, lapply(parsed, `[[`, "hap1"))
  hap2_all <- do.call(cbind, lapply(parsed, `[[`, "hap2"))
  n_alleles_all <- unlist(lapply(parsed, `[[`, "n_alleles"))
  list(ids = ids, hap1 = hap1_all, hap2 = hap2_all, n_alleles = n_alleles_all)
}

#' Build W_ah matrix from hap1/hap2 using training allele frequencies
#'
#' Implements Da (2015) W_ah coding:
#'   z_ah = x_ah - 2*p_ah
#' where p_ah = freq of allele a at block h, estimated from training.
#' Most frequent allele per block is dropped (reference allele).
#'
#' @param hap1 integer matrix (n x n_blocks), strand 1 allele codes (0-based)
#' @param hap2 integer matrix (n x n_blocks), strand 2 allele codes (0-based)
#' @param ref_hap1 training hap1 for allele frequency estimation (NULL = use hap1)
#' @param ref_hap2 training hap2 for allele frequency estimation (NULL = use hap2)
#' @return numeric matrix W_ah (n x n_cols_after_dropping_ref)
#' @noRd
.build_wah_from_hap <- function(hap1, hap2, ref_hap1 = NULL, ref_hap2 = NULL) {

  n_blocks <- ncol(hap1)

  # Reference for freq estimation: training data
  ref1 <- if (is.null(ref_hap1)) hap1 else ref_hap1
  ref2 <- if (is.null(ref_hap2)) hap2 else ref_hap2

  W_list <- vector("list", n_blocks)

  for (b in seq_len(n_blocks)) {
    # All alleles at block b from training (0-based codes)
    alleles_ref <- c(ref1[, b], ref2[, b])
    alleles_new <- c(hap1[, b], hap2[, b])
    unknown <- setdiff(unique(alleles_new), all_codes)
    if (length(unknown) > 0) {
      warning(sprintf("Block %d: test alleles not seen in training: %s. Coded as 0.",
                      b, paste(unknown, collapse = ", ")))
    }

    all_codes <- sort(unique(alleles_ref))
    n_alleles <- length(all_codes)

    if (n_alleles < 2) {
      # Monomorphic block: skip (contributes zero W columns)
      next
    }

    # Allele frequencies from training
    tbl   <- tabulate(alleles_ref + 1L, nbins = max(alleles_ref) + 1L)
    freqs <- tbl[all_codes + 1L] / sum(tbl[all_codes + 1L])

    # Drop most frequent allele (reference)
    ref_allele <- all_codes[which.max(freqs)]
    keep_codes <- all_codes[all_codes != ref_allele]
    keep_freqs <- freqs[all_codes != ref_allele]

    # Allele count matrix for new individuals (diploid)
    # x_ah = count of allele a at block h for individual i
    for (ki in seq_along(keep_codes)) {
      a    <- keep_codes[ki]
      p_a  <- keep_freqs[ki]
      x_ah <- as.integer(hap1[, b] == a) + as.integer(hap2[, b] == a)
      z_ah <- x_ah - 2 * p_a
      W_list[[b]] <- cbind(W_list[[b]], z_ah)
    }
  }

  # Remove NULL entries (monomorphic blocks)
  W_list <- Filter(Negate(is.null), W_list)

  if (length(W_list) == 0) {
    stop("All MH blocks are monomorphic — cannot build W_ah matrix.")
  }

  do.call(cbind, W_list)
}

#' Print method for masreml_pred
#' @export
print.masreml_pred <- function(x, ...) {
  cat("\n── masreml Prediction ────────────────────────────────\n")
  cat(sprintf("  n_train   : %d\n", x$n_train))
  cat(sprintf("  n_test    : %d\n", x$n_test))
  cat(sprintf("  Components: %s\n", paste(names(x$gebv), collapse = ", ")))
  g <- x$total_gebv
  cat(sprintf(
    "  GEBV range: [%.4f, %.4f]  mean=%.4f  sd=%.4f\n",
    min(g), max(g), mean(g), sd(g)
  ))
  if (!is.null(x$fitted)) {
    cat(sprintf(
      "  Fitted P() : [%.4f, %.4f]  mean=%.4f\n",
      min(x$fitted), max(x$fitted), mean(x$fitted)
    ))
  }
  cat("──────────────────────────────────────────────────────\n\n")
  invisible(x)
}