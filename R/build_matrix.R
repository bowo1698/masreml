#' Route and validate marker inputs, build G matrices
#'
#' Supports two input modes:
#'   1. Raw marker matrices (W) → Rust builds G
#'   2. Pre-built G matrices → passed directly to solver
#'
#' @keywords internal

# ============================================================
# Main router
# ============================================================

#' Parse and validate all marker/G inputs
#' Returns named list of G matrices ready for solver
#'
#' @param markers list with optional elements:
#'   snp_add: raw genotype matrix (n x m, values 0/1/2)
#'   snp_dom: raw genotype matrix (n x m, values 0/1/2)
#'   mh_add:  list of chr data.frames (hap_geno_1, ..., hap_geno_K)
#' @param G list with optional pre-built G matrices (n x n):
#'   snp_add, snp_dom, mh_add
#' @param ids character vector of individual IDs (for alignment)
#' @return named list of G matrices
#' @noRd
.build_g_matrices <- function(markers = NULL, G = NULL, ids = NULL) {

  g_out <- list()

  # ── Pre-built G matrices (Opsi C: pass directly) ──────────
  if (!is.null(G)) {
    for (nm in names(G)) {
      .validate_g_matrix(G[[nm]], nm, ids)
      g_out[[nm]] <- G[[nm]]
    }
  }

  # ── Raw marker matrices → Rust builder ────────────────────
  if (!is.null(markers)) {

    # SNP additive
    if (!is.null(markers$snp_add)) {
      w <- .validate_snp_matrix(markers$snp_add, "snp_add", ids)
      g_out[["snp_add"]] <- .build_g_snp(w, type = "additive")
    }

    # SNP dominance
    if (!is.null(markers$snp_dom)) {
      w <- .validate_snp_matrix(markers$snp_dom, "snp_dom", ids)
      g_out[["snp_dom"]] <- .build_g_snp(w, type = "dominance")
    }

    # MH additive
    if (!is.null(markers$mh_add)) {
      g_out[["mh_add"]] <- .build_g_mh(markers$mh_add, ids)
    }
  }

  if (length(g_out) == 0) {
    stop("No marker data or G matrices provided.")
  }

  g_out
}

# ============================================================
# SNP builder
# ============================================================

#' @noRd
.build_g_snp <- function(w, type = "additive", weights = NULL, allele_freq = NULL) {
  if (type == "additive") {
    G <- r_build_g_snp_add(w, weights, allele_freq)
  } else {
    G <- r_build_g_snp_dom(w)
  }
  rownames(G) <- rownames(w)
  colnames(G) <- rownames(w)
  G
}

#' @noRd
.validate_snp_matrix <- function(w, label, ids = NULL) {

  # Convert data.frame to matrix
  if (is.data.frame(w)) {
    # If first col is ID, use as rownames
    if (is.character(w[[1]]) || is.factor(w[[1]])) {
      rn <- as.character(w[[1]])
      w  <- as.matrix(w[, -1, drop = FALSE])
      rownames(w) <- rn
    } else {
      w <- as.matrix(w)
    }
  }

  if (!is.matrix(w)) {
    stop(sprintf("[%s] Must be a matrix or data.frame.", label))
  }

  if (!is.numeric(w)) {
    stop(sprintf("[%s] Must be numeric (0/1/2).", label))
  }

  if (anyNA(w)) {
    stop(sprintf("[%s] Contains NA values. Impute before analysis.", label))
  }

  invalid <- !w %in% c(0, 1, 2)
  if (any(invalid)) {
    stop(sprintf(
      "[%s] Invalid genotype values detected. Only 0/1/2 allowed.",
      label
    ))
  }

  # Align IDs if provided
  if (!is.null(ids) && !is.null(rownames(w))) {
    if (!all(ids %in% rownames(w))) {
      missing_ids <- ids[!ids %in% rownames(w)]
      stop(sprintf(
        "[%s] Missing IDs in marker data: %s",
        label, paste(head(missing_ids, 5), collapse = ", ")
      ))
    }
    w <- w[ids, , drop = FALSE]
  }

  storage.mode(w) <- "double"
  w
}

# ============================================================
# MH builder
# ============================================================

#' @noRd
.build_g_mh <- function(mh_list, ids = NULL, weights = NULL, ref_mh = NULL) {

  #── Detection mode: hap_matrix or mh_list ─────────────────
  # If the input is a matrix (n x n_blocks*2), convert to parsed directly
  if (is.matrix(mh_list) || is.data.frame(mh_list)) {
    return(.build_g_mh_from_hapmatrix(
      hap_matrix = mh_list,
      ref_hap    = ref_mh,
      ids        = ids,
      weights    = weights
    ))
  }

  # Validate: must be list of data.frames
  if (!is.list(mh_list)) {
    stop("[mh_add] Must be a list of data.frames (one per chromosome), or a haplotype matrix (n x n_blocks*2).")
  }

  # Parse each chromosome
  parsed <- lapply(seq_along(mh_list), function(k) {
    df <- mh_list[[k]]
    chr_label <- if (!is.null(names(mh_list))) {
      names(mh_list)[k]
    } else {
      paste0("chr", k)
    }
    .parse_mh_chr(df, chr_label)
  })

  # Extract IDs from first chr for alignment
  chr_ids <- parsed[[1]]$ids

  # Validate consistent IDs across chromosomes
  for (k in seq_along(parsed)) {
    if (!identical(parsed[[k]]$ids, chr_ids)) {
      stop(sprintf(
        "[mh_add] IDs in chr%d do not match chr1. Check input order.",
        k
      ))
    }
  }

  # Align to y IDs if provided
  if (!is.null(ids)) {
    if (!all(ids %in% chr_ids)) {
      missing_ids <- ids[!ids %in% chr_ids]
      stop(sprintf(
        "[mh_add] Missing IDs: %s",
        paste(head(missing_ids, 5), collapse = ", ")
      ))
    }
    idx <- match(ids, chr_ids)
    parsed <- lapply(parsed, function(p) {
      list(
        ids       = p$ids[idx],
        hap1      = p$hap1[idx, , drop = FALSE],
        hap2      = p$hap2[idx, , drop = FALSE],
        n_alleles = p$n_alleles
      )
    })
    chr_ids <- ids
  }

  # Stack all chromosomes: combine hap1/hap2 column-wise across chr
  hap1_all <- do.call(cbind, lapply(parsed, `[[`, "hap1"))
  hap2_all <- do.call(cbind, lapply(parsed, `[[`, "hap2"))
  n_alleles_all <- unlist(lapply(parsed, `[[`, "n_alleles"))

  # Parse ref_mh if available for training-based allele frequencies
  ref_h1_all <- NULL
  ref_h2_all <- NULL
  if (!is.null(ref_mh)) {
    parsed_ref <- lapply(seq_along(ref_mh), function(k) {
      df  <- ref_mh[[k]]
      chr <- if (!is.null(names(ref_mh))) names(ref_mh)[k] else paste0("chr", k)
      .parse_mh_chr(df, chr)
    })
    ref_h1_all <- do.call(cbind, lapply(parsed_ref, `[[`, "hap1"))
    ref_h2_all <- do.call(cbind, lapply(parsed_ref, `[[`, "hap2"))
  }

  # Call Rust builder
  G <- r_build_g_mh_add(
    hap1      = hap1_all,
    hap2      = hap2_all,
    n_alleles = as.integer(n_alleles_all),
    weights   = weights,
    ref_hap1  = ref_h1_all,
    ref_hap2  = ref_h2_all
  )

  rownames(G) <- chr_ids
  colnames(G) <- chr_ids
  G
}

#' @noRd
.parse_mh_chr <- function(df, chr_label) {

  if (!is.data.frame(df) && !is.matrix(df)) {
    stop(sprintf("[%s] Each chromosome entry must be a data.frame.", chr_label))
  }

  if (is.matrix(df)) df <- as.data.frame(df)

  # First column = ID
  ids <- as.character(df[[1]])
  data_cols <- df[, -1, drop = FALSE]
  n_data_cols <- ncol(data_cols)

  # Must have even number of data columns (paired strands)
  if (n_data_cols %% 2 != 0) {
    stop(sprintf(
      "[%s] Expected paired strand columns (even count), got %d.",
      chr_label, n_data_cols
    ))
  }

  n_blocks <- n_data_cols / 2

  # Separate strand1 (odd) and strand2 (even) columns
  s1_idx <- seq(1, n_data_cols, by = 2)
  s2_idx <- seq(2, n_data_cols, by = 2)

  hap1 <- as.matrix(data_cols[, s1_idx, drop = FALSE])
  hap2 <- as.matrix(data_cols[, s2_idx, drop = FALSE])

  # Validate: integer values, no NA
  if (anyNA(hap1) || anyNA(hap2)) {
    stop(sprintf("[%s] Missing values detected in haplotype data.", chr_label))
  }

  if (!all(hap1 == floor(hap1)) || !all(hap2 == floor(hap2))) {
    stop(sprintf("[%s] Haplotype allele codes must be integers.", chr_label))
  }

  # Compute n_alleles per block: max allele code + 1 (0-based)
  n_alleles <- sapply(seq_len(n_blocks), function(b) {
    max_s1 <- max(hap1[, b])
    max_s2 <- max(hap2[, b])
    max(max_s1, max_s2) + 1L
  })

  storage.mode(hap1) <- "integer"
  storage.mode(hap2) <- "integer"

  list(
    ids       = ids,
    hap1      = hap1,       # n x n_blocks
    hap2      = hap2,       # n x n_blocks
    n_alleles = n_alleles   # n_blocks integer vector
  )
}

#' Build G_mh directly from raw haplotype matrix (n x n_blocks*2)
#' Columns alternate strand1/strand2 per block: s1_b1, s2_b1, s1_b2, s2_b2, ...
#' Allele codes can be any integer (sparse, non-sequential) — re-encoded internally
#' @param hap_matrix integer matrix (n x n_blocks*2)
#' @param ref_hap integer matrix (n_ref x n_blocks*2), training rows for allele freq
#' @param ids character vector of individual IDs
#' @param weights numeric vector of PP weights per block (for GWABLUP)
#' @noRd
.build_g_mh_from_hapmatrix <- function(hap_matrix, ref_hap = NULL,
                                        ids = NULL, weights = NULL) {
  if (!is.matrix(hap_matrix)) hap_matrix <- as.matrix(hap_matrix)
  storage.mode(hap_matrix) <- "integer"

  n        <- nrow(hap_matrix)
  n_cols   <- ncol(hap_matrix)

  if (n_cols %% 2 != 0) {
    stop("[mh_add] hap_matrix must have even number of columns (paired strands).")
  }
  n_blocks <- n_cols / 2

  if (!is.null(ref_hap)) {
    if (!is.matrix(ref_hap)) ref_hap <- as.matrix(ref_hap)
    storage.mode(ref_hap) <- "integer"
    if (ncol(ref_hap) != n_cols) {
      stop(sprintf(
        "[mh_add] ref_hap ncol (%d) != hap_matrix ncol (%d).",
        ncol(ref_hap), n_cols
      ))
    }
  }

  # IDs
  chr_ids <- if (!is.null(ids)) {
    ids
  } else if (!is.null(rownames(hap_matrix))) {
    rownames(hap_matrix)
  } else {
    as.character(seq_len(n))
  }

  # Re-encode per block dan build hap1/hap2 matrices
  hap1_enc      <- matrix(0L, nrow = n, ncol = n_blocks)
  hap2_enc      <- matrix(0L, nrow = n, ncol = n_blocks)
  ref_h1_enc    <- if (!is.null(ref_hap)) matrix(0L, nrow=nrow(ref_hap), ncol=n_blocks) else NULL
  ref_h2_enc    <- if (!is.null(ref_hap)) matrix(0L, nrow=nrow(ref_hap), ncol=n_blocks) else NULL
  n_alleles_vec <- integer(n_blocks)

  for (b in seq_len(n_blocks)) {
    s1_col <- 2*b - 1
    s2_col <- 2*b

    h1 <- hap_matrix[, s1_col]
    h2 <- hap_matrix[, s2_col]
    r1 <- if (!is.null(ref_hap)) ref_hap[, s1_col] else h1
    r2 <- if (!is.null(ref_hap)) ref_hap[, s2_col] else h2

    enc <- .reencode_hap_block(h1, h2, r1, r2)

    hap1_enc[, b]    <- enc$h1
    hap2_enc[, b]    <- enc$h2
    n_alleles_vec[b] <- enc$n_alleles

    if (!is.null(ref_hap)) {
      ref_enc          <- .reencode_hap_block(r1, r2, r1, r2)
      ref_h1_enc[, b]  <- ref_enc$h1
      ref_h2_enc[, b]  <- ref_enc$h2
    }
  }

  # Call Rust builder
  G <- r_build_g_mh_add(
    hap1      = hap1_enc,
    hap2      = hap2_enc,
    n_alleles = as.integer(n_alleles_vec),
    weights   = weights,
    ref_hap1  = ref_h1_enc,
    ref_hap2  = ref_h2_enc
  )

  rownames(G) <- chr_ids
  colnames(G) <- chr_ids
  G
}

#' Re-encode allele codes to sequential 0-based using training reference
#' Unknown alleles in target (not seen in ref) are mapped to 0
#' @noRd
.reencode_hap_block <- function(h1, h2, ref_h1, ref_h2) {
  unique_alleles <- sort(unique(c(ref_h1, ref_h2)))
  allele_map     <- setNames(seq_along(unique_alleles) - 1L,
                              as.character(unique_alleles))

  recode <- function(x) {
    r <- allele_map[as.character(x)]
    r[is.na(r)] <- 0L
    as.integer(r)
  }

  list(
    h1        = recode(h1),
    h2        = recode(h2),
    n_alleles = length(unique_alleles)
  )
}

# ============================================================
# Pre-built G matrix validator
# ============================================================

#' Validate weights vector for GWABLUP
#' @param weights numeric vector of PP_j, length must match n_markers
#' @param n_markers expected length
#' @param label marker type label for error messages
#' @noRd
.validate_weights <- function(weights, n_markers, label) {
  if (is.null(weights)) return(invisible(NULL))

  if (!is.numeric(weights)) {
    stop(sprintf("[%s] weights must be a numeric vector.", label))
  }

  if (length(weights) != n_markers) {
    stop(sprintf(
      "[%s] weights length %d != n_markers %d.",
      label, length(weights), n_markers
    ))
  }

  if (anyNA(weights)) {
    stop(sprintf("[%s] weights contains NA values.", label))
  }

  if (any(weights < 0)) {
    stop(sprintf("[%s] weights must be non-negative.", label))
  }

  invisible(weights)
}

#' Build weighted G matrix from gwas_result (for GWABLUP)
#' @param markers list marker input (snp_add or mh_add)
#' @param gwas_result output from run_gwas() — list with element $pp
#' @param ids character vector individual IDs
#' @return named list with one G_wa matrix
#' @noRd
.build_g_matrices_weighted <- function(markers, gwas_result, ids = NULL, min_weight = 1e-4,
                                        ref_W = NULL, ref_mh = NULL) {

  if (is.null(gwas_result$pp)) {
    stop("gwas_result must contain $pp element. Run run_gwas() first.")
  }

  pp <- gwas_result$pp
  pp <- pmax(pp, min_weight)
  pp <- pp / mean(pp)
  g_out <- list()

  if (!is.null(markers$snp_add)) {
    w <- .validate_snp_matrix(markers$snp_add, "snp_add", ids)
    .validate_weights(pp, ncol(w), "snp_add")
    allele_freq <- if (!is.null(ref_W)) {
      ref_W <- .validate_snp_matrix(ref_W, "snp_add_ref")
      colMeans(ref_W) / 2
    } else NULL
    g_out[["snp_add"]] <- .build_g_snp(w, type = "additive", weights = pp,
                                         allele_freq = allele_freq)
  }

  if (!is.null(markers$mh_add)) {
    # Support both mh_list and hap_matrix input
    if (is.matrix(markers$mh_add)) {
      n_blocks <- ncol(markers$mh_add) / 2
      .validate_weights(pp, n_blocks, "mh_add")
    } else {
      parsed <- lapply(seq_along(markers$mh_add), function(k) {
        df <- markers$mh_add[[k]]
        chr_label <- if (!is.null(names(markers$mh_add))) {
          names(markers$mh_add)[k]
        } else paste0("chr", k)
        .parse_mh_chr(df, chr_label)
      })
      n_loci_total <- sum(sapply(parsed, function(p) ncol(p$hap1)))
      .validate_weights(pp, n_loci_total, "mh_add")
    }
    g_out[["mh_add"]] <- .build_g_mh(markers$mh_add, ids, weights = pp, ref_mh = ref_mh)
  }

  if (length(g_out) == 0) {
    stop("No marker data provided for weighted G matrix.")
  }

  g_out
}

#' @noRd
.validate_g_matrix <- function(g, label, ids = NULL) {

  if (!is.matrix(g)) {
    stop(sprintf("[%s] Pre-built G must be a matrix.", label))
  }

  if (!is.numeric(g)) {
    stop(sprintf("[%s] Pre-built G must be numeric.", label))
  }

  if (nrow(g) != ncol(g)) {
    stop(sprintf(
      "[%s] G matrix must be square: %d x %d",
      label, nrow(g), ncol(g)
    ))
  }

  if (anyNA(g)) {
    stop(sprintf("[%s] G matrix contains NA values.", label))
  }

  # Check symmetry (tolerance 1e-6)
  if (!isSymmetric(g, tol = 1e-6)) {
    stop(sprintf("[%s] G matrix is not symmetric.", label))
  }

  # Align IDs if rownames available
  if (!is.null(ids) && !is.null(rownames(g))) {
    if (!all(ids %in% rownames(g))) {
      missing_ids <- ids[!ids %in% rownames(g)]
      stop(sprintf(
        "[%s] Missing IDs in G matrix: %s",
        label, paste(head(missing_ids, 5), collapse = ", ")
      ))
    }
    g <- g[ids, ids, drop = FALSE]
  }

  invisible(g)
}