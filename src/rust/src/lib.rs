// src/lib.rs

//! # masreml — Rust kernel
//!
//! Computational backend for the `masreml` R package: universal REML-BLUP
//! genomic prediction supporting biallelic SNP, multi-allelic
//! microhaplotype, and pedigree-based relationship matrices.
//!
//! ## Module map
//!
//! - [`matrix`] — relationship-matrix construction:
//!   * `snp_additive` / `snp_dominance` — VanRaden (2008) additive and Da
//!     et al. (2014) / Wang & Da (2014) dominance G matrices.
//!   * `mh_additive` — Da (2015) $W_{\alpha h}$ additive $G$ for
//!     multi-allelic microhaplotypes, with the per-locus frequency-weighted
//!     row shrinkage.
//!   * `pedigree` — Henderson (1976) recursive numerator relationship
//!     matrix $A$.
//! - [`reml`] — variance-component estimation:
//!   * `he_regression` — Haseman–Elston regression (closed-form,
//!     starting values).
//!   * `ai_reml` — Average-Information REML (Johnson & Thompson, 1995).
//!   * `em_reml` — Expectation-Maximization REML (Dempster et al., 1977;
//!     Meyer, 1989). Slow but always non-negative.
//!   * `adaptive` — auto-selects between HE / AI / EM based on conditioning
//!     and convergence diagnostics.
//! - [`solver`] — fixed/random effects solvers given fitted variance
//!   components:
//!   * `cholesky` — direct factorization; default for $n < 10{,}000$.
//!   * `pcg` — preconditioned conjugate gradient; default for larger $n$.
//!   * `factorized` — reusable Cholesky cache used by both.
//! - [`gwas`] — single-marker association:
//!   * `emmax` — EMMAX (Kang et al., 2010) with a pre-factorized $V$.
//!   * `smoother` — local moving-average smoother for likelihood-ratio
//!     statistics across markers/blocks (used by GWABLUP).
//! - [`utils`] — small `ndarray_linalg` helpers used across modules.
//!
//! ## R-facing API
//!
//! Public entry points are wrapped with `#[extendr]` in this file and
//! exposed to R as `r_build_g_snp_add`, `r_build_g_snp_dom`,
//! `r_build_g_mh_add`, `r_build_a_ped`, `r_run_reml`, `r_run_emmax`, etc.
//! The `extendr_module!` macro at the bottom of this file registers them
//! all in one block.
//!
//! ## References
//!
//! - VanRaden, P. M. (2008). Efficient methods to compute genomic
//!   predictions. *J. Dairy Sci.*, 91:4414–4423.
//! - Da, Y. (2015). Multi-allelic haplotype model based on genetic
//!   partition. *BMC Genetics*, 16:144.
//! - Henderson, C. R. (1976). A simple method for computing the inverse
//!   of a numerator relationship matrix. *Biometrics*, 32:69–83.
//! - Johnson, D. L. & Thompson, R. (1995). Restricted maximum likelihood
//!   estimation of variance components. *J. Dairy Sci.*, 78:449–456.
//! - Kang, H. M. et al. (2010). Variance component model to account for
//!   sample structure in GWAS. *Nat. Genet.*, 42:348–354.

use extendr_api::prelude::*;

mod matrix;
mod reml;
mod solver;
mod utils;
mod gwas;

// Re-export semua extendr functions
use matrix::snp_additive::build_g_snp_add;
use matrix::snp_dominance::build_g_snp_dom;
use matrix::mh_additive::build_g_mh_add;
use matrix::pedigree::build_a_ped;
use reml::adaptive::run_reml;

/// Build the SNP additive genomic relationship matrix (VanRaden 2008).
///
/// # Arguments
///
/// - `w`: `(n, m)` matrix of raw SNP dosages with entries in `{0, 1, 2}`.
/// - `weights`: optional length-`m` per-marker weights. `NULL` gives
///   standard VanRaden G; a non-null vector gives the GWAS-weighted
///   matrix used by `gwablup()`.
/// - `allele_freq`: optional length-`m` reference allele frequencies.
///   `NULL` computes them from `w` itself (training set); pass
///   training frequencies when encoding a test set to avoid data
///   leakage.
///
/// # Returns
///
/// `(n, n)` symmetric positive (semi-)definite G matrix scaled so
/// that `mean(diag(G)) ≈ 1` under HWE.
///
/// # Errors
///
/// Returns an R-side error if (a) `w` has non-finite values, (b)
/// dimensions of `weights` or `allele_freq` don't match `m`, or (c)
/// every marker is monomorphic (scaling constant `k = 0`).
///
/// See [`crate::matrix::snp_additive`] for the underlying math.
#[extendr]
fn r_build_g_snp_add(
    w: RMatrix<f64>,
    weights: Nullable<&[f64]>,
    allele_freq: Nullable<&[f64]>,
) -> Result<RMatrix<f64>> {
    build_g_snp_add(w, weights, allele_freq)
}

/// Build the SNP dominance D matrix (Da et al. 2014 / Wang & Da 2014).
///
/// # Arguments
///
/// - `w`: `(n, m)` matrix of raw SNP dosages with entries in
///   `{0, 1, 2}`. The function computes the dominance encoding
///   `w_δ_ij` internally — do not pre-compute it on the R side.
///
/// # Returns
///
/// `(n, n)` symmetric dominance relationship matrix, scaled by
/// `k_δ = tr(W_δ W_δ') / n`.
///
/// See [`crate::matrix::snp_dominance`] for the coding rule and a
/// discussion of when to fit dominance variance separately.
#[extendr]
fn r_build_g_snp_dom(w: RMatrix<f64>) -> Result<RMatrix<f64>> {
    build_g_snp_dom(w)
}

/// Build the multi-allelic additive G matrix (Da 2015).
///
/// # Arguments
///
/// - `hap1`, `hap2`: `(n, n_loci)` integer matrices of phased
///   microhaplotype allele codes (paternal and maternal).
/// - `n_alleles`: length-`n_loci` vector giving the number of
///   distinct microhaplotypes per locus.
/// - `weights`: optional length-`n_loci` per-locus weights. `NULL`
///   gives standard MH GBLUP; non-null gives the GWAS-weighted
///   version (analogous to the SNP case).
/// - `ref_hap1`, `ref_hap2`: optional reference (training)
///   haplotype matrices used to derive allele frequencies and
///   choose baseline microhaplotypes. When `NULL`, frequencies are
///   computed from `hap1`/`hap2` themselves.
///
/// # Returns
///
/// `(n, n)` additive G matrix `G_αh / k_αh`. The post-encoding
/// row shrinkage (see [`crate::matrix::mh_additive`]) is applied
/// per locus before assembly.
///
/// # Errors
///
/// Returns an R-side error on shape mismatches or on a degenerate
/// dataset where all loci are monomorphic.
#[extendr]
fn r_build_g_mh_add(
    hap1: RMatrix<i32>,
    hap2: RMatrix<i32>,
    n_alleles: &[i32],
    weights: Nullable<&[f64]>,
    ref_hap1: Nullable<RMatrix<i32>>,
    ref_hap2: Nullable<RMatrix<i32>>,
) -> Result<RMatrix<f64>> {
    build_g_mh_add(hap1, hap2, n_alleles, weights, ref_hap1, ref_hap2)
}

/// Build the pedigree numerator relationship matrix A (Henderson 1976).
///
/// # Arguments
///
/// - `sire`, `dam`: length-`n` integer vectors with `0` for
///   unknown parent and 1-based indices into the same pedigree
///   otherwise. The pedigree **must be topologically pre-sorted**:
///   every parent index must be smaller than its offspring index.
///   This precondition is the caller's responsibility; violation
///   silently produces an incorrect A.
/// - `n`: number of individuals in the pedigree.
///
/// # Returns
///
/// `(n, n)` symmetric A matrix. The diagonal carries
/// `a_ii = 1 + F_i` (inbreeding); off-diagonals carry the
/// expected fraction of identity-by-descent under the recursion
/// fixed in v0.4.0.
///
/// See [`crate::matrix::pedigree`] for the recursion and the
/// v0.4.0 bug-fix notes.
#[extendr]
fn r_build_a_ped(sire: &[i32], dam: &[i32], n: i32) -> Result<RMatrix<f64>> {
    build_a_ped(sire, dam, n)
}

/// Run the adaptive REML dispatcher (HE → AI → optional EM
/// fallback).
///
/// # Arguments
///
/// - `y`: length-`n` phenotype vector.
/// - `x`: `(n, c)` fixed-effects design matrix.
/// - `g_list`: named R list of `(n, n)` relationship matrices. The
///   list names become the component labels in the output.
/// - `method`: `"auto"` (default cascade), `"AI"`, `"EM"`, `"HE"`,
///   or `"HI"` (HE → AI with no EM fallback). See
///   [`crate::reml::adaptive`] for the cascade details.
/// - `max_iter`: maximum iterations for AI / EM. HE is single-shot.
/// - `tol`: convergence tolerance on the max relative change in
///   variance components.
/// - `n_threads`: thread pool size for Rayon and OpenBLAS.
///   `0` or negative ⇒ all available cores.
///
/// # Returns
///
/// R list with `sigma2`, `h2`, `loglik`, `n_iter`, `algorithm`,
/// `converged`. On failure the same list shape is returned with an
/// extra `error` string and null placeholders; check
/// `converged == TRUE` on the R side to detect this.
#[extendr]
fn r_run_reml(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    method: &str,
    max_iter: i32,
    tol: f64,
    n_threads: i32,
) -> List {
    run_reml(y, x, g_list, method, max_iter, tol, n_threads)
}

/// Solve the mixed-model equations for BLUP (EBV) given fitted
/// variance components.
///
/// # Arguments
///
/// - `y`, `x`, `g_list`: same as `r_run_reml`.
/// - `sigma2`: length `n_random + 1` variance-component vector from
///   REML, in the order `(σ²_1, …, σ²_K, σ²_e)`.
/// - `solver`: `"auto"` (default), `"cholesky"`, or `"pcg"`. Auto
///   picks Cholesky for `n < 10 000` and PCG otherwise — see
///   [`crate::solver::auto_select_solver`].
/// - `max_iter`, `tol`: PCG convergence controls. Ignored when
///   Cholesky is used (Cholesky is exact, one shot).
///
/// # Returns
///
/// R list with:
/// - `fixed_effects`: length-`c` fixed-effect estimates.
/// - `total_gebv`: summed GEBV across all random-effect components.
/// - `solver`: which solver actually ran (`"cholesky"` or `"pcg"`).
/// - `n_iter`: PCG iteration count (`0` for Cholesky).
/// - One named field per relationship-matrix component, carrying its
///   per-component EBV vector.
///
/// # Errors
///
/// Returns an R-side error if (a) `sigma2.len() != n_random + 1`,
/// (b) V is not positive definite (typically from a negative σ² in
/// the input), or (c) PCG fails to converge within `max_iter`.
#[extendr]
fn r_solve_ebv(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    sigma2: &[f64],
    solver: &str,   // "auto", "cholesky", "pcg"
    max_iter: i32,
    tol: f64,
) -> Result<List> {
    let n = y.len();
    let n_random = g_list.len();

    if sigma2.len() != n_random + 1 {
        return Err(Error::from(format!(
            "sigma2 length {} must be n_random + 1 = {}",
            sigma2.len(), n_random + 1
        )));
    }

    // Parse inputs
    let y_arr = ndarray::Array1::from_vec(y.to_vec());
    let x_t = ndarray::Array2::from_shape_vec(
        (x.ncols(), x.nrows()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let x_arr = x_t.reversed_axes().to_owned();

    let mut g_matrices: Vec<(ndarray::Array2<f64>, String)> = Vec::new();
    for (name, robj) in g_list.iter() {
        let g_rmat = RMatrix::<f64>::try_from(robj)
            .map_err(|_| Error::from(
                format!("G matrix '{}' is not a numeric matrix", name)
            ))?;
        let g_t = ndarray::Array2::from_shape_vec(
            (g_rmat.ncols(), g_rmat.nrows()),
            g_rmat.data().to_vec()
        ).map_err(|e| Error::from(e.to_string()))?;
        let g = g_t.reversed_axes().to_owned();
        g_matrices.push((g, name.to_string()));
    }

    // Dispatch via solve_ebv
    let result = solver::solve_ebv(
        &y_arr, &x_arr, &g_matrices, sigma2,
        n, n_random, max_iter as usize, tol, solver
    ).map_err(|e| Error::from(e.to_string()))?;

    // Build output
    let total_gebv    = result.total_gebv();
    let fixed_effects = result.fixed_effects.to_vec();
    let solver_name   = result.solver.clone();
    let labels        = result.labels.clone();
    let gebv_vecs     = result.gebv.clone();

    let mut names: Vec<String> = vec![
        "fixed_effects".to_string(),
        "total_gebv".to_string(),
        "solver".to_string(),
        "n_iter".to_string(),
    ];
    let mut values: Vec<Robj> = vec![
        r!(fixed_effects),
        r!(total_gebv.to_vec()),
        r!(solver_name.as_str()),
        r!(result.n_iter as i32),
    ];

    for (label, gebv) in labels.iter().zip(gebv_vecs.iter()) {
        names.push(label.clone());
        values.push(r!(gebv.to_vec()));
    }

    let names_str: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    Ok(List::from_names_and_values(names_str, values)
        .map_err(|e| Error::from(e.to_string()))?)
}

/// Run an EMMAX (Kang et al. 2010) GWAS scan over biallelic SNP
/// markers using a single pre-factorised V.
///
/// # Arguments
///
/// - `w`: `(n, m)` centered SNP design matrix (VanRaden coding).
/// - `y`: length-`n` phenotype vector.
/// - `x`: `(n, c)` fixed-effects design.
/// - `sigma2_g`, `sigma2_e`: variance components from a null-model
///   REML fit (computed once before calling this).
/// - `g_u`: `(n, n)` genomic relationship matrix.
///
/// # Returns
///
/// R list with length-`m` numeric vectors:
/// - `lr`: per-marker likelihood-ratio statistic (`½ · (β̂ / SE)²`).
/// - `beta`: Wald effect estimate `β̂_j`.
/// - `se`: standard error of `β̂_j` from the cached Cholesky factor.
/// - `pval`: chi-squared p-value with df = 1.
///
/// # Performance
///
/// V is Cholesky-factored once. Each subsequent per-marker test
/// solves `V⁻¹ x_j` via the cached factor in `O(n²)` instead of
/// re-factorising in `O(n³)`. Per-marker work is parallelised over
/// markers via Rayon.
#[extendr]
fn r_run_emmax_snp(
    w: RMatrix<f64>,
    y: &[f64],
    x: RMatrix<f64>,
    sigma2_g: f64,
    sigma2_e: f64,
    g_u: RMatrix<f64>,
) -> Result<List> {
    use gwas::emmax::run_emmax_snp;
    use solver::factorized::FactorizedV;

    let n = y.len();
    let y_arr = ndarray::Array1::from_vec(y.to_vec());

    let w_t = ndarray::Array2::from_shape_vec(
        (w.ncols(), w.nrows()),
        w.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let w_arr = w_t.reversed_axes().to_owned();

    let x_t = ndarray::Array2::from_shape_vec(
        (x.ncols(), x.nrows()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let x_arr = x_t.reversed_axes().to_owned();

    let g_t = ndarray::Array2::from_shape_vec(
        (g_u.ncols(), g_u.nrows()),
        g_u.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let g_arr = g_t.reversed_axes().to_owned();

    let g_list = vec![(g_arr, "g".to_string())];
    let sigma2 = vec![sigma2_g, sigma2_e];
    let v_factor = FactorizedV::new(&g_list, &sigma2, n)
        .map_err(|e| Error::from(e.to_string()))?;

    let result = run_emmax_snp(&w_arr, &y_arr, &x_arr, &v_factor)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["lr", "beta", "se", "pval"],
        [
            r!(result.lr),
            r!(result.beta),
            r!(result.se),
            r!(result.pval),
        ]
    ).map_err(|e| Error::from(e.to_string()))?)
}

/// Run an EMMAX GWAS scan over multi-allelic microhaplotype
/// blocks. The block-level test is the multi-allelic generalisation
/// of [`r_run_emmax_snp`] — see [`crate::gwas::emmax`] for the
/// derivation.
///
/// # Arguments
///
/// - `w`: `(n, p)` flattened W_αh matrix where `p = Σ_b (h_b − 1)`
///   is the total number of non-baseline microhaplotype columns
///   across all blocks.
/// - `block_sizes`: length-`n_blocks` integer vector with the
///   number of non-baseline microhaplotypes per block (`h_b − 1`).
///   Used to slice `w` into per-block sub-matrices.
/// - `y`, `x`, `sigma2_g`, `sigma2_e`, `g_u`: same role as in
///   `r_run_emmax_snp`.
///
/// # Returns
///
/// R list with length-`n_blocks` vectors:
/// - `lr`: per-block likelihood ratio for the joint
///   `H_0: β_block = 0` (vector hypothesis with df = `h_b − 1`).
/// - `beta`: aggregated effect-size summary per block
///   (`||β̂_block||₂`).
/// - `se`: aggregated standard-error summary per block.
/// - `pval`: chi-squared p-value with df = `h_b − 1`.
///
/// # Why a block-level test
///
/// Testing each non-baseline microhaplotype individually loses
/// power because the block alleles share information — a single
/// joint test on all `h_b − 1` columns is the optimal score test
/// when the alternative is "this block has any association".
#[extendr]
fn r_run_emmax_mh(
    hap1: RMatrix<i32>,
    hap2: RMatrix<i32>,
    n_alleles: &[i32],
    y: &[f64],
    x: RMatrix<f64>,
    sigma2_g: f64,
    sigma2_e: f64,
    g_u: RMatrix<f64>,
) -> Result<List> {
    use gwas::emmax::run_emmax_mh;
    use matrix::mh_additive::build_w_mh_internal; 
    use solver::factorized::FactorizedV;

    let n = y.len();
    let y_arr = ndarray::Array1::from_vec(y.to_vec());

    // Parse hap1/hap2
    let h1 = ndarray::Array2::from_shape_vec(
        (hap1.nrows(), hap1.ncols()),
        hap1.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let h2 = ndarray::Array2::from_shape_vec(
        (hap2.nrows(), hap2.ncols()),
        hap2.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let n_alleles_usize: Vec<usize> = n_alleles.iter()
        .map(|&a| a as usize)
        .collect();

    // Build W_αh flat matrix + block_sizes dari hap1/hap2
    let (w_mh, block_sizes) = build_w_mh_internal(&h1, &h2, &n_alleles_usize, None, None)
        .map_err(|e| Error::from(e.to_string()))?;

    let x_t = ndarray::Array2::from_shape_vec(
        (x.ncols(), x.nrows()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let x_arr = x_t.reversed_axes().to_owned();

    // Build V = G_u * sigma2_g + I * sigma2_e
    let g_t = ndarray::Array2::from_shape_vec(
        (g_u.ncols(), g_u.nrows()),
        g_u.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let g_arr = g_t.reversed_axes().to_owned();

    let g_list = vec![(g_arr, "g".to_string())];
    let sigma2  = vec![sigma2_g, sigma2_e];
    let v_factor = FactorizedV::new(&g_list, &sigma2, n)
        .map_err(|e| Error::from(e.to_string()))?;

    let result = run_emmax_mh(&w_mh, &block_sizes, &y_arr, &x_arr, &v_factor)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["lr", "beta", "se", "pval"],
        [
            r!(result.lr),
            r!(result.beta),
            r!(result.se),
            r!(result.pval),
        ]
    ).map_err(|e| Error::from(e.to_string()))?)
}

/// Smooth per-marker / per-block likelihood-ratio statistics and
/// turn them into posterior probabilities for the GWABLUP pipeline.
///
/// # Pipeline
///
/// 1. **Moving average** with centred window of size `window`
///    (`floor(window/2)` markers on each side, shrunk at array
///    edges with no padding).
/// 2. **Posterior probability** per marker via Meuwissen et al. (2024):
///
///    ```text
///    PP_j = π · exp(LR_j) / (π · exp(LR_j) + (1 − π))
///    ```
///
///    Implemented in log-odds form for numerical stability:
///    `logit(PP_j) = LR_j + log(π / (1 − π))`.
///
/// # Arguments
///
/// - `lr`: per-marker (SNP) or per-block (MH) likelihood-ratio
///   statistics from `r_run_emmax_snp` / `r_run_emmax_mh`.
/// - `window`: moving-average window size (typical values 5–20).
/// - `pi`: prior probability of non-zero effect, in `(0, 1)`.
///   Typical default `0.001`.
///
/// # Returns
///
/// R list with two equal-length numeric vectors:
/// - `smoothed_lr`: the moving-average smoothed LR.
/// - `pp`: per-marker posterior probability of non-zero effect.
///
/// # GWABLUP usage
///
/// `pp` is then passed back to `r_build_g_snp_add` /
/// `r_build_g_mh_add` as the `weights` argument to build the
/// GWAS-weighted G matrix used downstream by `gwablup()`.
#[extendr]
fn r_smooth_and_pp(
    lr: &[f64],
    window: i32,
    pi: f64,
) -> Result<List> {
    use gwas::smoother::smooth_and_pp;

    let (smoothed_lr, pp) = smooth_and_pp(lr, window as usize, pi)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["smoothed_lr", "pp"],
        [r!(smoothed_lr), r!(pp)]
    ).map_err(|e| Error::from(e.to_string()))?)
}

extendr_module! {
    mod masreml;
    fn r_build_g_snp_add;
    fn r_build_g_snp_dom;
    fn r_build_g_mh_add;
    fn r_build_a_ped;
    fn r_run_reml;
    fn r_solve_ebv;
    fn r_run_emmax_snp;
    fn r_run_emmax_mh;
    fn r_smooth_and_pp;
}