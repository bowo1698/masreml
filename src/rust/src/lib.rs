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

/// Build SNP additive G matrix (VanRaden)
/// weights: optional PP_j vector for GWABLUP (NULL = standard GBLUP)
#[extendr]
fn r_build_g_snp_add(
    w: RMatrix<f64>,
    weights: Nullable<&[f64]>,
    allele_freq: Nullable<&[f64]>,
) -> Result<RMatrix<f64>> {
    build_g_snp_add(w, weights, allele_freq)
}

/// Build SNP dominance D matrix (Da et al. 2014)
#[extendr]
fn r_build_g_snp_dom(w: RMatrix<f64>) -> Result<RMatrix<f64>> {
    build_g_snp_dom(w)
}

/// Build MH additive Agh matrix (Da 2015)
/// weights: optional PP_j vector per locus for GWABLUP (NULL = standard GBLUP)
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

/// Build pedigree A matrix (Henderson)
#[extendr]
fn r_build_a_ped(sire: &[i32], dam: &[i32], n: i32) -> Result<RMatrix<f64>> {
    build_a_ped(sire, dam, n)
}

/// Run adaptive REML (HE -> AI-REML -> EM-REML fallback)
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

/// Solve EBV — auto-selects Cholesky (n < 10k) or PCG (n >= 10k)
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

/// Run EMMAX GWAS for SNP markers
/// Returns list(lr, beta, se, pval)
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

/// Run EMMAX GWAS for MH markers
/// block_sizes: integer vector, n_alleles-1 per block
/// Returns list(lr, beta, se, pval)
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

/// Smooth LR values and compute posterior probabilities
/// Returns list(smoothed_lr, pp)
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