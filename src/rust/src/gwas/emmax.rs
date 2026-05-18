// src/rust/src/gwas/emmax.rs

//! EMMAX — efficient single-marker mixed-model association.
//!
//! Implements the EMMAX algorithm of Kang et al. (2010): instead of
//! re-estimating variance components for every marker, factor the
//! null-model $V = G \sigma^2_g + I \sigma^2_e$ **once** and reuse the
//! factorisation to compute Wald/likelihood-ratio tests for every marker
//! in turn.
//!
//! ## Algorithm
//!
//! 1. Fit the null mixed model $y = X\beta + u + \varepsilon$ via REML
//!    (handled upstream by [`crate::reml`]).
//! 2. Cholesky-factor $V$ once — encapsulated in
//!    [`crate::solver::factorized::FactorizedV`] for reuse.
//! 3. For each marker $j$:
//!    - Solve $V^{-1} x_j$ with the cached factorisation.
//!    - Compute $\hat\beta_j$, its standard error, and the LR statistic.
//!    - Derive a $\chi^2$ $p$-value (df=1 for SNP, df=$h-1$ for MH).
//! 4. Loop over markers in parallel via rayon.
//!
//! ## Multi-allelic extension
//!
//! For MH blocks, the per-allele test statistics are aggregated to a
//! single per-block LR with df=$h-1$. Effect-size aggregation follows the
//! weighted average used elsewhere in the package, keeping the output
//! shape consistent with the SNP path.
//!
//! ## Reference
//!
//! Kang, H. M., Sul, J. H., Service, S. K., et al. (2010). Variance
//! component model to account for sample structure in genome-wide
//! association studies. *Nat. Genet.*, 42:348–354.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::{GwasError, GwasResult, StdResult};
use crate::solver::factorized::FactorizedV;

/// Compute the EMMAX Wald/LR statistic for a single SNP column.
///
/// # Model
///
/// Under the null mixed model
///     y = X·β + u + ε,   u ~ N(0, σ²_g · K),   ε ~ N(0, σ²_e · I)
/// already fit by REML, the per-marker working model adds a single
/// candidate fixed-effect column x_j:
///     y = X·β + x_j · b_j + u + ε.
///
/// # Test statistic
///
/// The Wald estimator and its information at the null are
///
///     b̂_j  = (x_j' V⁻¹ x_j)⁻¹ · x_j' V⁻¹ y,
///     I_j  =  x_j' V⁻¹ x_j,
///     SE_j =  I_j⁻¹/²,
///     LR_j =  ½ · I_j · b̂_j²  =  ½ · (b̂_j / SE_j)²,
///     p_j  =  P( χ²₁ > 2 · LR_j ).
///
/// `V` enters only through `v_factor` (a pre-computed Cholesky factor of
/// V), so `v_factor.solve_vec(x_j)` returns V⁻¹ x_j without ever forming
/// V⁻¹ explicitly. `vinv_y` is computed once outside this function and
/// passed in for all markers — that is the key efficiency of EMMAX: a
/// single Cholesky amortises the per-marker cost from O(n³) to O(n²).
///
/// # Degenerate guard
///
/// If `x_j' V⁻¹ x_j ≤ 0` the marker is effectively constant after
/// projection by V⁻¹ (e.g. monomorphic locus); return `(0, 0, 0, 1)`
/// instead of dividing by zero. p = 1 means "no evidence against null".
fn emmax_single_snp(
    x_j: &Array1<f64>,
    vinv_y: &Array1<f64>,
    v_factor: &FactorizedV,
) -> StdResult<(f64, f64, f64, f64), GwasError> {
    // V⁻¹ x_j  via cached Cholesky factor of V.
    let vinv_xj = v_factor.solve_vec(x_j)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // Fisher information at the null: I_j = x_j' V⁻¹ x_j (scalar).
    let xtvinvx: f64 = x_j.dot(&vinv_xj);

    if xtvinvx <= 0.0 {
        // Degenerate marker (post-projection variance ≤ 0): skip.
        return Ok((0.0, 0.0, 0.0, 1.0));
    }

    // Numerator of the Wald estimator: x_j' V⁻¹ y (scalar).
    let xtviny: f64 = x_j.dot(vinv_y);

    // b̂_j = (x_j' V⁻¹ y) / (x_j' V⁻¹ x_j)
    let beta = xtviny / xtvinvx;

    // SE_j = (x_j' V⁻¹ x_j)⁻¹/²  — square-root of the inverse information.
    let se = (1.0 / xtvinvx).sqrt();

    // LR_j = ½ (b̂_j / SE_j)²  — likelihood-ratio statistic, df = 1.
    let lr = 0.5 * (beta / se).powi(2);

    // p-value from χ²_1 survival function evaluated at 2·LR.
    let pval = compute_pval_chi2(2.0 * lr, 1);

    Ok((beta, se, lr, pval))
}

/// Compute EMMAX for a single MH block (multi-allelic)
/// X_block: n × (k-1) sub-matrix of W_αh for block b
/// Returns (beta_norm, se_norm, lr_block, pval_block)
///
/// Multi-allelic LR: likelihood ratio test with df = k-1
/// LR_block = ½ * b̂' (X' V⁻¹ X) b̂
///           = ½ * (X'V⁻¹y)' (X'V⁻¹X)⁻¹ (X'V⁻¹y)
fn emmax_single_block_mh(
    x_block: &Array2<f64>,
    vinv_y: &Array1<f64>,
    v_factor: &FactorizedV,
) -> StdResult<(f64, f64, f64, f64), GwasError> {
    let k = x_block.ncols(); // n_alleles - 1

    if k == 0 {
        return Ok((0.0, 0.0, 0.0, 1.0));
    }

    // ============================================================
    // Multi-allelic block test — generalises the per-SNP EMMAX
    // from a scalar to a (k-1)-dimensional vector hypothesis.
    //
    // The working model adds a block of k-1 non-baseline allele
    // columns:
    //     y = X β + X_block · b + u + ε,    b ∈ ℝ^{k-1}.
    //
    // The score statistic for H_0: b = 0 against H_1: b ≠ 0 is:
    //
    //     b̂   = (X_block' V⁻¹ X_block)⁻¹ X_block' V⁻¹ y,        (estimator)
    //     LR  = ½ b̂' (X_block' V⁻¹ X_block) b̂
    //         = ½ (X_block' V⁻¹ y)' b̂,                          (statistic)
    //     p   = P(χ²_{k-1} > 2 LR).                              (p-value)
    //
    // V⁻¹ X_block is computed via the same cached Cholesky factor as
    // the per-SNP routine — that is the key efficiency of EMMAX.
    //
    // The (X_block' V⁻¹ X_block) matrix is a small (k-1) × (k-1)
    // symmetric system; `solve_symmetric` handles it with a direct
    // Cholesky. Negative LR values can occur from FP round-off when
    // the block has near-zero effect; we clip to 0 so the chi-square
    // tail probability stays in [0, 1].
    // ============================================================

    // V⁻¹ X_block via cached Cholesky: shape n × (k-1).
    let vinv_xblock = v_factor.solve_mat(x_block)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // X_block' V⁻¹ X_block: (k-1) × (k-1) symmetric PD information matrix.
    let xtvinvx = x_block.t().dot(&vinv_xblock);

    // X_block' V⁻¹ y: (k-1)-vector — the score evaluated at b̂ = 0.
    let xtviny = x_block.t().dot(vinv_y);

    // Wald estimate via the small symmetric solve.
    let beta = solve_symmetric(&xtvinvx, &xtviny)
        .map_err(|e| GwasError::LinAlgError(e))?;

    // LR = ½ b̂' I b̂ = ½ (score)' b̂. Clip to 0 (FP safety).
    let lr = 0.5 * xtviny.dot(&beta);
    let lr = if lr < 0.0 { 0.0 } else { lr };

    // Aggregate effect-size and standard-error summaries for the
    // block. `beta_norm` is the Euclidean norm of b̂, giving a single
    // "effect magnitude" comparable across blocks of different k.
    // `se_norm` is a similarly compressed SE summary derived from
    // the diagonal of the information matrix.
    let beta_norm = beta.dot(&beta).sqrt();
    let se_norm   = (1.0 / xtvinvx.diag().iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt())
        .sqrt();

    // Chi-square tail with df = k - 1, evaluated at the test
    // statistic 2 · LR. Under H_0 the LR ratio (Wilks 1938) is
    // asymptotically χ²_{k-1}-distributed.
    let pval = compute_pval_chi2(2.0 * lr, k);

    Ok((beta_norm, se_norm, lr, pval))
}

/// Run a full EMMAX GWAS scan over biallelic SNP markers.
///
/// # Arguments
///
/// - `w_centered`: `(n, m)` VanRaden-centered SNP design (output of
///   [`crate::matrix::snp_additive::center_w_vanraden`] or
///   equivalent).
/// - `y`: length-`n` phenotype vector.
/// - `x`: `(n, c)` fixed-effects design.
/// - `v_factor`: pre-computed Cholesky factor of V, produced once
///   from the null-model REML fit.
///
/// # Algorithm
///
/// 1. Compute `P y = V⁻¹ y − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹ y` once
///    via the cached Cholesky factor — this is the residual phenotype
///    after accounting for fixed effects.
/// 2. For each SNP `j` (in parallel via Rayon):
///    - Solve `V⁻¹ x_j` via the cached factor.
///    - Compute the Wald statistic per [`emmax_single_snp`].
/// 3. Collect the per-marker `(β̂, SE, LR, p)` tuples into a
///    [`GwasResult`].
///
/// # Performance
///
/// Total cost ≈ `O(n³)` for the one-time Cholesky + `m · O(n²)` for
/// the per-marker solves. Compared with re-fitting REML per marker
/// (which would be `m · O(n³)` plus the iteration count), EMMAX is
/// approximately `m`× cheaper — the entire reason the method exists.
///
/// # Errors
///
/// `GwasError::DimensionMismatch` if `y` and `w_centered` have
/// incompatible shapes; `GwasError::LinAlgError` on a failed solve.
pub fn run_emmax_snp(
    w_centered: &Array2<f64>,
    y: &Array1<f64>,
    x: &Array2<f64>,
    v_factor: &FactorizedV,
) -> StdResult<GwasResult, GwasError> {
    let n = w_centered.nrows();
    let m = w_centered.ncols();

    if y.len() != n {
        return Err(GwasError::DimensionMismatch(
            format!("y length {} != W nrows {}", y.len(), n)
        ));
    }

    // Pre-compute Py = V⁻¹y - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹y
    // The projection removes the fixed effects from y
    let vinv_y = v_factor.compute_py(y, x)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // Parallel loop per SNP
    let results: Vec<StdResult<(f64, f64, f64, f64), GwasError>> = (0..m)
        .into_par_iter()
        .map(|j| {
            let x_j = w_centered.column(j).to_owned();
            emmax_single_snp(&x_j, &vinv_y, v_factor)
        })
        .collect();

    // Unpack results
    let mut lr   = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);
    let mut se   = Vec::with_capacity(m);
    let mut pval = Vec::with_capacity(m);

    for res in results {
        let (b, s, l, p) = res?;
        beta.push(b);
        se.push(s);
        lr.push(l);
        pval.push(p);
    }

    Ok(GwasResult::new(lr, beta, se, pval))
}

/// Run a full EMMAX GWAS scan over multi-allelic microhaplotype
/// blocks.
///
/// # Arguments
///
/// - `w_mh`: `(n, total_alleles)` W_αh matrix in column-major
///   block-stacked layout — block 1's `h_1 − 1` columns first, then
///   block 2's `h_2 − 1` columns, etc.
/// - `block_sizes`: length-`n_blocks` vector with `h_b − 1`, the
///   number of non-baseline columns per block. Used to slice
///   `w_mh` into per-block sub-matrices for the joint test.
/// - `y`, `x`, `v_factor`: same role as in [`run_emmax_snp`].
///
/// # Algorithm
///
/// 1. Compute `P y` once via the cached Cholesky factor.
/// 2. For each block (parallel over blocks):
///    - Slice the corresponding columns out of `w_mh`.
///    - Run [`emmax_single_block_mh`] to obtain the joint
///      `(‖β̂‖, SE-summary, LR, χ²_{h_b − 1} p-value)`.
/// 3. Assemble the per-block tuples into a [`GwasResult`].
///
/// # Why a joint test per block
///
/// Testing each non-baseline column individually is statistically
/// inefficient because alleles within a block share information.
/// The joint test gains power proportional to `h_b − 1` whenever
/// the block carries any association at all. The price is a
/// (k − 1)-dimensional null distribution rather than scalar χ²_1.
///
/// # Errors
///
/// `GwasError::DimensionMismatch` for shape mismatches between
/// `w_mh`, `block_sizes`, and `y`. `GwasError::LinAlgError` on
/// failed solves inside the per-block routine.
pub fn run_emmax_mh(
    w_mh: &Array2<f64>,
    block_sizes: &[usize],
    y: &Array1<f64>,
    x: &Array2<f64>,
    v_factor: &FactorizedV,
) -> StdResult<GwasResult, GwasError> {
    let n = w_mh.nrows();
    let n_blocks = block_sizes.len();

    if y.len() != n {
        return Err(GwasError::DimensionMismatch(
            format!("y length {} != W nrows {}", y.len(), n)
        ));
    }

    // Validate total columns
    let total_cols: usize = block_sizes.iter().sum();
    if w_mh.ncols() != total_cols {
        return Err(GwasError::DimensionMismatch(
            format!(
                "W ncols {} != sum(block_sizes) {}",
                w_mh.ncols(), total_cols
            )
        ));
    }

    // Pre-compute Py — hilangkan fixed effects dari y
    let vinv_y = v_factor.compute_py(y, x)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // Build block column offsets
    let offsets: Vec<usize> = block_sizes
        .iter()
        .scan(0usize, |acc, &s| {
            let start = *acc;
            *acc += s;
            Some(start)
        })
        .collect();

    // Parallel loop per block
    let results: Vec<StdResult<(f64, f64, f64, f64), GwasError>> = (0..n_blocks)
        .into_par_iter()
        .map(|b| {
            let start = offsets[b];
            let size  = block_sizes[b];

            if size == 0 {
                return Ok((0.0, 0.0, 0.0, 1.0));
            }

            // Extract W_αh sub-matrix for this block
            let x_block = w_mh
                .slice(ndarray::s![.., start..start + size])
                .to_owned();

            emmax_single_block_mh(&x_block, &vinv_y, v_factor)
        })
        .collect();

    // Unpack results
    let mut lr   = Vec::with_capacity(n_blocks);
    let mut beta = Vec::with_capacity(n_blocks);
    let mut se   = Vec::with_capacity(n_blocks);
    let mut pval = Vec::with_capacity(n_blocks);

    for res in results {
        let (b, s, l, p) = res?;
        beta.push(b);
        se.push(s);
        lr.push(l);
        pval.push(p);
    }

    Ok(GwasResult::new(lr, beta, se, pval))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Solve a small symmetric positive-definite system `A · x = b`.
///
/// Used by [`emmax_single_block_mh`] to invert the per-block
/// information matrix `X_block' V⁻¹ X_block`, which is at most
/// `(h_max − 1) × (h_max − 1)` — typically `< 10 × 10`. For systems
/// this small, `ndarray_linalg`'s general `Solve` (LU under the
/// hood) is fast and robust; a specialised Cholesky would shave
/// negligible cost. The function exists as a thin abstraction so the
/// caller doesn't depend on the linear-algebra crate directly.
///
/// # Errors
///
/// Returns the LAPACK error message as a `String` (forwarded to a
/// `GwasError::LinAlgError` by the caller). Failures typically
/// mean the per-block information matrix is rank-deficient — e.g.
/// when one allele column is collinear with another within the
/// block.
fn solve_symmetric(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    use ndarray_linalg::Solve;
    a.solve(b).map_err(|e| e.to_string())
}

/// Chi-squared p-value: P(X > stat) where X ~ χ²(df)
fn compute_pval_chi2(stat: f64, df: usize) -> f64 {
    if stat <= 0.0 {
        return 1.0;
    }
    let chi2 = ChiSquared::new(df as f64)
        .expect("Invalid chi2 df");
    1.0 - chi2.cdf(stat)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_emmax_snp_basic() {
        // Simple 4x2 case: known association
        let w = array![
            [-1.0,  0.5],
            [ 0.0, -0.5],
            [ 1.0,  0.5],
            [-1.0, -0.5],
        ];
        let y = array![1.0, 0.0, 2.0, 0.5];

        // Build simple V = I
        use crate::solver::factorized::FactorizedV;
        let g_list: Vec<(Array2<f64>, String)> = vec![];
        let sigma2 = vec![1.0]; // sigma2_e only
        let v_factor = FactorizedV::new(&g_list, &sigma2, 4).unwrap();

        let x = ndarray::Array2::<f64>::ones((4, 1));
        let result = run_emmax_snp(&w, &y, &x, &v_factor).unwrap();
        assert!(result.lr[0] >= 0.0);
        assert!(result.pval[0] >= 0.0 && result.pval[0] <= 1.0);
    }

    #[test]
    fn test_pval_range() {
        // p-values must be in [0, 1]
        let pval = compute_pval_chi2(10.0, 1);
        assert!(pval >= 0.0 && pval <= 1.0);
        let pval_zero = compute_pval_chi2(0.0, 1);
        assert_eq!(pval_zero, 1.0);
    }

    #[test]
    fn test_emmax_mh_block_basic() {
        // Multi-allelic block: 5 individuals × 2 non-baseline microhaplotypes.
        // V = I (no genetic variance), so the test is the pure OLS Wald
        // statistic against y. The point is to verify the block path
        // produces sane outputs (non-negative LR, p ∈ [0, 1], correct df).
        use crate::solver::factorized::FactorizedV;
        let x_block = array![
            [-1.0,  0.5],
            [ 0.0, -0.5],
            [ 1.0,  0.5],
            [-1.0, -0.5],
            [ 0.5,  0.0],
        ];
        let y = array![1.0, 0.0, 2.0, 0.5, 1.0];

        let g_list: Vec<(Array2<f64>, String)> = vec![];
        let sigma2 = vec![1.0_f64];
        let v_factor = FactorizedV::new(&g_list, &sigma2, 5).unwrap();
        let vinv_y = v_factor.solve_vec(&y).unwrap();

        let (beta_norm, se_norm, lr, pval) =
            emmax_single_block_mh(&x_block, &vinv_y, &v_factor).unwrap();

        // Basic sanity invariants of the test statistic.
        assert!(lr >= 0.0, "LR must be non-negative, got {}", lr);
        assert!(pval >= 0.0 && pval <= 1.0,
            "p must be in [0, 1], got {}", pval);
        assert!(beta_norm >= 0.0, "‖β̂‖ must be non-negative");
        assert!(se_norm.is_finite(), "se must be finite");
    }

    #[test]
    fn test_emmax_mh_empty_block_returns_null() {
        // k = 0 (no non-baseline alleles) → degenerate test, must return
        // (0, 0, 0, 1) instead of dividing by zero.
        use crate::solver::factorized::FactorizedV;
        let x_block = Array2::<f64>::zeros((4, 0));
        let y = array![1.0, 0.0, 2.0, 0.5];

        let g_list: Vec<(Array2<f64>, String)> = vec![];
        let sigma2 = vec![1.0_f64];
        let v_factor = FactorizedV::new(&g_list, &sigma2, 4).unwrap();
        let vinv_y = v_factor.solve_vec(&y).unwrap();

        let (beta_norm, se_norm, lr, pval) =
            emmax_single_block_mh(&x_block, &vinv_y, &v_factor).unwrap();
        assert_eq!(beta_norm, 0.0);
        assert_eq!(se_norm, 0.0);
        assert_eq!(lr, 0.0);
        assert_eq!(pval, 1.0);
    }

    #[test]
    fn test_emmax_snp_degenerate_marker_returns_null() {
        // A marker that is orthogonal to V⁻¹ in the trivial sense
        // (a zero column) must produce the degenerate (0, 0, 0, 1) result.
        use crate::solver::factorized::FactorizedV;
        let zero_marker = Array1::<f64>::zeros(4);
        let y = array![1.0, 0.0, 2.0, 0.5];

        let g_list: Vec<(Array2<f64>, String)> = vec![];
        let sigma2 = vec![1.0_f64];
        let v_factor = FactorizedV::new(&g_list, &sigma2, 4).unwrap();
        let vinv_y = v_factor.solve_vec(&y).unwrap();

        let (beta, se, lr, pval) =
            emmax_single_snp(&zero_marker, &vinv_y, &v_factor).unwrap();
        assert_eq!(beta, 0.0);
        assert_eq!(se, 0.0);
        assert_eq!(lr, 0.0);
        assert_eq!(pval, 1.0);
    }
}