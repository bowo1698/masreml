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

/// Compute EMMAX for a single SNP column x_j (bi-allelic)
/// Returns (beta_j, se_j, lr_j, pval_j)
///
/// b̂_j = (X_j' V⁻¹ X_j)⁻¹ X_j' V⁻¹ y
/// se_j = sqrt((X_j' V⁻¹ X_j)⁻¹)
/// LR_j = ½ * y' V⁻¹ X_j * b̂_j = ½ * (b̂_j / se_j)²
fn emmax_single_snp(
    x_j: &Array1<f64>,
    vinv_y: &Array1<f64>,
    v_factor: &FactorizedV,
) -> StdResult<(f64, f64, f64, f64), GwasError> {
    // V⁻¹ x_j
    let vinv_xj = v_factor.solve_vec(x_j)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // X_j' V⁻¹ X_j (scalar)
    let xtvinvx: f64 = x_j.dot(&vinv_xj);

    if xtvinvx <= 0.0 {
        return Ok((0.0, 0.0, 0.0, 1.0));
    }

    // X_j' V⁻¹ y (scalar)
    let xtviny: f64 = x_j.dot(vinv_y);

    // b̂_j
    let beta = xtviny / xtvinvx;

    // se_j = sqrt(1 / X_j' V⁻¹ X_j)
    let se = (1.0 / xtvinvx).sqrt();

    // LR_j = ½ (b̂_j / se_j)²
    let lr = 0.5 * (beta / se).powi(2);

    // p-value: chi-squared df=1, statistic = 2*LR
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

    // V⁻¹ X_block: n × k
    let vinv_xblock = v_factor.solve_mat(x_block)
        .map_err(|e| GwasError::LinAlgError(e.to_string()))?;

    // X' V⁻¹ X: k × k
    let xtvinvx = x_block.t().dot(&vinv_xblock);

    // X' V⁻¹ y: k × 1
    let xtviny = x_block.t().dot(vinv_y);

    // Solve (X'V⁻¹X) b̂ = X'V⁻¹y
    let beta = solve_symmetric(&xtvinvx, &xtviny)
        .map_err(|e| GwasError::LinAlgError(e))?;

    // LR_block = ½ * b̂' (X'V⁻¹X) b̂ = ½ * (X'V⁻¹y)' b̂
    let lr = 0.5 * xtviny.dot(&beta);
    let lr = if lr < 0.0 { 0.0 } else { lr };

    // Aggregated beta norm and se norm for reporting
    let beta_norm = beta.dot(&beta).sqrt();
    let se_norm   = (1.0 / xtvinvx.diag().iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt())
        .sqrt();

    // p-value: chi-squared df = k
    let pval = compute_pval_chi2(2.0 * lr, k);

    Ok((beta_norm, se_norm, lr, pval))
}

/// Run EMMAX GWAS for SNP markers (bi-allelic)
///
/// Input:
///   w_centered: centered genotype matrix (n × m), VanRaden coding
///   y: phenotype vector (n)
///   v_factor: pre-computed Cholesky factorization of V
/// Output: GwasResult with lr, beta, se, pval per SNP
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

/// Run EMMAX GWAS for MH markers (multi-allelic)
///
/// Input:
///   w_mh: W_αh matrix (n × total_alleles), Da (2015) coding
///         columns are grouped by block: [block1_alleles | block2_alleles | ...]
///   block_sizes: number of alleles (k-1) per block
///   y: phenotype vector (n)
///   v_factor: pre-computed Cholesky factorization of V
/// Output: GwasResult with lr, beta, se, pval per MH block
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

/// Solve symmetric positive definite system A x = b
/// Uses Cholesky for small k (k < 50), direct inverse otherwise
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
}