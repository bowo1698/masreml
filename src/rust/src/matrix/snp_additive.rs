//! SNP additive relationship matrix (VanRaden 2008).
//!
//! Builds the additive genomic relationship matrix
//!
//! ```text
//! G = W W' / k,    k = 2 · sum_j p_j (1 - p_j)
//! ```
//!
//! where $W$ is the column-centered SNP matrix $W_{ij} = X_{ij} - 2 p_j$
//! with $X_{ij} \in \{0, 1, 2\}$ allele dosages and $p_j$ the reference
//! (training-set) allele frequency at marker $j$.
//!
//! ## GWABLUP support
//!
//! Optionally accepts a per-marker weight vector $d = (d_1, \dots, d_m)$
//! to construct a GWAS-weighted $G$ matrix:
//!
//! ```text
//! G_wa = sum_j d_j · w_j w_j' / sum_j d_j · 2 p_j (1 - p_j)
//! ```
//!
//! When `weights = None` this collapses to the standard VanRaden $G$.
//! Weights typically come from a smoothed likelihood-ratio statistic
//! (see [`crate::gwas::smoother`]).
//!
//! ## Parallelism
//!
//! The $WW^\top$ product is computed with `ndarray::parallel` row-chunked
//! across rayon threads. The scaling constant $k$ is computed via
//! [`crate::matrix::compute_k`] without materialising $WW^\top$ first.

use extendr_api::prelude::*;
use ndarray::{Array2, Axis};
use ndarray::parallel::prelude::*;

use super::{GMatrix, MatrixError, validate_w, StdResult};

/// Compute reference allele frequencies from the raw `{0, 1, 2}` SNP matrix.
///
/// Each entry `W_raw[i, j] ∈ {0, 1, 2}` records the allele dosage at marker
/// `j` for individual `i`. The maximum-likelihood frequency estimate per
/// marker (under independence and full data) is half of the column mean:
///
/// ```text
/// p_j = (1 / 2n) · Σ_i W_raw[i, j] = mean(W_raw[, j]) / 2
/// ```
///
/// This drives the VanRaden centering in [`center_w_vanraden`]: each column
/// gets shifted by `2 p_j` so the expected centered value is zero under
/// Hardy–Weinberg.
fn compute_allele_freq(w_raw: &Array2<f64>) -> Vec<f64> {
    let n = w_raw.nrows() as f64;
    (0..w_raw.ncols())
        .map(|j| {
            let col_sum: f64 = w_raw.column(j).iter().sum();
            col_sum / (2.0 * n)
        })
        .collect()
}

/// VanRaden (2008) column centering:
///
/// ```text
/// W_centered[i, j] = W_raw[i, j] − 2 p_j
/// ```
///
/// Under HWE the column expectation E[W_raw[·, j]] = 2 p_j, so the centered
/// matrix has zero column mean by construction. In finite samples the
/// empirical column mean is close to but not exactly zero — analogous to the
/// situation with the Da (2015) multi-allelic encoding.
///
/// The centering is parallelised per column using rayon: each marker is a
/// fully independent column update, so this scales linearly with thread
/// count up to memory-bandwidth limits.
fn center_w_vanraden(w_raw: &Array2<f64>, p: &[f64]) -> Array2<f64> {
    let (n, m) = w_raw.dim();
    let mut w_c = Array2::<f64>::zeros((n, m));

    // Paralel per kolom (per marker)
    w_c.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let pj = p[j];
            let center = 2.0 * pj;
            col.iter_mut()
                .zip(w_raw.column(j).iter())
                .for_each(|(out, &raw)| {
                    *out = raw - center;
                });
        });
    w_c
}

/// Build the SNP additive genomic relationship matrix (VanRaden 2008).
///
/// # Mathematical definition
///
/// Given raw `{0, 1, 2}` allele dosages `W_raw` of shape `(n, m)` and
/// reference allele frequencies `p_j = mean(W_raw[, j]) / 2`, the centered
/// matrix is `W_c[i, j] = W_raw[i, j] − 2 p_j`, and the additive G matrix is
///
/// ```text
/// G = W_c · W_c' / k,    k = tr(W_c W_c') / n = 2 · Σ_j p_j (1 − p_j)
/// ```
///
/// The normalisation `k` ensures that `mean(diag(G)) ≈ 1 + F` where `F` is
/// the average inbreeding coefficient. Under HWE with no inbreeding,
/// `mean(diag(G)) ≈ 1`, matching the pedigree-based numerator relationship
/// matrix convention.
///
/// # GWABLUP option
///
/// If `weights` is `Some(d)` with one entry per marker, each marker is
/// scaled by `√d_j` before forming the Gram, giving the GWAS-weighted
/// matrix
///
/// ```text
/// G_wa = Σ_j d_j · w_j w_j' / k_wa,    k_wa = 2 · Σ_j d_j p_j (1 − p_j).
/// ```
///
/// Setting all `d_j = 1` reduces to the unweighted VanRaden G.
///
/// # Inputs
///
/// - `w_raw`: `(n, m)` matrix of allele dosages.
/// - `weights`: optional length-`m` per-marker weights (GWABLUP).
/// - `allele_freq`: optional length-`m` reference frequencies. If `None`,
///   frequencies are computed from `w_raw` itself; if `Some`, they are used
///   verbatim (typical for test-set encoding using training frequencies).
///
/// # Errors
///
/// Returns `MatrixError::InvalidDimension` if dimensions mismatch or if
/// every marker is monomorphic (so `k = 0`).
pub fn build_g_snp_add_internal(
    w_raw: &Array2<f64>,
    weights: Option<&[f64]>,
    allele_freq: Option<&[f64]>, 
) -> StdResult<GMatrix, MatrixError> {
    validate_w(w_raw, "SNP additive W")?;

    let p = match allele_freq {
        Some(freq) => {
            if freq.len() != w_raw.ncols() {
                return Err(MatrixError::InvalidDimension(
                    format!(
                        "allele_freq length {} != ncols {}",
                        freq.len(), w_raw.ncols()
                    )
                ));
            }
            freq.to_vec()
        }
        None => compute_allele_freq(w_raw),
    };
    let w_c = center_w_vanraden(w_raw, &p);

    // Validate weights length if provided
    if let Some(d) = weights {
        if d.len() != w_raw.ncols() {
            return Err(MatrixError::InvalidDimension(
                format!(
                    "weights length {} != ncols {}",
                    d.len(), w_raw.ncols()
                )
            ));
        }
    }

    // k = 2 * sum(p_j * (1 - p_j))
    // always count from allele_freq (training of full)
    let k = 2.0 * p.iter().map(|&pj| pj * (1.0 - pj)).sum::<f64>();
    if k == 0.0 {
        return Err(MatrixError::InvalidDimension(
            "k = 0, all markers are monomorphic".to_string()
        ));
    }

    let g = match weights {
        Some(d) => {
            let w_scaled = scale_columns(&w_c, d);
            let k_weighted = 2.0 * p.iter()
                .zip(d.iter())
                .map(|(&pj, &dj)| pj * (1.0 - pj) * dj)
                .sum::<f64>();
            if k_weighted == 0.0 {
                return Err(MatrixError::InvalidDimension(
                    "k = 0 after weighting".to_string()
                ));
            }
            compute_gram_parallel(&w_scaled, k_weighted)
        }
        None => {
            compute_gram_parallel(&w_c, k)
        }
    };

    Ok(GMatrix::new(g))
}

/// Calculate WW'/k in parallel (symmetric, calculate lower triangle only)
pub fn compute_gram_parallel(w: &Array2<f64>, k: f64) -> Array2<f64> {
    let n = w.nrows();
    let mut g = Array2::<f64>::zeros((n, n));

    // Calculate the lower triangle parallels per row
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..=i)
                .map(|j| {
                    let dot: f64 = w.row(i)
                        .iter()
                        .zip(w.row(j).iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    dot / k
                })
                .collect()
        })
        .collect();

    // Contents of symmetric matrix
    for i in 0..n {
        for j in 0..=i {
            g[[i, j]] = rows[i][j];
            g[[j, i]] = rows[i][j];
        }
    }
    g
}

/// Scale columns of W by sqrt(weights)
/// W_scaled[:,j] = W[:,j] * sqrt(weights[j])
fn scale_columns(w: &Array2<f64>, weights: &[f64]) -> Array2<f64> {
    let mut ws = w.to_owned();
    ws.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let scale = weights[j].sqrt();
            col.iter_mut().for_each(|x| *x *= scale);
        });
    ws
}

/// Extendr entry point
/// Input R matrix: raw genotype 0/1/2 (n × m)
/// Output R matrix: G matrix (n × n)
#[extendr]
pub fn build_g_snp_add(
    w: RMatrix<f64>,
    weights: Nullable<&[f64]>,
    allele_freq: Nullable<&[f64]>, 
) -> Result<RMatrix<f64>> {
    let nrow = w.nrows();
    let ncol = w.ncols();
    let data: Vec<f64> = w.data().to_vec();
    let w_transposed = Array2::from_shape_vec((ncol, nrow), data)
        .map_err(|e| Error::from(e.to_string()))?;
    let w_arr = w_transposed.reversed_axes().to_owned();

    let w_opt: Option<Vec<f64>> = match weights {
        Nullable::NotNull(d) => Some(d.to_vec()),
        Nullable::Null       => None,
    };

    let freq_opt: Option<Vec<f64>> = match allele_freq {
        Nullable::NotNull(d) => Some(d.to_vec()),
        Nullable::Null       => None,
    };

    let gmat = build_g_snp_add_internal(
        &w_arr,
        w_opt.as_deref(),
        freq_opt.as_deref(),
    ).map_err(|e| Error::from(e.to_string()))?;

    let g_vec: Vec<f64> = gmat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(nrow, nrow, |r, c| g_vec[r * nrow + c]))
}

// ============================================================================
// Unit tests
// ============================================================================
//
// Deterministic checks of the VanRaden pipeline: allele-frequency
// computation, column centering, and the full G assembly. Run with
// `cargo test` or compile-verify with `cargo check --tests`.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn allele_freq_matches_column_mean_over_two() {
        // Two markers, three individuals each:
        //   marker 0 dosages: (0, 1, 2) → sum 3, p = 3 / (2·3) = 0.5
        //   marker 1 dosages: (2, 2, 1) → sum 5, p = 5 / 6 ≈ 0.8333
        let w = array![
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ];
        let p = compute_allele_freq(&w);
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn vanraden_centering_zeros_under_perfect_hwe_freq() {
        // If we pass the exact column mean / 2 as p_j, centered column
        // should have zero column mean (by construction).
        let w = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ];
        let p = compute_allele_freq(&w);
        let wc = center_w_vanraden(&w, &p);
        for j in 0..2 {
            let mean: f64 = wc.column(j).iter().sum::<f64>() / wc.nrows() as f64;
            assert!(mean.abs() < 1e-12,
                "column {} centered mean = {} (expected ≈ 0)", j, mean);
        }
    }

    #[test]
    fn g_diag_mean_close_to_one_under_hwe() {
        // VanRaden 2008: under HWE with no inbreeding, mean(diag(G)) ≈ 1.
        // Construct a small balanced dataset and confirm the property
        // holds up to finite-sample noise.
        let w = array![
            [0.0, 0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 0.0],
        ];
        let g = build_g_snp_add_internal(&w, None, None).expect("G build");
        let n = g.g.nrows();
        let mean_diag: f64 = (0..n).map(|i| g.g[[i, i]]).sum::<f64>() / n as f64;
        // Small sample: expect close to 1 but not exact.
        assert!(mean_diag > 0.5 && mean_diag < 2.0,
            "mean(diag(G)) = {} outside reasonable range", mean_diag);
    }

    #[test]
    fn g_is_symmetric() {
        let w = array![
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let g = build_g_snp_add_internal(&w, None, None).expect("G build");
        let n = g.g.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                assert!((g.g[[i, j]] - g.g[[j, i]]).abs() < 1e-12,
                    "G not symmetric at ({}, {}): {} vs {}",
                    i, j, g.g[[i, j]], g.g[[j, i]]);
            }
        }
    }

    #[test]
    fn monomorphic_markers_yield_error() {
        // All markers monomorphic ⇒ p_j ∈ {0, 1} ⇒ k = 0 ⇒ Err.
        let w = array![
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ];
        let result = build_g_snp_add_internal(&w, None, None);
        assert!(result.is_err(), "expected error on monomorphic input");
    }
}