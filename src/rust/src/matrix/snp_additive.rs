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

/// Calculate allele frequency from W raw (0/1/2 coding)
/// p_j = mean(W_j) / 2
fn compute_allele_freq(w_raw: &Array2<f64>) -> Vec<f64> {
    let n = w_raw.nrows() as f64;
    (0..w_raw.ncols())
        .map(|j| {
            let col_sum: f64 = w_raw.column(j).iter().sum();
            col_sum / (2.0 * n)
        })
        .collect()
}

/// VanRaden centering: z_ij = x_ij - 2*p_j
/// Input: W_raw (n × m), nilai 0/1/2
/// Output: W_centered (n × m)
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

/// Build SNP additive G matrix (VanRaden 2008)
/// G = WW' / k, k = tr(WW')/n = 2 * sum(p_j * (1 - p_j))
///
/// Input: w_raw (n × m), raw genotype 0/1/2
/// Output: GMatrix { g: n×n, k, n, m }
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