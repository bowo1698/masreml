//! Pedigree-based numerator relationship matrix $A$ (Henderson 1976).
//!
//! Builds the $n \times n$ expected relationship matrix from a recoded
//! pedigree using Henderson's recursive formula:
//!
//! ```text
//! a_ii = 1 + F_i,              F_i = 0.5 · a[sire_i, dam_i]
//! a_ij = 0.5 · (a[i, sire_j] + a[i, dam_j])   for i < j
//! ```
//!
//! Founders (sire = dam = 0) contribute $a_{ii} = 1$ and zero off-diagonals
//! to ancestors not in the recoded pedigree.
//!
//! ## Input format
//!
//! Animals must be **topologically pre-sorted** so that any parent appears
//! before its offspring. Sire and dam indices use 0 for unknown parent;
//! all other indices are 1-based references into the same pedigree array.
//! Validation of this ordering is the caller's responsibility — the
//! recursion silently produces wrong results if the order is violated.
//!
//! ## Memory
//!
//! The full $A$ matrix is dense $n \times n$ (the recursion does not
//! exploit sparsity). For large pedigrees, consider $A^{-1}$ sparse forms
//! (Quaas, 1976) — not implemented here but a natural extension.
//!
//! ## Reference
//!
//! Henderson, C. R. (1976). A simple method for computing the inverse of
//! a numerator relationship matrix used in prediction of breeding values.
//! *Biometrics*, 32:69–83.

use extendr_api::prelude::*;
use ndarray::Array2;

use super::{GMatrix, MatrixError, StdResult};

/// Build numerator relationship matrix A from pedigree
/// using Henderson (1976) recursive algorithm
///
/// Algorithm:
///   a_ii = 1 + F_i (F_i = inbreeding coefficient)
///   a_ij = 0.5 * (a[i, sire_j] + a[i, dam_j])  for i < j
///
/// Input:
///   sire: integer vector of sire indices (0 = unknown)
///   dam:  integer vector of dam indices  (0 = unknown)
///   n:    number of individuals
///
/// Output: GMatrix { g: A matrix (n×n), k=1.0, n, m=0 }
pub fn build_a_ped_internal(
    sire: &[i32],
    dam: &[i32],
    n: usize,
) -> StdResult<GMatrix, MatrixError> {
    if sire.len() != n || dam.len() != n {
        return Err(MatrixError::InvalidDimension(
            format!("sire/dam length {} must equal n {}", sire.len(), n)
        ));
    }

    let mut a = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        let si = sire[i] as usize; // 0 = unknown
        let di = dam[i] as usize;  // 0 = unknown

        // Diagonal: a_ii = 1 + F_i
        // F_i = 0.5 * a[si-1, di-1] if both parents known
        let f_i = if si > 0 && di > 0 {
            0.5 * a[[si - 1, di - 1]]
        } else {
            0.0
        };
        a[[i, i]] = 1.0 + f_i;

        // Off-diagonal: a_ij for j < i
        for j in 0..i {
            let sj = sire[j] as usize;
            let dj = dam[j] as usize;

            // a_ij = 0.5 * (a[i, sire_j] + a[i, dam_j])
            let a_i_sj = if sj > 0 { a[[i.max(sj-1), i.min(sj-1)]] } else { 0.0 };
            let a_i_dj = if dj > 0 { a[[i.max(dj-1), i.min(dj-1)]] } else { 0.0 };

            let val = 0.5 * (a_i_sj + a_i_dj);
            a[[i, j]] = val;
            a[[j, i]] = val;
        }
    }

    Ok(GMatrix::new(a))
}

/// Extendr entry point
/// Input:
///   sire: integer vector (1-based, 0 = unknown)
///   dam:  integer vector (1-based, 0 = unknown)
///   n:    number of individuals
/// Output: A matrix (n x n)
#[extendr]
pub fn build_a_ped(sire: &[i32], dam: &[i32], n: i32) -> Result<RMatrix<f64>> {
    let n_usize = n as usize;
    let amat = build_a_ped_internal(sire, dam, n_usize)
        .map_err(|e| Error::from(e.to_string()))?;

    let g_vec: Vec<f64> = amat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(n_usize, n_usize, |r, c| g_vec[r * n_usize + c]))
}