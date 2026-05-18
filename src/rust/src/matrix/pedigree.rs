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

        // Off-diagonal: a_ij for j < i, Henderson (1976) recursion.
        //
        // The kinship between i (younger) and j (older) is the average
        // kinship of j with each parent of i:
        //
        //     a_{i, j} = 0.5 · ( a_{j, sire_i} + a_{j, dam_i} ),   j < i.
        //
        // Using parents of i (not j) is essential: if j is a founder
        // with unknown parents, a recursion in terms of j's parents
        // would collapse to zero even when i is a direct descendant of
        // j. Both sire_i − 1 and dam_i − 1 are < i because the pedigree
        // is required to be topologically pre-sorted, so the entries
        // a[j, sire_i − 1] and a[j, dam_i − 1] have already been filled
        // by the time we reach row i.
        for j in 0..i {
            let a_j_si = if si > 0 {
                a[[j.max(si - 1), j.min(si - 1)]]
            } else {
                0.0
            };
            let a_j_di = if di > 0 {
                a[[j.max(di - 1), j.min(di - 1)]]
            } else {
                0.0
            };

            let val = 0.5 * (a_j_si + a_j_di);
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

// ============================================================================
// Unit tests
// ============================================================================
//
// Henderson (1976) recursion on small pedigrees where the expected A matrix
// can be derived by hand. Tests verify both the diagonal (inbreeding) and
// off-diagonal (kinship) entries.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_founder_yields_identity() {
        // n = 1, no parents: A = [[1]].
        let a = build_a_ped_internal(&[0], &[0], 1).expect("build A");
        assert_eq!(a.g.nrows(), 1);
        assert!((a.g[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn full_sib_family_diagonal_and_symmetry() {
        // Pedigree:
        //   1 — founder (sire = 0, dam = 0)
        //   2 — founder (sire = 0, dam = 0)
        //   3 — offspring of (1, 2)
        //   4 — offspring of (1, 2)  (full sib of 3)
        //
        // This test only verifies the parts of A that the current
        // recursion gets right: diagonal entries (1.0 with no inbreeding)
        // and full-matrix symmetry. The off-diagonal parent-offspring
        // entries are tested separately in `full_sib_offspring_kinship`
        // which is currently `#[ignore]`d pending a fix to the off-
        // diagonal recursion (see that test's docstring).
        let sire = [0, 0, 1, 1];
        let dam  = [0, 0, 2, 2];
        let a = build_a_ped_internal(&sire, &dam, 4).expect("build A");

        // Diagonals: 1.0 (no inbreeding, parents unrelated for inds 2-3).
        for i in 0..4 {
            assert!((a.g[[i, i]] - 1.0).abs() < 1e-12,
                "diag[{}] = {}", i, a.g[[i, i]]);
        }
        // Symmetry — must always hold regardless of recursion details.
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!((a.g[[i, j]] - a.g[[j, i]]).abs() < 1e-12,
                    "asymmetry at ({}, {})", i, j);
            }
        }
    }

    /// Off-diagonal Henderson (1976) kinship check.
    ///
    /// Verifies the corrected recursion
    /// `a_ij = 0.5 · (a_{j, sire_i} + a_{j, dam_i})` on a full-sib
    /// family with two unrelated founder parents. Parent–offspring
    /// pairs should give 0.5, founders should be uncorrelated, full
    /// sibs should share 0.5 of their additive variance.
    #[test]
    fn full_sib_offspring_kinship() {
        let sire = [0, 0, 1, 1];
        let dam  = [0, 0, 2, 2];
        let a = build_a_ped_internal(&sire, &dam, 4).expect("build A");

        // Founders unrelated.
        assert!((a.g[[0, 1]] - 0.0).abs() < 1e-12);
        // Parent–offspring should be 0.5 in each direction.
        assert!((a.g[[0, 2]] - 0.5).abs() < 1e-12);
        assert!((a.g[[1, 2]] - 0.5).abs() < 1e-12);
        assert!((a.g[[0, 3]] - 0.5).abs() < 1e-12);
        assert!((a.g[[1, 3]] - 0.5).abs() < 1e-12);
        // Full sibs share half their additive variance: a_34 = 0.5.
        assert!((a.g[[2, 3]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn inbreeding_from_self_mating_is_detected() {
        // Pedigree:
        //   1 — founder
        //   2 — founder
        //   3 — offspring of (1, 2)
        //   4 — offspring of (3, 3)   (selfing)
        //
        // F_4 = 0.5 · A[3, 3] = 0.5,   so A[4, 4] = 1 + 0.5 = 1.5.
        let sire = [0, 0, 1, 3];
        let dam  = [0, 0, 2, 3];
        let a = build_a_ped_internal(&sire, &dam, 4).expect("build A");
        assert!((a.g[[3, 3]] - 1.5).abs() < 1e-12,
            "A[3,3] = {} (expected 1.5)", a.g[[3, 3]]);
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        // sire and dam length must equal n.
        let result = build_a_ped_internal(&[0, 0], &[0], 2);
        assert!(result.is_err());
    }
}