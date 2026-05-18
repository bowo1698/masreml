// src/utils/linalg.rs

//! Shared linear algebra primitives.
//!
//! Thin wrappers around `ndarray_linalg` LAPACK bindings plus a few
//! REML-specific helpers that appear in more than one place:
//!
//! - `solve_matrix(V, B)` — solve $V X = B$ for matrix $B$ via LU/Cholesky
//!   under the hood; returns a `MatrixError` on failure so REML can
//!   classify the cause (singular $V$, dimension mismatch, etc.).
//! - `compute_py(V, X, y)` — compute $Py$ where
//!   $P = V^{-1} - V^{-1} X (X' V^{-1} X)^{-1} X' V^{-1}$, the projection
//!   used inside HE, AI, and EM-REML scoring functions.
//! - `log_det_cholesky(L)` — log-determinant via Cholesky factor, used to
//!   evaluate the REML log-likelihood.
//! - `set_num_threads(n)` — global rayon thread-pool configuration. Called
//!   once at the top of [`crate::reml::adaptive`] so all parallel sections
//!   honour the same thread count.
//!
//! Functions are deliberately small and unit-testable in isolation; they
//! exist here rather than inlined into their callers so all REML
//! algorithms share the same numerically tested implementations.

use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use rayon::ThreadPoolBuilder;
use crate::matrix::MatrixError;

/// Solve `V · X = B` for matrix right-hand side `B`.
///
/// Iterates over the columns of `B` and calls
/// `ndarray_linalg::Solve::solve` on each. `ndarray_linalg` chooses
/// LU or Cholesky internally depending on `V`'s structure (it does
/// not assume symmetry).
///
/// # Performance note
///
/// For repeated solves with the same `V`, prefer
/// [`crate::solver::factorized::FactorizedV`], which caches the
/// Cholesky factor once and reuses it across calls. This helper is
/// intended for one-shot solves where no factor is cached yet.
///
/// # Errors
///
/// Returns `MatrixError::NotPositiveDefinite` if any column solve
/// fails. The error label is not strictly accurate (LU also fails on
/// singular matrices, not just NPD), but it bridges to a single
/// upstream error variant for R-side reporting.
pub fn solve_matrix(
    v: &Array2<f64>,
    b: &Array2<f64>,
) -> std::result::Result<Array2<f64>, MatrixError> {
    let ncols = b.ncols();
    let nrows = b.nrows();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for j in 0..ncols {
        let col = b.column(j).to_owned();
        let sol = v.solve(&col)
            .map_err(|_| MatrixError::NotPositiveDefinite)?;
        out.column_mut(j).assign(&sol);
    }
    Ok(out)
}

/// Compute the REML projection
/// `P y = V⁻¹ y − V⁻¹ X · (X' V⁻¹ X)⁻¹ · X' V⁻¹ y`.
///
/// `P` is the standard residual-projection matrix used inside
/// HE, AI, and EM-REML scoring functions; it accounts for the
/// fixed-effects subspace spanned by `X` without ever forming
/// `P` itself (`P` is dense `n × n`, expensive to materialise).
///
/// # Algorithm
///
/// ```text
/// 1. φ_X = V⁻¹ X          (column-by-column linear solves)
/// 2. φ_y = V⁻¹ y          (single solve)
/// 3. β̂   = (X' φ_X)⁻¹ X' φ_y   (small c × c solve)
/// 4. P y = φ_y − φ_X · β̂.
/// ```
///
/// # Performance note
///
/// Each call re-factors `V` because this helper does not cache.
/// Inside a REML iteration, prefer `FactorizedV::compute_py` which
/// shares the Cholesky factor across all uses within the same
/// iteration (score, AI matrix, BLUP).
pub fn compute_py(
    v: &Array2<f64>,
    y: &Array1<f64>,
    x: &Array2<f64>,
) -> std::result::Result<Array1<f64>, MatrixError> {

    // Solve V * phi1 = X (column by column)
    let phi1 = solve_matrix(v, x)?;

    // Solve V * phi2 = y
    let phi2 = v.solve(y)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;

    // X'V^-1 X = X' * phi1
    let xtvinvx = x.t().dot(&phi1);

    // X'V^-1 y = X' * phi2
    let xtviny = x.t().dot(&phi2);

    // Solve (X'V^-1 X) * coef = X'V^-1 y
    let coef = xtvinvx.solve(&xtviny)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;

    // Py = phi2 - phi1 * coef
    Ok(phi2 - phi1.dot(&coef))
}

/// Compute `log |V|` via Cholesky.
///
/// # Identity
///
/// If `V = L · L'` is the Cholesky factorisation, then
///
/// ```text
/// |V| = |L · L'| = |L|² = (Π_i L_{ii})²,
/// log |V| = 2 · Σ_i log(L_{ii}).
/// ```
///
/// This is more numerically stable than computing `|V|` from the
/// product of diagonal entries: summing logs avoids overflow /
/// underflow for the determinants of large or near-singular V that
/// appear in REML log-likelihood evaluations.
///
/// # Errors
///
/// `MatrixError::NotPositiveDefinite` if `V` is not positive
/// definite (Cholesky requires PD; a near-zero pivot here usually
/// signals a degenerate variance-component combination).
pub fn log_det_cholesky(v: &Array2<f64>) -> std::result::Result<f64, MatrixError> {
    use ndarray_linalg::{Cholesky, UPLO};
    let l = v.cholesky(UPLO::Lower)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;
    let log_det = l.diag()
        .iter()
        .map(|&x: &f64| x.ln())
        .sum::<f64>() * 2.0;
    Ok(log_det)
}

/// Configure the global thread count for both Rayon and the BLAS
/// backend used by `ndarray_linalg`.
///
/// # Why both
///
/// - **Rayon** parallelises the high-level loops in `masreml`
///   (pairwise products in HE regression, per-marker EMMAX,
///   per-column linear solves). Without an explicit setting, Rayon
///   uses all available CPU cores.
/// - **OpenBLAS** parallelises every dense matrix operation
///   underneath `ndarray_linalg`. Without an explicit setting, it
///   also defaults to all cores.
///
/// If both default to "all cores", the two thread pools end up
/// competing for cores (oversubscription), which on macOS can drop
/// throughput by 5–20 % and on Linux can cause severe contention
/// for the BLAS lock. Calling this once at the top of the REML
/// dispatcher pins both pools to the same explicit count so they
/// cooperate rather than compete.
///
/// # Mechanism for OpenBLAS
///
/// OpenBLAS reads `OPENBLAS_NUM_THREADS` from the process
/// environment at runtime; we set the env var so subsequent BLAS
/// calls honour it. This is a global per-process setting — no way
/// to set it per-call without rebuilding OpenBLAS.
pub fn set_num_threads(n_threads: usize) {
    // Rayon global thread pool (idempotent: errors silently if
    // already built, which happens after the first call).
    let _ = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global();

    // OpenBLAS picks this up at runtime via the env variable.
    std::env::set_var("OPENBLAS_NUM_THREADS", n_threads.to_string());
}

// ============================================================================
// Unit tests
// ============================================================================
//
// Deterministic checks of the linear-algebra primitives that drive
// REML scoring and the BLUP solve. Run with `cargo test`.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn solve_matrix_identity_returns_input() {
        // V = I  ⇒  solve_matrix(I, B) = B.
        let v = Array2::<f64>::eye(3);
        let b = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];
        let x = solve_matrix(&v, &b).expect("solve");
        for i in 0..3 {
            for j in 0..2 {
                assert!((x[[i, j]] - b[[i, j]]).abs() < 1e-12,
                    "x[{}, {}] = {} ≠ b = {}", i, j, x[[i, j]], b[[i, j]]);
            }
        }
    }

    #[test]
    fn solve_matrix_round_trip() {
        // For invertible V: V · solve_matrix(V, B) should equal B
        // up to floating-point round-off.
        let v = array![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ];
        let b = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ];
        let x = solve_matrix(&v, &b).expect("solve");
        let bb = v.dot(&x);
        for i in 0..3 {
            for j in 0..2 {
                assert!((bb[[i, j]] - b[[i, j]]).abs() < 1e-10,
                    "round-trip drift at ({}, {}): {} vs {}",
                    i, j, bb[[i, j]], b[[i, j]]);
            }
        }
    }

    #[test]
    fn log_det_matches_direct_for_identity() {
        // log |I| = 0.
        let v = Array2::<f64>::eye(4);
        let ld = log_det_cholesky(&v).expect("log det");
        assert!(ld.abs() < 1e-12, "log|I| should be 0, got {}", ld);
    }

    #[test]
    fn log_det_matches_direct_for_diagonal() {
        // V = diag(d) ⇒ log|V| = Σ log d_i.
        let d = [2.0_f64, 3.0, 5.0, 7.0];
        let mut v = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            v[[i, i]] = d[i];
        }
        let ld = log_det_cholesky(&v).expect("log det");
        let expected: f64 = d.iter().map(|x| x.ln()).sum();
        assert!((ld - expected).abs() < 1e-12,
            "log|diag| = {} (expected {})", ld, expected);
    }

    #[test]
    fn log_det_rejects_non_pd() {
        // Indefinite matrix: Cholesky fails.
        let v = array![
            [1.0, 0.0],
            [0.0, -1.0],
        ];
        let result = log_det_cholesky(&v);
        assert!(result.is_err(), "expected NotPositiveDefinite error");
    }

    #[test]
    fn compute_py_zero_y_returns_zero() {
        // P · 0 = 0 regardless of V or X.
        let v = Array2::<f64>::eye(3);
        let x = array![[1.0], [1.0], [1.0]];
        let y = Array1::<f64>::zeros(3);
        let py = compute_py(&v, &y, &x).expect("compute_py");
        for i in 0..3 {
            assert!(py[i].abs() < 1e-12);
        }
    }
}