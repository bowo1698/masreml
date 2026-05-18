// src/solver/cholesky.rs

//! Cholesky-based EBV solver.
//!
//! Thin wrapper around [`super::factorized::FactorizedV`] that exposes a
//! `solve_cholesky_internal(...)` entry point matching the signature
//! expected by [`super::solve_ebv`]. The heavy lifting (factorisation +
//! triangular solves) is in the `factorized` module so it can be reused
//! by other consumers such as [`crate::gwas::emmax`].
//!
//! Direct factorisation is exact (no convergence tolerance) and produces
//! BLUPs in a single pass, but memory scales as $O(n^2)$ for the factor
//! itself; for $n \gtrsim 10{,}000$ prefer the iterative
//! [`super::pcg`] solver.

use ndarray::{Array1, Array2};

use super::{BlupResult, SolverError, StdResult};
use super::factorized::FactorizedV;

/// Solve the mixed-model equations via Cholesky factorisation.
///
/// Wraps [`super::factorized::FactorizedV`] to factor V once and
/// then apply the cached factor for both `V⁻¹ y` and `V⁻¹ X`. The
/// returned [`BlupResult`] contains per-component EBVs, the
/// fixed-effect coefficients, and `solver = "cholesky"` for
/// diagnostic identification.
///
/// # Complexity
///
/// `O(n³)` for the one-time factorisation, then `O(n²)` for each
/// downstream solve. The `n_random` parameter is accepted but
/// unused at this level (it is bundled into `g_list` already).
pub fn solve_cholesky_internal(
    y: &Array1<f64>,
    x: &Array2<f64>,
    g_list: &[(Array2<f64>, String)],
    sigma2: &[f64],
    n: usize,
    _n_random: usize,
) -> StdResult<BlupResult, SolverError> {
    let factor = FactorizedV::new(g_list, sigma2, n)?;
    factor.solve_blup(y, x, g_list, sigma2)
}

// ============================================================================
// Unit tests
// ============================================================================
//
// Round-trip tests that confirm V⁻¹ y is computed correctly by the
// cached Cholesky path. Run with `cargo test`.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn factorized_v_solves_identity_y() {
        // V = I ⇒ FactorizedV.solve_vec(y) = y.
        let g_list: Vec<(Array2<f64>, String)> = vec![
            (Array2::<f64>::eye(3), "g".to_string()),
        ];
        let sigma2 = vec![0.0_f64, 1.0]; // σ²_g = 0, σ²_e = 1 ⇒ V = I.
        let factor = FactorizedV::new(&g_list, &sigma2, 3).expect("factor V");
        let y = array![1.0, 2.0, 3.0];
        let x = factor.solve_vec(&y).expect("solve");
        for i in 0..3 {
            assert!((x[i] - y[i]).abs() < 1e-12,
                "expected y[{}] = {}, got {}", i, y[i], x[i]);
        }
    }

    #[test]
    fn factorized_v_round_trip() {
        // For an invertible V, V · (V⁻¹ y) should reproduce y.
        let g = array![
            [2.0, 0.5, 0.0],
            [0.5, 2.0, 0.5],
            [0.0, 0.5, 2.0],
        ];
        let g_list: Vec<(Array2<f64>, String)> = vec![(g.clone(), "g".to_string())];
        let sigma2 = vec![1.0_f64, 0.5];
        let n = 3;
        let factor = FactorizedV::new(&g_list, &sigma2, n).expect("factor V");

        let y = array![1.0, -1.0, 2.0];
        let x = factor.solve_vec(&y).expect("solve");

        // Rebuild V manually and check V · x ≈ y.
        let v = &g * sigma2[0] + Array2::<f64>::eye(n) * sigma2[1];
        let vy = v.dot(&x);
        for i in 0..n {
            assert!((vy[i] - y[i]).abs() < 1e-10,
                "round-trip drift at {}: {} vs {}", i, vy[i], y[i]);
        }
    }

    #[test]
    fn factorized_v_rejects_negative_variance() {
        // σ² < 0 ⇒ V loses positive definiteness ⇒ factorisation fails.
        let g_list: Vec<(Array2<f64>, String)> = vec![
            (Array2::<f64>::eye(3), "g".to_string()),
        ];
        // σ²_g = 1, σ²_e = -2 ⇒ V = G − 2 I = -I, not PD.
        let sigma2 = vec![1.0_f64, -2.0];
        let result = FactorizedV::new(&g_list, &sigma2, 3);
        assert!(result.is_err(), "expected NotPositiveDefinite error");
    }
}