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

/// Solve EBV via Cholesky factorization
/// Thin wrapper around FactorizedV
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