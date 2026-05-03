/// src/solver/cholesky.rs
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