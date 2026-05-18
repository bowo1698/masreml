//! EBV/BLUP solvers given fitted variance components.
//!
//! Given $V = \sum_i G_i \sigma^2_i + I \sigma^2_e$ from
//! [`crate::reml`], these submodules solve the mixed-model equations to
//! produce estimated breeding values (EBVs) and fixed-effect estimates.
//!
//! ## Available solvers
//!
//! - [`cholesky`] — direct Cholesky factorisation of $V$. Wraps
//!   [`factorized::FactorizedV`] for reuse across multiple right-hand
//!   sides. Default for $n < 10{,}000$.
//! - [`pcg`] — preconditioned conjugate gradient. Avoids forming
//!   $V^{-1}$ explicitly; the diagonal of $V$ serves as the
//!   preconditioner. Default for $n \ge 10{,}000$.
//! - [`factorized`] — shared Cholesky cache used by `cholesky` and
//!   re-exported to [`crate::gwas::emmax`] so EMMAX does not re-factor
//!   $V$ per marker.
//!
//! ## Dispatcher
//!
//! [`solve_ebv`] is the public entry point. It honours an explicit
//! `solver = "cholesky" | "pcg"` argument or auto-selects via
//! [`auto_select_solver`] based on $n$. The returned [`BlupResult`]
//! contains per-random-effect EBVs, fixed-effect estimates, and
//! diagnostic info (which solver ran, iteration count for PCG).

pub mod cholesky;
pub mod pcg;
pub mod factorized;

use ndarray::Array1;
use thiserror::Error;

pub type StdResult<T, E> = std::result::Result<T, E>;

/// Error types for EBV solvers
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Cholesky factorization failed: V not positive definite")]
    NotPositiveDefinite,

    #[error("PCG failed to converge after {0} iterations")]
    PcgNotConverged(usize),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Linear algebra error: {0}")]
    LinAlgError(String),
}

impl From<SolverError> for extendr_api::Error {
    fn from(e: SolverError) -> Self {
        extendr_api::Error::from(e.to_string())
    }
}

/// EBV solution result
#[derive(Debug)]
pub struct BlupResult {
    /// Estimated breeding values per random effect
    /// gebv[i] = EBV vector for i-th random effect (length n)
    pub gebv: Vec<Array1<f64>>,
    /// Labels for each random effect
    pub labels: Vec<String>,
    /// Fixed effect estimates
    pub fixed_effects: Array1<f64>,
    /// Solver used: "cholesky" or "pcg"
    pub solver: String,
    /// Number of PCG iterations (0 if Cholesky)
    pub n_iter: usize,
}

impl BlupResult {
    pub fn new(
        gebv: Vec<Array1<f64>>,
        labels: Vec<String>,
        fixed_effects: Array1<f64>,
        solver: &str,
        n_iter: usize,
    ) -> Self {
        Self {
            gebv,
            labels,
            fixed_effects,
            solver: solver.to_string(),
            n_iter,
        }
    }

    /// Total GEBV = sum across all random effects
    pub fn total_gebv(&self) -> Array1<f64> {
        if self.gebv.is_empty() {
            return Array1::zeros(0);
        }
        let n = self.gebv[0].len();
        self.gebv.iter().fold(
            Array1::<f64>::zeros(n),
            |acc, g| acc + g
        )
    }
}

/// Auto-select solver based on n
pub fn auto_select_solver(n: usize) -> &'static str {
    const N_THRESHOLD: usize = 10_000;
    if n < N_THRESHOLD { "cholesky" } else { "pcg" }
}

/// Dispatch to appropriate solver
pub fn solve_ebv(
    y: &ndarray::Array1<f64>,
    x: &ndarray::Array2<f64>,
    g_list: &[(ndarray::Array2<f64>, String)],
    sigma2: &[f64],
    n: usize,
    n_random: usize,
    max_iter: usize,
    tol: f64,
    solver: &str,
) -> StdResult<BlupResult, SolverError> {
    let solver_used = match solver {
        "cholesky" => "cholesky",
        "pcg"      => "pcg",
        _          => auto_select_solver(n),
    };
    match solver_used {
        "cholesky" => cholesky::solve_cholesky_internal(
            y, x, g_list, sigma2, n, n_random
        ),
        _ => pcg::solve_pcg_internal(
            y, x, g_list, sigma2, n, n_random, max_iter, tol
        ),
    }
}