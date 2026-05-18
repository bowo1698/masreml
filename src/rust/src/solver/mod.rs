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

    /// Sum the per-component EBVs into a single total GEBV vector.
    ///
    /// The mixed model can split the genetic value across multiple
    /// relationship matrices (e.g. SNP additive + SNP dominance +
    /// pedigree A). Each component produces its own EBV vector via
    /// Henderson's formula `û_i = σ²_i · G_i · P y`. The total
    /// genomic breeding value of individual `j` is the sum across
    /// all components:
    ///
    /// ```text
    /// GEBV[j] = Σ_i û_i[j].
    /// ```
    ///
    /// Returns an empty array if no components are present (a
    /// degenerate case that should not happen in practice but is
    /// handled defensively so downstream code can rely on a valid
    /// `Array1` return).
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

/// Pick a solver automatically based on sample size.
///
/// # Threshold
///
/// `n < 10 000` → `"cholesky"`, `n ≥ 10 000` → `"pcg"`. The crossover
/// is empirical: at n ≈ 10 000 the memory cost of the dense Cholesky
/// factor (n² f64s ≈ 800 MB) starts to dominate, and PCG's `O(n²)`
/// per iteration with low iteration counts becomes more attractive
/// than the one-time `O(n³)` factorisation.
///
/// # Override
///
/// The R-side wrapper exposes a `solver` argument that bypasses this
/// auto-selection: pass `"cholesky"` or `"pcg"` explicitly to pin a
/// particular solver regardless of `n`. Useful for exact
/// reproducibility (Cholesky is deterministic; PCG iteration counts
/// can drift with thread count) or for benchmarking the two paths
/// on the same data.
pub fn auto_select_solver(n: usize) -> &'static str {
    const N_THRESHOLD: usize = 10_000;
    if n < N_THRESHOLD { "cholesky" } else { "pcg" }
}

/// Top-level solver dispatch.
///
/// Honours the user's explicit `solver` argument (`"cholesky"` or
/// `"pcg"`) or falls back to [`auto_select_solver`] for any other
/// value (`"auto"`, empty string, etc.). Delegates the actual work
/// to [`cholesky::solve_cholesky_internal`] or
/// [`pcg::solve_pcg_internal`].
///
/// # Arguments
///
/// - `y`, `x`, `g_list`, `sigma2`: standard mixed-model inputs.
/// - `n`, `n_random`: cached dimension info.
/// - `max_iter`, `tol`: PCG convergence controls (ignored by
///   Cholesky).
/// - `solver`: `"cholesky"`, `"pcg"`, or anything else (auto-select).
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