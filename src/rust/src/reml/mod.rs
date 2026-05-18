//! Variance-component estimation via REML.
//!
//! Implements three REML algorithms and an adaptive dispatcher:
//!
//! - [`he_regression`] — Haseman–Elston regression on phenotype products.
//!   Closed-form, single-shot, used for starting values and sanity checks.
//! - [`ai_reml`] — Average-Information REML (Johnson & Thompson, 1995).
//!   Newton-style updates with the AI matrix; fast but can diverge or hit
//!   negative variances on ill-conditioned problems.
//! - [`em_reml`] — Expectation-Maximization REML (Dempster et al., 1977;
//!   Meyer, 1989). Always produces non-negative variance components but
//!   converges slowly.
//! - [`adaptive`] — auto-selector that runs HE first for starting values,
//!   tries AI, and falls back to EM if AI fails to converge or produces
//!   negative components.
//!
//! ## Common interfaces
//!
//! - [`RemlData`] — bundles the response, fixed-effect design $X$, and the
//!   list of random-effect $G$ matrices. Constructed once by the
//!   adaptive entry point.
//! - [`VarianceComponents`] — return type containing $\sigma^2$ per
//!   component plus convergence diagnostics (iterations, log-likelihood).
//! - [`RemlError`] — structured error type bridged to
//!   `extendr_api::Error` for clean R-side reporting.
//!
//! ## Numerical infrastructure
//!
//! Inversion of the working matrix $V = \sum_i G_i \sigma^2_i + I \sigma^2_e$
//! is delegated to [`crate::utils::linalg`] helpers, which use
//! `ndarray_linalg` LAPACK bindings. Threading is controlled by
//! `set_num_threads` from the same module.

pub mod he_regression;
pub mod ai_reml;
pub mod em_reml;
pub mod adaptive;

use ndarray::{Array1, Array2};
use thiserror::Error;

pub type StdResult<T, E> = std::result::Result<T, E>;

/// Error types for REML estimation
#[derive(Error, Debug)]
pub enum RemlError {
    #[error("REML failed to converge after {0} iterations")]
    NotConverged(usize),

    #[error("AI matrix is singular or not positive definite")]
    SingularAI,

    #[error("Variance components became negative during iteration")]
    NegativeVariance,

    #[error("Linear algebra error: {0}")]
    LinAlgError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<RemlError> for extendr_api::Error {
    fn from(e: RemlError) -> Self {
        extendr_api::Error::from(e.to_string())
    }
}

/// Variance component result from REML
#[derive(Debug, Clone)]
pub struct VarianceComponents {
    /// Estimated variance components [sigma2_1, ..., sigma2_k, sigma2_e]
    pub sigma2: Vec<f64>,
    /// Heritability per genetic component: sigma2_i / sigma2_p
    pub h2: Vec<f64>,
    /// Log-likelihood at convergence
    pub loglik: f64,
    /// Number of iterations to converge
    pub n_iter: usize,
    /// Algorithm used: "HE", "AI", "EM", "HI"
    pub algorithm: String,
    /// Convergence achieved
    pub converged: bool,
}

impl VarianceComponents {
    pub fn new(
        sigma2: Vec<f64>,
        loglik: f64,
        n_iter: usize,
        algorithm: &str,
        converged: bool,
    ) -> Self {
        let n_genetic = sigma2.len() - 1; // exclude sigma2_e
        let sigma2_p: f64 = sigma2.iter().sum();
        let h2 = (0..n_genetic)
            .map(|i| sigma2[i] / sigma2_p)
            .collect();
        Self {
            sigma2,
            h2,
            loglik,
            n_iter,
            algorithm: algorithm.to_string(),
            converged,
        }
    }
}

/// Common REML data structure passed across algorithms
pub struct RemlData {
    /// Phenotype vector (n)
    pub y: Array1<f64>,
    /// Fixed effects design matrix (n x c)
    pub x: Array2<f64>,
    /// List of genetic relationship matrices [(G1, label), ...]
    pub g_list: Vec<(Array2<f64>, String)>,
    /// Number of individuals
    pub n: usize,
    /// Number of random effects (genetic components)
    pub n_random: usize,
}

impl RemlData {
    pub fn new(
        y: Array1<f64>,
        x: Array2<f64>,
        g_list: Vec<(Array2<f64>, String)>,
    ) -> Self {
        let n = y.len();
        let n_random = g_list.len();
        Self { y, x, g_list, n, n_random }
    }

    /// Build V matrix from current variance components
    /// V = sum_i(G_i * sigma2_i) + I * sigma2_e
    pub fn build_v(&self, sigma2: &[f64]) -> Array2<f64> {
        let sigma2_e = sigma2[self.n_random];
        let mut v = Array2::<f64>::eye(self.n) * sigma2_e;
        for (i, (g, _)) in self.g_list.iter().enumerate() {
            v += &(g * sigma2[i]);
        }
        v
    }
}