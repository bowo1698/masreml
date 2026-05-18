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

/// Output of a REML fit.
///
/// Carries the fitted variance components and derived heritabilities
/// together with convergence diagnostics. Returned by every REML
/// algorithm in this module (HE, AI, EM, adaptive) so the
/// downstream R wrapper sees a uniform shape regardless of which
/// path actually ran.
///
/// # Fields
///
/// - `sigma2`: variance components in order
///   `(σ²_1, …, σ²_K, σ²_e)`. Length `n_random + 1`.
/// - `h2`: per-component narrow-sense heritability,
///   `σ²_k / σ²_phenotype` where `σ²_phenotype = Σ σ²`.
///   Length `n_random`.
/// - `loglik`: restricted log-likelihood at the returned `sigma2`.
/// - `n_iter`: iteration count for AI / EM; `1` for HE (single-shot).
/// - `algorithm`: which algorithm actually ran. Useful for
///   diagnostics; values include `"AI"`, `"EM"`, `"HE"`, `"HI"`,
///   `"HI+EM"`, `"HE+EM"`.
/// - `converged`: `true` if AI/EM hit the tolerance, `false`
///   otherwise. HE always returns `true` because it is closed-form.
#[derive(Debug, Clone)]
pub struct VarianceComponents {
    /// Estimated variance components `[σ²_1, …, σ²_K, σ²_e]`.
    pub sigma2: Vec<f64>,
    /// Per-component heritability `σ²_i / σ²_p`. Length `n_random`.
    pub h2: Vec<f64>,
    /// Restricted log-likelihood at the returned σ².
    pub loglik: f64,
    /// Iterations to converge (`1` for closed-form HE).
    pub n_iter: usize,
    /// Algorithm label: `"HE"`, `"AI"`, `"EM"`, `"HI"`, `"HI+EM"`,
    /// or `"HE+EM"`.
    pub algorithm: String,
    /// `true` if convergence tolerance was met.
    pub converged: bool,
}

impl VarianceComponents {
    /// Construct from variance components plus diagnostics.
    ///
    /// Heritabilities are derived from `sigma2` here so callers
    /// cannot accidentally pass an inconsistent `h2` vector. The
    /// definition is `h²_k = σ²_k / Σ σ²` (all components including
    /// residual go into the denominator), which matches the standard
    /// narrow-sense heritability used in animal breeding.
    pub fn new(
        sigma2: Vec<f64>,
        loglik: f64,
        n_iter: usize,
        algorithm: &str,
        converged: bool,
    ) -> Self {
        let n_genetic = sigma2.len() - 1; // exclude σ²_e
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

/// Bundle of inputs shared across every REML algorithm in this
/// module.
///
/// Constructed once by the adaptive dispatcher and passed by
/// reference to HE, AI, and EM. Owning the data here (rather than
/// re-parsing it inside each algorithm) keeps the algorithm
/// implementations focused on the math and lets us swap algorithms
/// (HE → AI → EM fallback) without re-reading R inputs.
///
/// # Fields
///
/// - `y`: length-`n` phenotype vector.
/// - `x`: `(n, c)` fixed-effects design.
/// - `g_list`: per-component relationship matrices, each tagged with
///   the R-side component label so it can be propagated through to
///   the BLUP output.
/// - `n`, `n_random`: cached counts derived from `y` and `g_list`,
///   stored once so every algorithm sees the same values.
pub struct RemlData {
    /// Phenotype vector (length `n`).
    pub y: Array1<f64>,
    /// Fixed-effects design `(n × c)`.
    pub x: Array2<f64>,
    /// `(G_i, label)` tuples, one per random-effect component.
    pub g_list: Vec<(Array2<f64>, String)>,
    /// Number of individuals (matches `y.len()`).
    pub n: usize,
    /// Number of genetic random-effect components (matches `g_list.len()`).
    pub n_random: usize,
}

impl RemlData {
    /// Build a `RemlData` and cache the derived counts.
    ///
    /// `n` and `n_random` are computed once at construction; the
    /// algorithm code can then read them as plain fields without
    /// re-deriving from `y` or `g_list`.
    pub fn new(
        y: Array1<f64>,
        x: Array2<f64>,
        g_list: Vec<(Array2<f64>, String)>,
    ) -> Self {
        let n = y.len();
        let n_random = g_list.len();
        Self { y, x, g_list, n, n_random }
    }

    /// Build the mixed-model variance matrix V from a candidate
    /// variance-component vector.
    ///
    /// # Formula
    ///
    /// ```text
    /// V = Σ_i σ²_i · G_i + σ²_e · I,
    /// ```
    ///
    /// where the σ²_i are the random-effect variances (one per
    /// `G_i` in `g_list`) and σ²_e is the residual variance (the
    /// last entry of `sigma2`).
    ///
    /// # When V is rebuilt
    ///
    /// Every REML iteration that changes σ² rebuilds V via this
    /// method. AI- and EM-REML re-evaluate the gradient / EM update
    /// at the new σ² using the new V; HE regression calls `build_v`
    /// only once, after its closed-form OLS has produced σ²_HE, to
    /// compute the final log-likelihood for reporting.
    ///
    /// # Performance
    ///
    /// `O(K · n²)` where `K = n_random`. Each iteration's biggest
    /// cost is usually the downstream `cholesky(V)` (`O(n³)`), not
    /// this assembly — but if you ever profile a REML run and see
    /// `build_v` near the top, that's the signal to switch to a
    /// lazy / cached V (e.g. by passing `FactorizedV` directly).
    pub fn build_v(&self, sigma2: &[f64]) -> Array2<f64> {
        let sigma2_e = sigma2[self.n_random];
        let mut v = Array2::<f64>::eye(self.n) * sigma2_e;
        for (i, (g, _)) in self.g_list.iter().enumerate() {
            v += &(g * sigma2[i]);
        }
        v
    }
}