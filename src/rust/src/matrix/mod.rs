//! Relationship-matrix construction.
//!
//! Each submodule produces a positive semi-definite $G$ (or $A$ for
//! pedigree) matrix that can be combined in a multi-component mixed model
//! by the [`crate::reml`] kernels.
//!
//! Shared infrastructure:
//!
//! - [`GMatrix`] — owned $n \times n$ matrix wrapper used as the common
//!   return type so REML can iterate over heterogeneous components.
//! - [`compute_k`] — computes $\mathrm{tr}(WW^\top)/n$ directly from $W$
//!   without forming $WW^\top$, the scaling constant for normalising
//!   marker-based $G$ matrices.
//! - [`validate_w`] — defensive checks (non-empty, finite values) before
//!   building $G$; returns a structured `MatrixError` for upstream R-side
//!   error reporting.
//! - [`MatrixError`] — error enum bridged to `extendr_api::Error` so R
//!   surfaces a clean error message instead of a Rust panic.

pub mod snp_additive;
pub mod snp_dominance;
pub mod mh_additive;
pub mod pedigree;

use ndarray::Array2;
use thiserror::Error;

pub type StdResult<T, E> = std::result::Result<T, E>;

/// Error types for matrix construction
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("W matrix is empty or invalid dimension: {0}")]
    InvalidDimension(String),

    #[error("W matrix contains NaN or Inf")]
    InvalidValues,

    #[error("Invalid pedigree: {0}")]
    InvalidPedigree(String),

    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
}

impl From<MatrixError> for extendr_api::Error {
    fn from(e: MatrixError) -> Self {
        extendr_api::Error::from(e.to_string())
    }
}

/// G matrix construction results
pub struct GMatrix {
    pub g: Array2<f64>,
}

impl GMatrix {
    pub fn new(g: Array2<f64>) -> Self {
        Self { g }
    }
}

/// Compute `tr(W · W') / n` directly from `W`, without forming the
/// `n × n` Gram product first.
///
/// # Identity
///
/// ```text
/// tr(W · W') = Σ_{i, j} W[i, j]² = ‖W‖_F²    (Frobenius norm squared).
/// ```
///
/// So `tr(W W') / n` is just the sum of squared entries divided by
/// the row count. Computing it this way costs `O(n · m)` instead of
/// the `O(n² · m)` you would pay to form `W · W'` first — a big
/// saving when `n` is large and we only need the scalar scale `k`.
///
/// # Use
///
/// `k` is the normalising constant in `G = W W' / k` for both the
/// VanRaden SNP G ([`super::snp_additive`]) and the multi-allelic
/// G_αh ([`super::mh_additive`]). It guarantees
/// `mean(diag(G)) ≈ 1 + F̄` so heritability is on the standard scale.
pub fn compute_k(w: &Array2<f64>) -> f64 {
    let n = w.nrows() as f64;
    let trace_wwt: f64 = w.iter().map(|x| x * x).sum();
    trace_wwt / n
}

/// Sanity-check a design matrix before building a G matrix from it.
///
/// Two failure modes are caught:
///
/// 1. **Empty dimensions** (`nrows == 0` or `ncols == 0`) — building
///    `W W'` would succeed but produce a `(0, 0)` G matrix that
///    later crashes the REML / BLUP solvers. Catch it here with a
///    descriptive error that names the offending matrix via `label`.
///
/// 2. **Non-finite entries** (NaN or ±Inf) — `W W'` propagates them
///    silently, and they ultimately surface as Cholesky failures in
///    obscure call sites. Detecting at this entry point gives the
///    user a clear error rather than "factorisation failed".
///
/// The `label` argument is propagated into the error message so the
/// user can tell which design matrix triggered the failure (SNP,
/// dominance, MH, etc.).
pub fn validate_w(w: &Array2<f64>, label: &str) -> StdResult<(), MatrixError> {
    if w.nrows() == 0 || w.ncols() == 0 {
        return Err(MatrixError::InvalidDimension(
            format!("{}: nrows={}, ncols={}", label, w.nrows(), w.ncols())
        ));
    }
    if w.iter().any(|x| x.is_nan() || x.is_infinite()) {
        return Err(MatrixError::InvalidValues);
    }
    Ok(())
}