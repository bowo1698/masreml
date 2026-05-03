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

/// Efficiently compute tr(WW')/n without explicit WW' form
/// tr(WW') = sum of all squared elements of W
pub fn compute_k(w: &Array2<f64>) -> f64 {
    let n = w.nrows() as f64;
    let trace_wwt: f64 = w.iter().map(|x| x * x).sum();
    trace_wwt / n
}

/// W matrix validation
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