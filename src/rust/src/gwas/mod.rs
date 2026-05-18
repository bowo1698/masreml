// src/rust/src/gwas/mod.rs

//! Single-marker association testing.
//!
//! Two algorithms with complementary roles in the GWABLUP pipeline:
//!
//! - [`emmax`] — EMMAX (Kang et al., 2010). Per-marker mixed-model
//!   association test that reuses a single pre-factorised $V$ across all
//!   markers, giving GWAS-quality $p$-values at a fraction of the per-SNP
//!   REML cost.
//! - [`smoother`] — moving-average smoother for likelihood-ratio
//!   statistics across adjacent markers/blocks, used to construct the
//!   GWAS-weighted $G$ matrix (`G_wa`) in [`crate::matrix::snp_additive`]
//!   and [`crate::matrix::mh_additive`].
//!
//! ## Output convention
//!
//! [`GwasResult`] uses per-marker (SNP) or per-block (MH) aggregated
//! statistics: LR test statistic, effect estimate $\hat\beta$, standard
//! error, and $p$-value. For multi-allelic MH blocks, $\hat\beta$ and SE
//! are aggregated across alleles within a block, and the $p$-value uses
//! a chi-squared with $\mathrm{df} = h - 1$ (number of non-baseline
//! microhaplotypes).

pub mod emmax;
pub mod smoother;

use thiserror::Error;

pub type StdResult<T, E> = std::result::Result<T, E>;

/// Error types for GWAS
#[derive(Error, Debug)]
pub enum GwasError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Linear algebra error: {0}")]
    LinAlgError(String),
}

impl From<GwasError> for extendr_api::Error {
    fn from(e: GwasError) -> Self {
        extendr_api::Error::from(e.to_string())
    }
}

/// GWAS result per marker (SNP) or block (MH)
#[derive(Debug, Clone)]
pub struct GwasResult {
    /// Log-likelihood ratio per marker/block
    pub lr: Vec<f64>,
    /// Effect estimate per marker/block
    /// For MH: vector of per-allele effects aggregated per block
    pub beta: Vec<f64>,
    /// Standard error per marker/block
    /// For MH: aggregated se per block
    pub se: Vec<f64>,
    /// p-value per marker/block (chi-squared, df=1 for SNP, df=k-1 for MH)
    pub pval: Vec<f64>,
}

impl GwasResult {
    pub fn new(
        lr: Vec<f64>,
        beta: Vec<f64>,
        se: Vec<f64>,
        pval: Vec<f64>,
    ) -> Self {
        Self { lr, beta, se, pval }
    }
}