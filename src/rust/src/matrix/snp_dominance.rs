//! SNP dominance relationship matrix.
//!
//! Implements the dominance coding of Da et al. (2014) and Wang & Da
//! (2014):
//!
//! ```text
//! w_δ_ij = -2 · p_j²            if genotype AA  (homozygous reference)
//! w_δ_ij =  2 · p_j (1 - p_j)   if genotype Aa  (heterozygous)
//! w_δ_ij = -2 · (1 - p_j)²      if genotype aa  (homozygous alternate)
//! ```
//!
//! with $p_j$ the reference allele frequency. The dominance G matrix is
//! then
//!
//! ```text
//! D = W_δ W_δ' / k_δ,    k_δ = tr(W_δ W_δ') / n
//! ```
//!
//! Under HWE, the expected value of $W_\delta^{(j)}$ is zero, so the
//! coding is mean-zero by construction — analogous to the
//! frequency-centered additive coding in [`super::snp_additive`].
//!
//! ## Why a separate dominance G?
//!
//! Additive and dominance variance components are orthogonal under HWE.
//! In a mixed model
//!
//! ```text
//! y = X β + Z g + Z d + ε
//! Var(g) = G · σ²_a    Var(d) = D · σ²_d
//! ```
//!
//! REML can estimate $\sigma^2_a$ and $\sigma^2_d$ separately. Dropping
//! the dominance term biases the additive heritability estimate upward
//! when dominance variance is non-trivial.

use extendr_api::prelude::*;
use ndarray::{Array2, Axis};
use rayon::prelude::*;

use super::{GMatrix, MatrixError, compute_k, validate_w, StdResult};
use super::snp_additive::compute_gram_parallel;

/// SNP dominance coding (Da et al. 2014, Wang & Da 2014)
/// w_δ_ij:
///   genotype AA (0): -2*p²_j  [homozygous ref]  
///   genotype Aa (1):  2*p_j*(1-p_j)  [heterozygous]
///   genotype aa (2): -2*(1-p_j)²  [homozygous alt]
///
/// Where p_j = reference allele frequency
fn compute_allele_freq(w_raw: &Array2<f64>) -> Vec<f64> {
    let n = w_raw.nrows() as f64;
    (0..w_raw.ncols())
        .map(|j| {
            let col_sum: f64 = w_raw.column(j).iter().sum();
            col_sum / (2.0 * n)
        })
        .collect()
}

/// Apply the Da-Wang dominance coding rule to every entry of the
/// raw `{0, 1, 2}` SNP matrix, in parallel over columns.
///
/// # Coding rule (per marker `j` with frequency `p_j`)
///
/// ```text
/// W_raw[i, j] = 0 (AA, homozygous reference)  → -2 · p_j²
/// W_raw[i, j] = 1 (Aa, heterozygous)          →  2 · p_j · (1 − p_j)
/// W_raw[i, j] = 2 (aa, homozygous alternate)  → -2 · (1 − p_j)²
/// ```
///
/// Missing values (any code other than 0/1/2) are silently set to 0
/// — they contribute zero to the dominance G but don't propagate
/// NaN. This is a deliberate choice: the upstream pipeline filters
/// missing genotypes before getting here, so any non-{0,1,2} value
/// represents a data-quality error rather than an expected case.
///
/// # Why this coding
///
/// Under HWE, `E[W_δ_{i, j}] = 0` by direct expansion:
/// `p_j² · (-2p_j²) + 2 p_j(1 - p_j) · 2 p_j(1 - p_j) + (1 - p_j)² · (-2(1 - p_j)²)`
/// simplifies to zero. The coding is therefore "mean-zero by
/// construction" in the same way as Da's multi-allelic encoding
/// (see `mh_additive.rs`).
fn code_w_dominance(w_raw: &Array2<f64>, p: &[f64]) -> Array2<f64> {
    let (n, m) = w_raw.dim();
    let mut w_d = Array2::<f64>::zeros((n, m));

    w_d.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let pj = p[j];
            let p2 = pj * pj;
            let q2 = (1.0 - pj) * (1.0 - pj);
            let pq2 = 2.0 * pj * (1.0 - pj);

            col.iter_mut()
                .zip(w_raw.column(j).iter())
                .for_each(|(out, &raw)| {
                    *out = match raw as i32 {
                        0 => -2.0 * p2,
                        1 => pq2,
                        2 => -2.0 * q2,
                        _ => 0.0,  // missing → treat as 0
                    };
                });
        });
    w_d
}

/// Build SNP dominance D matrix (Da et al. 2014)
/// D = W_δ W_δ' / k_δ
///
/// Input: w_raw (n × m), raw genotype 0/1/2
/// Output: GMatrix { g: n×n dominance matrix, k, n, m }
pub fn build_g_snp_dom_internal(
    w_raw: &Array2<f64>,
) -> StdResult<GMatrix, MatrixError> {
    validate_w(w_raw, "SNP dominance W")?;

    let p = compute_allele_freq(w_raw);
    let w_d = code_w_dominance(w_raw, &p);
    let k = compute_k(&w_d);

    if k == 0.0 {
        return Err(MatrixError::InvalidDimension(
            "k_delta = 0, no dominance variance".to_string()
        ));
    }

    let g = compute_gram_parallel(&w_d, k);

    Ok(GMatrix::new(g))
}

/// Extendr entry point
/// Input R matrix: raw genotype 0/1/2 (n × m)
/// Output R matrix: D matrix (n × n)
#[extendr]
pub fn build_g_snp_dom(w: RMatrix<f64>) -> Result<RMatrix<f64>> {
    let nrow = w.nrows();
    let ncol = w.ncols();
    let data: Vec<f64> = w.data().to_vec();
    let w_transposed = Array2::from_shape_vec((ncol, nrow), data)
        .map_err(|e| Error::from(e.to_string()))?;
    let w_arr = w_transposed.reversed_axes().to_owned();

    let gmat = build_g_snp_dom_internal(&w_arr)
        .map_err(|e| Error::from(e.to_string()))?;

    let n = gmat.g.nrows();
    let g_vec: Vec<f64> = gmat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(n, n, |r, c| g_vec[r * n + c]))
}