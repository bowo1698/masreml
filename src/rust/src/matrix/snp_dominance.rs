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

/// Apply dominance coding per parallel column
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
    let w_arr = Array2::from_shape_vec((nrow, ncol), data)
        .map_err(|e| Error::from(e.to_string()))?;

    let gmat = build_g_snp_dom_internal(&w_arr)
        .map_err(|e| Error::from(e.to_string()))?;

    let n = gmat.g.nrows();
    let g_vec: Vec<f64> = gmat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(n, n, |r, c| g_vec[r * n + c]))
}