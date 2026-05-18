//masreml/src/rust/src/matrix/mh_additive.rs
use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::{GMatrix, MatrixError, StdResult};

/// Frequency-weighted row shrinkage (per haplotype locus).
///
/// Applied after Da (2015) Eqs. 22-24 encoding as a per-row rank-1
/// deflation along the allele-frequency vector p:
///     W'_{i,k} = W_{i,k} - p_k * S / F
/// where S = sum_l p_l W_{i,l} and F = sum_l p_l.
///
/// The p-weighted row sum is partially damped to S * (1 - Q/F) 
/// with Q = sum_l p_l^2. 
/// function in masbayes (src/rust/src/matrix.rs).
fn frequency_weighted_row_shrinkage(w: &mut Array2<f64>, freqs: &[f64]) {
    let n = w.nrows();
    let n_col = w.ncols();
    let freq_sum: f64 = freqs.iter().sum();
    if freq_sum < 1e-10 {
        return;
    }
    for i in 0..n {
        let weighted_sum: f64 = (0..n_col).map(|k| freqs[k] * w[[i, k]]).sum();
        for k in 0..n_col {
            w[[i, k]] -= freqs[k] * weighted_sum / freq_sum;
        }
    }
}

/// Da (2015) W_αh coding for multi-allelic haplotype markers
///
/// For each individual i and haplotype allele k (excluding most frequent):
///   w_ij,k = 2*p_k          if individual does not carry allele k (i,j != k)
///   w_ij,k = -(1 - 2*p_k)  if individual carries allele k once (i!=j, i=k or j=k)
///   w_ij,k = -2*(1 - p_k)  if individual is homozygous for allele k (i=j=k)
///
/// Input:
///   hap1, hap2: paternal/maternal haplotype vectors (n), integer allele codes
///   p: allele frequencies vector
///   drop_idx: index of most frequent allele to drop
///
/// Output: W_αh matrix (n x n_alleles-1)
fn code_w_mh(
    hap1: &Array1<i32>,
    hap2: &Array1<i32>,
    p: &[f64],
    drop_idx: usize,
) -> Array2<f64> {
    let n = hap1.len();
    let n_alleles = p.len();

    // Allele indices excluding dropped (most frequent) allele
    let allele_idx: Vec<usize> = (0..n_alleles)
        .filter(|&k| k != drop_idx)
        .collect();

    let n_cols = allele_idx.len();
    let mut w = Array2::<f64>::zeros((n, n_cols));

    // Parallel over individuals
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let a1 = hap1[i] as usize;
            let a2 = hap2[i] as usize;

            let a1 = if a1 < n_alleles { a1 } else { 0 };
            let a2 = if a2 < n_alleles { a2 } else { 0 };

            allele_idx.iter()
                .map(|&k| {
                    let pk = p[k];
                    if a1 != k && a2 != k {
                        // Individual does not carry allele k
                        2.0 * pk
                    } else if a1 != a2 && (a1 == k || a2 == k) {
                        // Individual carries allele k once (heterozygous)
                        -(1.0 - 2.0 * pk)
                    } else {
                        // Individual homozygous for allele k (a1 == a2 == k)
                        -2.0 * (1.0 - pk)
                    }
                })
                .collect()
        })
        .collect();

    // Fill matrix
    for i in 0..n {
        for (col, &val) in rows[i].iter().enumerate() {
            w[[i, col]] = val;
        }
    }
    w
}

/// Build MH additive Agh matrix (Da 2015)
/// Agh = W_αh * W_αh' / k_αh
/// k_αh = tr(W_αh * W_αh') / n
///
/// Input:
///   hap1: paternal haplotype matrix (n x n_loci), integer allele codes
///   hap2: maternal haplotype matrix (n x n_loci), integer allele codes
///   n_alleles_per_locus: number of distinct alleles per locus
///
/// Output: GMatrix { g: n×n, k, n, m }
pub fn build_g_mh_add_internal(
    hap1: &Array2<i32>,
    hap2: &Array2<i32>,
    n_alleles_per_locus: &[usize],
    weights: Option<&[f64]>,        // weight per locus (PP_j from GWABLUP)
    ref_hap1: Option<&Array2<i32>>,
    ref_hap2: Option<&Array2<i32>>,
) -> StdResult<GMatrix, MatrixError> {
    let n = hap1.nrows();
    let n_loci = hap1.ncols();

    // Validate weights length if provided
    if let Some(d) = weights {
        if d.len() != n_loci {
            return Err(MatrixError::InvalidDimension(
                format!(
                    "weights length {} != n_loci {}",
                    d.len(), n_loci
                )
            ));
        }
    }

    if hap1.dim() != hap2.dim() {
        return Err(MatrixError::InvalidDimension(
            format!("hap1 and hap2 dimensions must match: {:?} vs {:?}",
                hap1.dim(), hap2.dim())
        ));
    }

    if n_alleles_per_locus.len() != n_loci {
        return Err(MatrixError::InvalidDimension(
            format!("n_alleles_per_locus length {} must equal n_loci {}",
                n_alleles_per_locus.len(), n_loci)
        ));
    }

    // Accumulate W_αh W_αh' across all loci
    // Agh_numerator = sum over loci of (W_l * W_l')
    let mut agh_num = Array2::<f64>::zeros((n, n));
    let mut k_total = 0.0f64;

    for locus in 0..n_loci {
        let h1_col = hap1.column(locus).to_owned();
        let h2_col = hap2.column(locus).to_owned();
        let n_alleles = n_alleles_per_locus[locus];

        // Skip monomorphic loci
        if n_alleles < 2 {
            continue;
        }

        // Compute allele frequencies for this locus
        let (ref_h1_col, ref_h2_col);
        let (h1_for_freq, h2_for_freq) = if let (Some(r1), Some(r2)) = (ref_hap1, ref_hap2) {
            ref_h1_col = r1.column(locus).to_owned();
            ref_h2_col = r2.column(locus).to_owned();
            (&ref_h1_col, &ref_h2_col)
        } else {
            (&h1_col, &h2_col)
        };
        let n_ref = h1_for_freq.len();

        let mut counts = vec![0usize; n_alleles];
        h1_for_freq.iter().for_each(|&a| {
            let idx = a as usize;
            if idx < counts.len() { counts[idx] += 1; }
        });
        h2_for_freq.iter().for_each(|&a| {
            let idx = a as usize;
            if idx < counts.len() { counts[idx] += 1; }
        });
        let total_alleles = (2 * n_ref) as f64;
        let p: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / total_alleles)
            .collect();

        // Find most frequent allele to drop — from ref (training)
        let drop_idx = counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Code W_αh for this locus
        let mut w_l = code_w_mh(&h1_col, &h2_col, &p, drop_idx);

        // Sum-to-zero constraint — freqs of kept alleles (excluding drop_idx)
        let kept_freqs: Vec<f64> = (0..p.len())
            .filter(|&k| k != drop_idx)
            .map(|k| p[k])
            .collect();
        frequency_weighted_row_shrinkage(&mut w_l, &kept_freqs);

        // Scale W_l by sqrt(weight) per locus
        // W_l_scaled = W_l * sqrt(d_l)
        // W_l_scaled W_l_scaled' = W_l W_l' * d_l
        // k: tr(W_scaled W_scaled') / n
        let scale = weights.map_or(1.0, |d| d[locus].sqrt());
        let w_l_scaled = w_l.mapv(|v| v * scale);

        // Accumulate W_l_scaled * W_l_scaled'
        let wwt = w_l_scaled.dot(&w_l_scaled.t());
        agh_num += &wwt;

        // Accumulate tr for k
        let tr: f64 = (0..n).map(|i| wwt[[i, i]]).sum();
        k_total += tr;
    }

    if k_total == 0.0 {
        return Err(MatrixError::InvalidDimension(
            "k_αh = 0: all loci monomorphic".to_string()
        ));
    }

    // k_αh = tr(W_αh W_αh') / n (pooled across loci)
    let k = k_total / n as f64;

    // Agh = W_αh W_αh' / k_αh
    let agh = agh_num.mapv(|v| v / k);

    Ok(GMatrix::new(agh))
}

/// Extendr entry point
/// Input:
///   hap1: paternal haplotype matrix (n x n_loci), integer R matrix
///   hap2: maternal haplotype matrix (n x n_loci), integer R matrix
///   n_alleles: integer vector, number of alleles per locus
/// Output: Agh matrix (n x n)
#[extendr]
pub fn build_g_mh_add(
    hap1: RMatrix<i32>,
    hap2: RMatrix<i32>,
    n_alleles: &[i32],
    weights: Nullable<&[f64]>,
    ref_hap1: Nullable<RMatrix<i32>>,
    ref_hap2: Nullable<RMatrix<i32>>,
) -> Result<RMatrix<f64>> {
    let nrow = hap1.nrows();
    let ncol = hap1.ncols();

    let h1_t = Array2::from_shape_vec(
        (ncol, nrow),
        hap1.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let h1 = h1_t.reversed_axes().to_owned();

    let h2_t = Array2::from_shape_vec(
        (ncol, nrow),
        hap2.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;
    let h2 = h2_t.reversed_axes().to_owned();

    let n_alleles_usize: Vec<usize> = n_alleles.iter()
        .map(|&a| a as usize)
        .collect();

    let w_opt: Option<Vec<f64>> = match weights {
        Nullable::NotNull(d) => Some(d.to_vec()),
        Nullable::Null       => None,
    };

    let ref_h1_opt: Option<Array2<i32>> = match ref_hap1 {
        Nullable::NotNull(m) => {
            let nr = m.nrows(); let nc = m.ncols();
            let t = Array2::from_shape_vec((nc, nr), m.data().to_vec())
                .map_err(|e| Error::from(e.to_string()))?;
            Some(t.reversed_axes().to_owned())
        },
        Nullable::Null => None,
    };
    let ref_h2_opt: Option<Array2<i32>> = match ref_hap2 {
        Nullable::NotNull(m) => {
            let nr = m.nrows(); let nc = m.ncols();
            let t = Array2::from_shape_vec((nc, nr), m.data().to_vec())
                .map_err(|e| Error::from(e.to_string()))?;
            Some(t.reversed_axes().to_owned())
        },
        Nullable::Null => None,
    };

    let gmat = build_g_mh_add_internal(
        &h1,
        &h2,
        &n_alleles_usize,
        w_opt.as_deref(),
        ref_h1_opt.as_ref(),
        ref_h2_opt.as_ref(),
    ).map_err(|e| Error::from(e.to_string()))?;

    let n = gmat.g.nrows();
    let g_vec: Vec<f64> = gmat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(n, n, |r, c| g_vec[r * n + c]))
}

/// Build W_αh flat matrix from hap1/hap2
/// Returns (w_mh, block_sizes)
/// w_mh: n × total_alleles, columns grouped per block
/// block_sizes: n_alleles-1 per locus
pub fn build_w_mh_internal(
    hap1: &Array2<i32>,
    hap2: &Array2<i32>,
    n_alleles_per_locus: &[usize],
    ref_hap1: Option<&Array2<i32>>,
    ref_hap2: Option<&Array2<i32>>,
) -> StdResult<(Array2<f64>, Vec<usize>), MatrixError> {
    let n      = hap1.nrows();
    let n_loci = hap1.ncols();

    // Calculate the total column W_αh
    let block_sizes: Vec<usize> = n_alleles_per_locus.iter()
        .map(|&a| if a >= 2 { a - 1 } else { 0 })
        .collect();
    let total_cols: usize = block_sizes.iter().sum();

    let mut w_mh = Array2::<f64>::zeros((n, total_cols));
    let mut col_offset = 0usize;

    for locus in 0..n_loci {
        let n_alleles = n_alleles_per_locus[locus];
        if n_alleles < 2 {
            continue;
        }

        let h1_col = hap1.column(locus).to_owned();
        let h2_col = hap2.column(locus).to_owned();

        // Compute allele frequencies
        let (ref_h1_col, ref_h2_col);
        let (h1_for_freq, h2_for_freq) = if let (Some(r1), Some(r2)) = (ref_hap1, ref_hap2) {
            ref_h1_col = r1.column(locus).to_owned();
            ref_h2_col = r2.column(locus).to_owned();
            (&ref_h1_col, &ref_h2_col)
        } else {
            (&h1_col, &h2_col)
        };
        let n_ref = h1_for_freq.len();

        let mut counts = vec![0usize; n_alleles];
        h1_for_freq.iter().for_each(|&a| {
            let idx = a as usize;
            if idx < counts.len() { counts[idx] += 1; }
        });
        h2_for_freq.iter().for_each(|&a| {
            let idx = a as usize;
            if idx < counts.len() { counts[idx] += 1; }
        });
        let total_alleles = (2 * n_ref) as f64;
        let p: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / total_alleles)
            .collect();

        let drop_idx = counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Build W_αh for this locus
        let mut w_l = code_w_mh(&h1_col, &h2_col, &p, drop_idx);

        // Sum-to-zero constraint
        let kept_freqs: Vec<f64> = (0..p.len())
            .filter(|&k| k != drop_idx)
            .map(|k| p[k])
            .collect();
        frequency_weighted_row_shrinkage(&mut w_l, &kept_freqs);

        let n_cols = w_l.ncols();

        // Fill in w_mh
        w_mh.slice_mut(ndarray::s![.., col_offset..col_offset + n_cols])
            .assign(&w_l);
        col_offset += n_cols;
    }

    Ok((w_mh, block_sizes))
}