use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::{GMatrix, MatrixError, StdResult};

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
) -> StdResult<GMatrix, MatrixError> {
    let n = hap1.nrows();
    let n_loci = hap1.ncols();

    // Validate weights length jika provided
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
        let mut counts = vec![0usize; n_alleles];
        h1_col.iter().for_each(|&a| counts[a as usize] += 1);
        h2_col.iter().for_each(|&a| counts[a as usize] += 1);
        let total_alleles = (2 * n) as f64;
        let p: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / total_alleles)
            .collect();

        // Find most frequent allele to drop
        let drop_idx = counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Code W_αh for this locus
        let w_l = code_w_mh(&h1_col, &h2_col, &p, drop_idx);

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
) -> Result<RMatrix<f64>> {
    let nrow = hap1.nrows();
    let ncol = hap1.ncols();

    let h1 = Array2::from_shape_vec(
        (nrow, ncol),
        hap1.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let h2 = Array2::from_shape_vec(
        (nrow, ncol),
        hap2.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let n_alleles_usize: Vec<usize> = n_alleles.iter()
        .map(|&a| a as usize)
        .collect();

    let w_opt: Option<Vec<f64>> = match weights {
        Nullable::NotNull(d) => Some(d.to_vec()),
        Nullable::Null       => None,
    };

    let gmat = build_g_mh_add_internal(
        &h1,
        &h2,
        &n_alleles_usize,
        w_opt.as_deref(),
    ).map_err(|e| Error::from(e.to_string()))?;

    let n = gmat.g.nrows();
    let g_vec: Vec<f64> = gmat.g.into_raw_vec();
    Ok(RMatrix::new_matrix(n, n, |r, c| g_vec[r * n + c]))
}

/// Build W_αh flat matrix dari hap1/hap2
/// Returns (w_mh, block_sizes)
/// w_mh: n × total_alleles, kolom dikelompokkan per blok
/// block_sizes: n_alleles-1 per locus
pub fn build_w_mh_internal(
    hap1: &Array2<i32>,
    hap2: &Array2<i32>,
    n_alleles_per_locus: &[usize],
) -> StdResult<(Array2<f64>, Vec<usize>), MatrixError> {
    let n      = hap1.nrows();
    let n_loci = hap1.ncols();

    // Hitung total kolom W_αh
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
        let mut counts = vec![0usize; n_alleles];
        h1_col.iter().for_each(|&a| counts[a as usize] += 1);
        h2_col.iter().for_each(|&a| counts[a as usize] += 1);
        let total_alleles = (2 * n) as f64;
        let p: Vec<f64> = counts.iter()
            .map(|&c| c as f64 / total_alleles)
            .collect();

        let drop_idx = counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Build W_αh untuk locus ini
        let w_l = code_w_mh(&h1_col, &h2_col, &p, drop_idx);
        let n_cols = w_l.ncols();

        // Isi ke w_mh
        w_mh.slice_mut(ndarray::s![.., col_offset..col_offset + n_cols])
            .assign(&w_l);
        col_offset += n_cols;
    }

    Ok((w_mh, block_sizes))
}