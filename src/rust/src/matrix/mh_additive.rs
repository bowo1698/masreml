//masreml/src/rust/src/matrix/mh_additive.rs

//! Multi-allelic additive relationship matrix $G_{\alpha h}$ (Da, 2015).
//!
//! Builds the $n \times n$ additive G matrix from phased microhaplotype
//! data, locus by locus:
//!
//! 1. Compute training-set allele frequencies $p_k$ per locus.
//! 2. Choose the most frequent microhaplotype as the baseline and drop it
//!    from the design matrix (identifiability constraint).
//! 3. Encode the remaining $h - 1$ microhaplotypes with the Da (2015)
//!    three-value rule into a per-locus $W_l$ matrix.
//! 4. Apply the per-locus
//!    [`frequency_weighted_row_shrinkage`] — see its docstring for the
//!    precise definition; identical to the equivalent step in `masbayes`.
//! 5. Optionally scale each locus by `sqrt(weight_l)` for GWABLUP.
//! 6. Accumulate $G_{\alpha h} = \sum_l W_l W_l^\top$ and the scaling
//!    constant $k_{\alpha h} = \mathrm{tr}(G_{\alpha h}) / n$, returning
//!    the normalised matrix $G_{\alpha h} / k_{\alpha h}$.
//!
//! ## Train / test alignment
//!
//! Allele frequencies and baseline microhaplotypes are taken from the
//! reference (training) data when `ref_hap1` / `ref_hap2` are supplied;
//! otherwise they are computed from the data being encoded. Always pass
//! training references when encoding a test set — see the discussion in
//! [`crate::matrix::snp_additive`] for the analogous SNP case.
//!
//! ## Parallelism
//!
//! Locus-level $W_l$ encoding is parallelised over individuals via rayon.
//! The outer loop over loci is sequential because each locus accumulates
//! into the shared $G$ matrix.
//!
//! ## Reference
//!
//! Da, Y. (2015). Multi-allelic haplotype model based on genetic partition
//! for genomic prediction and variance component estimation using SNP
//! markers. *BMC Genetics*, 16:144.

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

// ============================================================================
// Unit tests
// ============================================================================
//
// Deterministic checks for the Da (2015) encoding and the
// frequency-weighted row shrinkage step. Run with `cargo test` or compile-
// verify with `cargo check --tests`.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn shrinkage_preserves_dimensions() {
        let mut w = array![
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 2.5],
        ];
        let freqs = [0.4_f64, 0.3, 0.2];
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        assert_eq!(w.nrows(), 2);
        assert_eq!(w.ncols(), 3);
    }

    #[test]
    fn shrinkage_partial_damping_factor() {
        // Verify that the post-shrink p-weighted row sum equals
        // S · (1 - Q/F) where Q = Σ p_l², F = Σ p_l.
        let p = [0.3_f64, 0.2];
        let mut w = array![[1.0, 2.0]];

        let s_before: f64 = p.iter().zip(w.row(0).iter()).map(|(pl, wl)| pl * wl).sum();
        frequency_weighted_row_shrinkage(&mut w, &p);
        let s_after: f64 = p.iter().zip(w.row(0).iter()).map(|(pl, wl)| pl * wl).sum();

        let q: f64 = p.iter().map(|x| x * x).sum();
        let f: f64 = p.iter().sum();
        let expected = s_before * (1.0 - q / f);
        assert!((s_after - expected).abs() < 1e-10,
            "expected weighted sum {} after shrinkage, got {}", expected, s_after);
    }

    #[test]
    fn shrinkage_skips_zero_frequency_block() {
        // freq_sum < 1e-10 short-circuits, leaving W untouched.
        let mut w = array![[1.0, 2.0]];
        let original = w.clone();
        let freqs = [0.0_f64, 0.0];
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        assert_eq!(w, original);
    }

    #[test]
    fn code_w_mh_da_encoding_three_classes() {
        // Single locus, h = 3 alleles, baseline (most frequent) dropped.
        // Frequencies: [0.5, 0.3, 0.2], baseline = allele 0 (drop_idx = 0).
        // After drop, columns correspond to allele indices 1, 2.
        let p = vec![0.5_f64, 0.3, 0.2];
        let drop_idx = 0_usize;

        // Four individuals exercising the three Da genotype classes:
        //   ind 0: (0, 0) — homozygous baseline, both non-baseline absent
        //   ind 1: (1, 1) — homozygous allele 1
        //   ind 2: (0, 1) — heterozygous, one copy of allele 1
        //   ind 3: (1, 2) — heterozygous, one of allele 1 and one of allele 2
        let h1 = Array1::from_vec(vec![0, 1, 0, 1]);
        let h2 = Array1::from_vec(vec![0, 1, 1, 2]);

        let w = code_w_mh(&h1, &h2, &p, drop_idx);
        assert_eq!(w.nrows(), 4);
        assert_eq!(w.ncols(), 2);  // p.len() − 1

        // Expected values from Eqs. 22-24 (no shrinkage yet at this layer).
        // Allele 1 column (p_1 = 0.3):
        //   ind 0 (absent):       2 · 0.3 = 0.6
        //   ind 1 (homozygous):  -2 · (1 - 0.3) = -1.4
        //   ind 2 (one copy):    -(1 - 2 · 0.3) = -0.4
        //   ind 3 (one copy):    -(1 - 2 · 0.3) = -0.4
        assert!((w[[0, 0]] -  0.6).abs() < 1e-12);
        assert!((w[[1, 0]] - -1.4).abs() < 1e-12);
        assert!((w[[2, 0]] - -0.4).abs() < 1e-12);
        assert!((w[[3, 0]] - -0.4).abs() < 1e-12);

        // Allele 2 column (p_2 = 0.2):
        //   ind 0 (absent):       2 · 0.2 = 0.4
        //   ind 1 (absent):       2 · 0.2 = 0.4
        //   ind 2 (absent):       2 · 0.2 = 0.4
        //   ind 3 (one copy):    -(1 - 2 · 0.2) = -0.6
        assert!((w[[0, 1]] -  0.4).abs() < 1e-12);
        assert!((w[[1, 1]] -  0.4).abs() < 1e-12);
        assert!((w[[2, 1]] -  0.4).abs() < 1e-12);
        assert!((w[[3, 1]] - -0.6).abs() < 1e-12);
    }
}