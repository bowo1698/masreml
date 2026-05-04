// src/lib.rs
use extendr_api::prelude::*;

mod matrix;
mod reml;
mod solver;
mod utils;
mod gwas;

// Re-export semua extendr functions
use matrix::snp_additive::build_g_snp_add;
use matrix::snp_dominance::build_g_snp_dom;
use matrix::mh_additive::build_g_mh_add;
use matrix::pedigree::build_a_ped;
use reml::adaptive::run_reml;

/// Build SNP additive G matrix (VanRaden)
/// weights: optional PP_j vector for GWABLUP (NULL = standard GBLUP)
#[extendr]
fn r_build_g_snp_add(
    w: RMatrix<f64>,
    weights: Nullable<&[f64]>,
) -> Result<RMatrix<f64>> {
    build_g_snp_add(w, weights)
}

/// Build SNP dominance D matrix (Da et al. 2014)
#[extendr]
fn r_build_g_snp_dom(w: RMatrix<f64>) -> Result<RMatrix<f64>> {
    build_g_snp_dom(w)
}

/// Build MH additive Agh matrix (Da 2015)
/// weights: optional PP_j vector per locus for GWABLUP (NULL = standard GBLUP)
#[extendr]
fn r_build_g_mh_add(
    hap1: RMatrix<i32>,
    hap2: RMatrix<i32>,
    n_alleles: &[i32],
    weights: Nullable<&[f64]>,
) -> Result<RMatrix<f64>> {
    build_g_mh_add(hap1, hap2, n_alleles, weights)
}

/// Build pedigree A matrix (Henderson)
#[extendr]
fn r_build_a_ped(sire: &[i32], dam: &[i32], n: i32) -> Result<RMatrix<f64>> {
    build_a_ped(sire, dam, n)
}

/// Run adaptive REML (HE -> AI-REML -> EM-REML fallback)
#[extendr]
fn r_run_reml(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    method: &str,
    max_iter: i32,
    tol: f64,
    n_threads: i32,
) -> List {
    run_reml(y, x, g_list, method, max_iter, tol, n_threads)
}

/// Solve EBV — auto-selects Cholesky (n < 10k) or PCG (n >= 10k)
#[extendr]
fn r_solve_ebv(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    sigma2: &[f64],
    solver: &str,   // "auto", "cholesky", "pcg"
    max_iter: i32,
    tol: f64,
) -> Result<List> {
    let n = y.len();
    let n_random = g_list.len();

    if sigma2.len() != n_random + 1 {
        return Err(Error::from(format!(
            "sigma2 length {} must be n_random + 1 = {}",
            sigma2.len(), n_random + 1
        )));
    }

    // Parse inputs
    let y_arr = ndarray::Array1::from_vec(y.to_vec());
    let x_arr = ndarray::Array2::from_shape_vec(
        (x.nrows(), x.ncols()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let mut g_matrices: Vec<(ndarray::Array2<f64>, String)> = Vec::new();
    for (name, robj) in g_list.iter() {
        let g_rmat = RMatrix::<f64>::try_from(robj)
            .map_err(|_| Error::from(
                format!("G matrix '{}' is not a numeric matrix", name)
            ))?;
        let g = ndarray::Array2::from_shape_vec(
            (g_rmat.nrows(), g_rmat.ncols()),
            g_rmat.data().to_vec()
        ).map_err(|e| Error::from(e.to_string()))?;
        g_matrices.push((g, name.to_string()));
    }

    // Dispatch via solve_ebv
    let result = solver::solve_ebv(
        &y_arr, &x_arr, &g_matrices, sigma2,
        n, n_random, max_iter as usize, tol, solver
    ).map_err(|e| Error::from(e.to_string()))?;

    // Build output
    let total_gebv    = result.total_gebv();
    let fixed_effects = result.fixed_effects.to_vec();
    let solver_name   = result.solver.clone();
    let labels        = result.labels.clone();
    let gebv_vecs     = result.gebv.clone();

    let mut names: Vec<String> = vec![
        "fixed_effects".to_string(),
        "total_gebv".to_string(),
        "solver".to_string(),
        "n_iter".to_string(),
    ];
    let mut values: Vec<Robj> = vec![
        r!(fixed_effects),
        r!(total_gebv.to_vec()),
        r!(solver_name.as_str()),
        r!(result.n_iter as i32),
    ];

    for (label, gebv) in labels.iter().zip(gebv_vecs.iter()) {
        names.push(label.clone());
        values.push(r!(gebv.to_vec()));
    }

    let names_str: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    Ok(List::from_names_and_values(names_str, values)
        .map_err(|e| Error::from(e.to_string()))?)
}

/// Run EMMAX GWAS for SNP markers
/// Returns list(lr, beta, se, pval)
#[extendr]
fn r_run_emmax_snp(
    w: RMatrix<f64>,
    y: &[f64],
    x: RMatrix<f64>,
    sigma2_g: f64,
    sigma2_e: f64,
    g_u: RMatrix<f64>,
) -> Result<List> {
    use gwas::emmax::run_emmax_snp;
    use solver::factorized::FactorizedV;

    let n = y.len();
    let y_arr = ndarray::Array1::from_vec(y.to_vec());

    let w_arr = ndarray::Array2::from_shape_vec(
        (w.nrows(), w.ncols()),
        w.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let x_arr = ndarray::Array2::from_shape_vec(
        (x.nrows(), x.ncols()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let g_arr = ndarray::Array2::from_shape_vec(
        (g_u.nrows(), g_u.ncols()),
        g_u.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let g_list = vec![(g_arr, "g".to_string())];
    let sigma2 = vec![sigma2_g, sigma2_e];
    let v_factor = FactorizedV::new(&g_list, &sigma2, n)
        .map_err(|e| Error::from(e.to_string()))?;

    let result = run_emmax_snp(&w_arr, &y_arr, &x_arr, &v_factor)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["lr", "beta", "se", "pval"],
        [
            r!(result.lr),
            r!(result.beta),
            r!(result.se),
            r!(result.pval),
        ]
    ).map_err(|e| Error::from(e.to_string()))?)
}

/// Run EMMAX GWAS for MH markers
/// block_sizes: integer vector, n_alleles-1 per block
/// Returns list(lr, beta, se, pval)
#[extendr]
fn r_run_emmax_mh(
    hap1: RMatrix<i32>,
    hap2: RMatrix<i32>,
    n_alleles: &[i32],
    y: &[f64],
    x: RMatrix<f64>,
    sigma2_g: f64,
    sigma2_e: f64,
    g_u: RMatrix<f64>,
) -> Result<List> {
    use gwas::emmax::run_emmax_mh;
    use matrix::mh_additive::build_w_mh_internal; 
    use solver::factorized::FactorizedV;

    let n = y.len();
    let y_arr = ndarray::Array1::from_vec(y.to_vec());

    // Parse hap1/hap2
    let h1 = ndarray::Array2::from_shape_vec(
        (hap1.nrows(), hap1.ncols()),
        hap1.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let h2 = ndarray::Array2::from_shape_vec(
        (hap2.nrows(), hap2.ncols()),
        hap2.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let n_alleles_usize: Vec<usize> = n_alleles.iter()
        .map(|&a| a as usize)
        .collect();

    // Build W_αh flat matrix + block_sizes dari hap1/hap2
    let (w_mh, block_sizes) = build_w_mh_internal(&h1, &h2, &n_alleles_usize)
        .map_err(|e| Error::from(e.to_string()))?;

    let x_arr = ndarray::Array2::from_shape_vec(
        (x.nrows(), x.ncols()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    // Build V = G_u * sigma2_g + I * sigma2_e
    let g_arr = ndarray::Array2::from_shape_vec(
        (g_u.nrows(), g_u.ncols()),
        g_u.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let g_list = vec![(g_arr, "g".to_string())];
    let sigma2  = vec![sigma2_g, sigma2_e];
    let v_factor = FactorizedV::new(&g_list, &sigma2, n)
        .map_err(|e| Error::from(e.to_string()))?;

    let result = run_emmax_mh(&w_mh, &block_sizes, &y_arr, &x_arr, &v_factor)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["lr", "beta", "se", "pval"],
        [
            r!(result.lr),
            r!(result.beta),
            r!(result.se),
            r!(result.pval),
        ]
    ).map_err(|e| Error::from(e.to_string()))?)
}

/// Smooth LR values and compute posterior probabilities
/// Returns list(smoothed_lr, pp)
#[extendr]
fn r_smooth_and_pp(
    lr: &[f64],
    window: i32,
    pi: f64,
) -> Result<List> {
    use gwas::smoother::smooth_and_pp;

    let (smoothed_lr, pp) = smooth_and_pp(lr, window as usize, pi)
        .map_err(|e| Error::from(e.to_string()))?;

    Ok(List::from_names_and_values(
        ["smoothed_lr", "pp"],
        [r!(smoothed_lr), r!(pp)]
    ).map_err(|e| Error::from(e.to_string()))?)
}

extendr_module! {
    mod masreml;
    fn r_build_g_snp_add;
    fn r_build_g_snp_dom;
    fn r_build_g_mh_add;
    fn r_build_a_ped;
    fn r_run_reml;
    fn r_solve_ebv;
    fn r_run_emmax_snp;
    fn r_run_emmax_mh;
    fn r_smooth_and_pp;
}