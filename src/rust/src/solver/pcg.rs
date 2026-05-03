use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, UPLO, Solve};

use super::{BlupResult, SolverError, StdResult};

/// Preconditioned Conjugate Gradient (PCG) solver
/// Recommended for n >= 10,000
///
/// Solves V * x = b iteratively without forming V^-1 explicitly
/// Preconditioner: M = diag(V) (Jacobi preconditioner)
///
/// Algorithm:
///   r0 = b - V*x0
///   z0 = M^-1 * r0
///   p0 = z0
///   for k = 0, 1, 2, ...:
///     alpha_k = (r_k' z_k) / (p_k' V p_k)
///     x_{k+1} = x_k + alpha_k * p_k
///     r_{k+1} = r_k - alpha_k * V * p_k
///     z_{k+1} = M^-1 * r_{k+1}
///     beta_k  = (r_{k+1}' z_{k+1}) / (r_k' z_k)
///     p_{k+1} = z_{k+1} + beta_k * p_k
pub fn solve_pcg_internal(
    y: &Array1<f64>,
    x: &Array2<f64>,
    g_list: &[(Array2<f64>, String)],
    sigma2: &[f64],
    n: usize,
    n_random: usize,
    max_iter: usize,
    tol: f64,
) -> StdResult<BlupResult, SolverError> {
    // Build V
    let sigma2_e = sigma2[n_random];
    let mut v = Array2::<f64>::eye(n) * sigma2_e;
    for (i, (g, _)) in g_list.iter().enumerate() {
        v += &(g * sigma2[i]);
    }

    // Jacobi preconditioner: M = diag(V)
    let m_diag: Array1<f64> = (0..n)
        .map(|i| v[[i, i]].max(1e-10))
        .collect::<Vec<f64>>()
        .into();

    // Solve for Py via PCG
    // First solve fixed effects projection:
    // Py = V^-1 y - V^-1 X (X'V^-1 X)^-1 X'V^-1 y
    let vinv_y = pcg_solve(&v, y, &m_diag, max_iter, tol)?;
    let vinv_x = solve_vinv_x(&v, x, &m_diag, max_iter, tol)?;

    // Fixed effects: b = (X'V^-1 X)^-1 X'V^-1 y
    let xtvinvx = x.t().dot(&vinv_x);
    let xtviny = x.t().dot(&vinv_y);

    // Small system — use Cholesky directly
    let chol_xx = xtvinvx.cholesky(UPLO::Lower)
        .map_err(|_| SolverError::NotPositiveDefinite)?;
    let fixed_effects = chol_xx.solve(&xtviny)
        .map_err(|_| SolverError::NotPositiveDefinite)?;

    // Py = V^-1 y - V^-1 X * b
    let py = &vinv_y - &vinv_x.dot(&fixed_effects);

    // EBV per random effect: u_i = G_i * Py * sigma2_i
    let gebv: Vec<Array1<f64>> = g_list.iter()
        .enumerate()
        .map(|(i, (g, _))| g.dot(&py) * sigma2[i])
        .collect();

    let labels: Vec<String> = g_list.iter()
        .map(|(_, label)| label.clone())
        .collect();

    Ok(BlupResult::new(
        gebv,
        labels,
        fixed_effects,
        "pcg",
        max_iter,
    ))
}

/// Core PCG solver: solve V * x = b
fn pcg_solve(
    v: &Array2<f64>,
    b: &Array1<f64>,
    m_diag: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> StdResult<Array1<f64>, SolverError> {
    let n = b.len();
    let mut x = Array1::<f64>::zeros(n);
    let mut r = b - &v.dot(&x);
    let mut z: Array1<f64> = &r / m_diag;
    let mut p = z.clone();
    let mut rz = r.dot(&z);

    for _iter in 0..max_iter {
        let vp = v.dot(&p);
        let pvp = p.dot(&vp);

        if pvp.abs() < 1e-15 {
            break;
        }

        let alpha = rz / pvp;
        x = &x + &(&p * alpha);
        r = &r - &(&vp * alpha);

        // Check convergence: ||r|| / ||b||
        let res_norm = r.dot(&r).sqrt();
        let b_norm = b.dot(b).sqrt().max(1e-15);
        if res_norm / b_norm < tol {
            return Ok(x);
        }

        z = &r / m_diag;
        let rz_new = r.dot(&z);
        let beta = rz_new / rz.max(1e-15);
        p = &z + &(&p * beta);
        rz = rz_new;
    }

    // Return best solution even if not fully converged
    Ok(x)
}

/// Solve V * X_out = X column by column via PCG
fn solve_vinv_x(
    v: &Array2<f64>,
    x: &Array2<f64>,
    m_diag: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> StdResult<Array2<f64>, SolverError> {
    let n = x.nrows();
    let c = x.ncols();
    let mut vinv_x = Array2::<f64>::zeros((n, c));

    for j in 0..c {
        let x_col = x.column(j).to_owned();
        let sol = pcg_solve(v, &x_col, m_diag, max_iter, tol)?;
        vinv_x.column_mut(j).assign(&sol);
    }
    Ok(vinv_x)
}

/// Extendr entry point for PCG solver
#[extendr]
pub fn solve_pcg(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    sigma2: &[f64],
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

    let y_arr = Array1::from_vec(y.to_vec());
    let x_arr = Array2::from_shape_vec(
        (x.nrows(), x.ncols()),
        x.data().to_vec()
    ).map_err(|e| Error::from(e.to_string()))?;

    let mut g_matrices: Vec<(Array2<f64>, String)> = Vec::new();
    for (name, robj) in g_list.iter() {
        let g_rmat = RMatrix::<f64>::try_from(robj)
            .map_err(|_| Error::from(
                format!("G matrix '{}' is not a numeric matrix", name)
            ))?;
        let g = Array2::from_shape_vec(
            (g_rmat.nrows(), g_rmat.ncols()),
            g_rmat.data().to_vec()
        ).map_err(|e| Error::from(e.to_string()))?;
        g_matrices.push((g, name.to_string()));
    }

    let result = solve_pcg_internal(
        &y_arr, &x_arr, &g_matrices, sigma2,
        n, n_random, max_iter as usize, tol
    ).map_err(|e| Error::from(e.to_string()))?;

    let total_gebv = result.total_gebv();
    Ok(list!(
        fixed_effects = result.fixed_effects.to_vec(),
        total_gebv    = total_gebv.to_vec(),
        solver        = result.solver,
        n_iter        = result.n_iter as i32
    ))
}