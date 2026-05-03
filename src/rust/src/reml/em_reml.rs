use ndarray::{Array1, Array2};

use super::{RemlData, RemlError, VarianceComponents, StdResult};
use super::he_regression::compute_reml_loglik;
use crate::utils::linalg::{compute_py, solve_matrix};

/// Expectation-Maximization REML (EM-REML)
/// Dempster et al. (1977), Meyer (1989)
///
/// Update equations (V-based):
///   sigma2_i[t+1] = sigma2_i[t]^2 / n_i * (y'P G_i Py + tr(G_i * C_ii))
///   sigma2_e[t+1] = sigma2_e[t]^2 / n * (y'PPy + tr(C_ee))
///
/// where:
///   P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1
///   C_ii = sigma2_i * (I - sigma2_i * G_i * P)  [posterior covariance]
///   tr(G_i * C_ii) approximated efficiently via trace formula
///
/// EM-REML is guaranteed to converge but slowly (~100-1000 iterations)
/// Used as fallback when AI-REML diverges
pub fn run_em_reml(
    data: &RemlData,
    init_sigma2: &[f64],
    max_iter: usize,
    tol: f64,
) -> StdResult<VarianceComponents, RemlError> {
    let n = data.n;
    let n_random = data.n_random;
    let n_params = n_random + 1;

    if init_sigma2.len() != n_params {
        return Err(RemlError::InvalidInput(
            format!("init_sigma2 length {} != n_params {}",
                init_sigma2.len(), n_params)
        ));
    }

    let mut sigma2 = init_sigma2.to_vec();
    let mut loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Build V from current sigma2
        let v = data.build_v(&sigma2);

        // Compute Py
        let py = compute_py(&v, &data.y, &data.x)
            .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

        // Compute new sigma2 via EM update equations
        let sigma2_new = em_update(
            &v, &data.x, &py, &data.g_list, &sigma2, n, n_random
        )?;

        // Check convergence: max relative change in sigma2
        let max_change = sigma2_new.iter()
            .zip(sigma2.iter())
            .map(|(&new, &old)| ((new - old) / old.abs().max(1e-10)).abs())
            .fold(0.0f64, f64::max);

        sigma2 = sigma2_new;

        if max_change < tol {
            converged = true;
            let v_final = data.build_v(&sigma2);
            loglik = compute_reml_loglik(&v_final, &data.y, &data.x)
                .map_err(|e| RemlError::LinAlgError(e.to_string()))?;
            break;
        }

        // Log-likelihood every 10 iterations for monitoring
        if iter % 10 == 0 {
            let v_curr = data.build_v(&sigma2);
            loglik = compute_reml_loglik(&v_curr, &data.y, &data.x)
                .unwrap_or(f64::NEG_INFINITY);
        }
    }

    // Final log-likelihood if not converged
    if !converged {
        let v_final = data.build_v(&sigma2);
        loglik = compute_reml_loglik(&v_final, &data.y, &data.x)
            .unwrap_or(f64::NEG_INFINITY);
    }

    Ok(VarianceComponents::new(
        sigma2,
        loglik,
        n_iter,
        "EM",
        converged,
    ))
}

/// EM update for all variance components
fn em_update(
    v: &Array2<f64>,
    x: &Array2<f64>,
    py: &Array1<f64>,
    g_list: &[(Array2<f64>, String)],
    sigma2: &[f64],
    n: usize,
    n_random: usize,
) -> StdResult<Vec<f64>, RemlError> {
    let mut sigma2_new = Vec::with_capacity(n_random + 1);

    // Precompute V^-1 X dan (X'V^-1 X)^-1 sekali
    let vinv_x = solve_matrix(v, x)
        .map_err(|_| RemlError::SingularAI)?;
    let xtvinvx = x.t().dot(&vinv_x);

    for (i, (g, _)) in g_list.iter().enumerate() {
        let g_py = g.dot(py);
        let ytpgipy = py.dot(&g_py);

        // tr(V^-1 G) — exact
        let vinv_g = solve_matrix(v, g)
            .map_err(|_| RemlError::SingularAI)?;
        let tr_vinv_g: f64 = (0..n).map(|i| vinv_g[[i, i]]).sum();

        // Correction: tr((X'V^-1 X)^-1 X'V^-1 G V^-1 X)
        let vinv_g_vinv_x = vinv_g.dot(&vinv_x);
        let xt_vinv_g_vinv_x = x.t().dot(&vinv_g_vinv_x);
        let correction = solve_matrix(&xtvinvx, &xt_vinv_g_vinv_x)
            .map_err(|_| RemlError::SingularAI)?;
        let tr_correction: f64 = (0..correction.nrows())
            .map(|i| correction[[i, i]]).sum();

        // tr(P G_i) = tr(V^-1 G_i) - correction
        let tr_pg = tr_vinv_g - tr_correction;

        // EM update: sigma2_new = sigma2^2/n * (y'PGPy + tr(PG))
        let s2_i = sigma2[i];
        let s2_new = s2_i + (s2_i * s2_i / n as f64) * (ytpgipy - tr_pg);
        sigma2_new.push(s2_new.max(1e-6));
    }

    // Residual variance update
    let ytppy = py.dot(py);

    // tr(V^-1)
    let identity = Array2::<f64>::eye(n);
    let vinv = solve_matrix(v, &identity)
        .map_err(|_| RemlError::SingularAI)?;
    let tr_vinv: f64 = (0..n).map(|i| vinv[[i, i]]).sum();

    // Correction for residual: tr((X'V^-1 X)^-1 X'V^-2 X)
    let vinv2_x = solve_matrix(v, &vinv_x)
        .map_err(|_| RemlError::SingularAI)?;
    let xt_vinv2_x = x.t().dot(&vinv2_x);
    let correction_e = solve_matrix(&xtvinvx, &xt_vinv2_x)
        .map_err(|_| RemlError::SingularAI)?;
    let tr_correction_e: f64 = (0..correction_e.nrows())
        .map(|i| correction_e[[i, i]]).sum();

    let tr_p = tr_vinv - tr_correction_e;

    let s2_e = sigma2[n_random];
    let s2_e_new = s2_e + (s2_e * s2_e / n as f64) * (ytppy - tr_p);
    sigma2_new.push(s2_e_new.max(1e-6));

    Ok(sigma2_new)
}