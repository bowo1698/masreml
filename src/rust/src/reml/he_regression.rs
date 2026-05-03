use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use rayon::prelude::*;

use super::{RemlData, RemlError, VarianceComponents, StdResult};
use crate::utils::linalg::{log_det_cholesky, compute_py, solve_matrix};

/// Haseman-Elston (HE) regression for variance component estimation
///
/// Model: y_hat_i * y_hat_j = mu + sum_k( K_k[i,j] * h2_k ) + e_ij
/// where y_hat = adjusted phenotype (residuals from fixed effects)
///
/// Solves via OLS: beta = (Z'Z)^-1 Z'r
/// where Z = [1, K1_ij, K2_ij, ...] and r = y_hat_i * y_hat_j
///
/// Returns starting values for AI-REML or final estimates if n >= 50k
pub fn run_he_regression(
    data: &RemlData,
    _tol: f64,
) -> StdResult<VarianceComponents, RemlError>  {
    let n = data.n;
    let n_random = data.n_random;

    // Step 1: Adjust phenotype for fixed effects
    // y_hat = y - X(X'X)^-1 X'y = M*y where M = I - X(X'X)^-1 X'
    let y_hat = adjust_fixed_effects(&data.y, &data.x)?;

    // Step 2: Build HE regression design matrix Z and response r
    // Z is n*(n-1)/2 x (n_random + 1) [intercept + K_ij per component]
    // r is n*(n-1)/2 vector of y_hat_i * y_hat_j (upper triangle pairs)
    let n_pairs = n * (n - 1) / 2;
    let n_cols = n_random + 1; // intercept + one per G matrix

    // Build pairs in parallel
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i+1..n).map(move |j| (i, j)))
        .collect();

    // Response: r_ij = y_hat_i * y_hat_j
    let r: Vec<f64> = pairs.par_iter()
        .map(|&(i, j)| y_hat[i] * y_hat[j])
        .collect();

    // Design matrix Z: [1, K1_ij, K2_ij, ...]
    let z_data: Vec<f64> = pairs.par_iter()
        .flat_map(|&(i, j)| {
            let mut row = vec![1.0f64]; // intercept
            for (g, _) in &data.g_list {
                row.push(g[[i, j]]);
            }
            row
        })
        .collect();

    let z = Array2::from_shape_vec((n_pairs, n_cols), z_data)
        .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

    let r_arr = Array1::from_vec(r);

    // Step 3: OLS solution beta = (Z'Z)^-1 Z'r
    let ztz = z.t().dot(&z);
    let ztr = z.t().dot(&r_arr);

    let beta = ztz.solve(&ztr)
        .map_err(|_| RemlError::SingularAI)?;

    // Step 4: Convert h2 estimates to variance components
    // sigma2_i = var(y_hat) * h2_i
    let var_yhat: f64 = {
        let mean = y_hat.mean().unwrap_or(0.0);
        y_hat.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
            / (n - 1) as f64
    };

    // beta[0] = intercept (mu), beta[1..] = h2 estimates
    let mut sigma2 = Vec::with_capacity(n_random + 1);
    let mut h2_sum = 0.0f64;

    for k in 1..=n_random {
        // Constrain h2 to [0.01, 0.99] to avoid degenerate starting values
        let h2_k = beta[k].max(0.01).min(0.99);
        sigma2.push(var_yhat * h2_k);
        h2_sum += h2_k;
    }

    // sigma2_e = var(y_hat) * (1 - sum(h2_k))
    let h2_e = (1.0 - h2_sum).max(0.01);
    sigma2.push(var_yhat * h2_e);

    // Compute approximate log-likelihood at HE estimates
    let v = data.build_v(&sigma2);
    let loglik = compute_reml_loglik(&v, &data.y, &data.x)?;

    Ok(VarianceComponents::new(
        sigma2,
        loglik,
        1,
        "HE",
        true,
    ))
}

/// Adjust phenotype for fixed effects via OLS projection
/// y_hat = y - X(X'X)^-1 X'y
fn adjust_fixed_effects(
    y: &Array1<f64>,
    x: &Array2<f64>,
) -> StdResult<Array1<f64>, RemlError> {
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);

    let b_hat = xtx.solve(&xty)
        .map_err(|_| RemlError::LinAlgError(
            "X'X is singular — check fixed effects matrix".to_string()
        ))?;

    // y_hat = y - X * b_hat
    Ok(y - &x.dot(&b_hat))
}

/// Compute restricted log-likelihood
/// lnL = -0.5 * (log|V| + log|X'V^-1 X| + y'Py)
pub fn compute_reml_loglik(
    v: &Array2<f64>,
    y: &Array1<f64>,
    x: &Array2<f64>,
) -> StdResult<f64, RemlError> {

    let log_det_v = log_det_cholesky(v)
        .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

    let py = compute_py(v, y, x)
        .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

    let ytpy = y.dot(&py);

    // log|X'V^-1 X|: solve V * phi1 = X column by column
    let phi1 = solve_matrix(v, x)
        .map_err(|e| RemlError::LinAlgError(e.to_string()))?;
    let xtvinvx = x.t().dot(&phi1);
    let log_det_xtvinvx = log_det_cholesky(&xtvinvx)
        .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

    Ok(-0.5 * (log_det_v + log_det_xtvinvx + ytpy))
}