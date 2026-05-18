// src/reml/ai_reml.rs

//! Average-Information REML (AI-REML).
//!
//! Implements the Newton-style update of Johnson & Thompson (1995):
//!
//! ```text
//! σ²⁽ᵗ⁺¹⁾ = σ²⁽ᵗ⁾ + AI⁻¹ · score
//! ```
//!
//! where the score and AI matrix are computed from $P = V^{-1} -
//! V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}$ applied to the response. AI averages
//! the observed and expected information matrices, which tends to be
//! better-conditioned than either alone and converges at a near-Newton
//! rate close to the optimum.
//!
//! ## Algorithm
//!
//! 1. Initialise $\sigma^2$ from HE (or user-supplied values).
//! 2. Compute $V$, factor it once via Cholesky.
//! 3. Compute $P y$ and the AI matrix.
//! 4. Solve $\Delta = AI^{-1} \cdot \mathrm{score}$ and update $\sigma^2$.
//! 5. Iterate until $\|\Delta\| < \mathrm{tol}$ or `max_iter`.
//!
//! ## When AI fails
//!
//! - Variance components heading negative — AI does not enforce
//!   non-negativity; the [`super::adaptive`] dispatcher catches this and
//!   falls back to EM.
//! - Singular AI matrix — happens when components are non-identifiable.
//! - Slow convergence with badly scaled $G$ matrices — sometimes solved
//!   by scaling $G$ to have $\mathrm{tr}(G)/n = 1$ before passing in.
//!
//! ## Reference
//!
//! Johnson, D. L. & Thompson, R. (1995). Restricted maximum likelihood
//! estimation of variance components for univariate animal models using
//! sparse matrix techniques and average information. *J. Dairy Sci.*,
//! 78:449–456.

use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

use super::{RemlData, RemlError, VarianceComponents, StdResult};
use super::he_regression::compute_reml_loglik;
use crate::utils::linalg::{compute_py, solve_matrix};

/// Average Information REML (AI-REML)
/// Johnson & Thompson (1995)
///
/// Iterative update:
///   theta[t+1] = theta[t] - (AI[t])^-1 * dL/dtheta[t]
///
/// where:
///   AI_ij = 0.5 * (Z_i u_i / sigma2_i)' P (Z_j u_j / sigma2_j)
///   dL/dsigma2_i = -0.5 * (tr(P * G_i) - y'P G_i Py)
///   dL/dsigma2_e = -0.5 * (tr(P) - y'P^2 y)
pub fn run_ai_reml(
    data: &RemlData,
    init_sigma2: &[f64],
    max_iter: usize,
    tol: f64,
) -> StdResult<VarianceComponents, RemlError> {
    let n_random = data.n_random;
    let n_params = n_random + 1; // genetic components + residual

    // Validate starting values
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

        let v = data.build_v(&sigma2);
        let py = compute_py(&v, &data.y, &data.x)
            .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

        let grad = compute_gradient(
            &v, &data.x, &py, &data.g_list, n_random
        )?;

        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        if max_grad < tol {
            converged = true;
            loglik = compute_reml_loglik(&v, &data.y, &data.x)
                .map_err(|e| RemlError::LinAlgError(e.to_string()))?;
            break;
        }

        let ai = compute_ai_matrix(
            &v, &py, &data.g_list, n_random
        )?;

        let delta = ai.solve(&Array1::from_vec(grad.clone()))
            .map_err(|_| RemlError::SingularAI)?;

        sigma2 = update_sigma2_with_step_halving(
            &sigma2, &delta, tol
        )?;

        let v_new = data.build_v(&sigma2);
        loglik = compute_reml_loglik(&v_new, &data.y, &data.x)
            .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

    }

    Ok(VarianceComponents::new(
        sigma2,
        loglik,
        n_iter,
        "AI",
        converged,
    ))
}

/// Compute gradient vector dL/dtheta
fn compute_gradient(
    v: &Array2<f64>,
    x: &Array2<f64>,
    py: &Array1<f64>,
    g_list: &[(Array2<f64>, String)],
    n_random: usize,
) -> StdResult<Vec<f64>, RemlError> {
    let mut grad = Vec::with_capacity(n_random + 1);
    let n = v.nrows();

    // Precompute V^-1 X dan (X'V^-1 X)^-1 sekali saja
    let vinv_x = solve_matrix(v, x)
        .map_err(|_| RemlError::SingularAI)?;
    let xtvinvx = x.t().dot(&vinv_x);

    // Gradient for each genetic component
    for (g, _) in g_list.iter() {
        let g_py = g.dot(py);
        let ytpgipy = py.dot(&g_py);

        // tr(V^-1 G)
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

        // tr(P G) = tr(V^-1 G) - correction
        let tr_pg = tr_vinv_g - tr_correction;
        grad.push(-0.5 * (tr_pg - ytpgipy));
    }

    // Gradient for residual variance
    // tr(P) = tr(V^-1) - tr((X'V^-1 X)^-1 X'V^-2 X)
    let identity = Array2::<f64>::eye(n);
    let vinv = solve_matrix(v, &identity)
        .map_err(|_| RemlError::SingularAI)?;
    let tr_vinv: f64 = (0..n).map(|i| vinv[[i, i]]).sum();

    let vinv2_x = solve_matrix(v, &vinv_x)
        .map_err(|_| RemlError::SingularAI)?;
    let xt_vinv2_x = x.t().dot(&vinv2_x);
    let correction_e = solve_matrix(&xtvinvx, &xt_vinv2_x)
        .map_err(|_| RemlError::SingularAI)?;
    let tr_correction_e: f64 = (0..correction_e.nrows())
        .map(|i| correction_e[[i, i]]).sum();

    let tr_p = tr_vinv - tr_correction_e;
    let ytppy = py.dot(py);
    grad.push(-0.5 * (tr_p - ytppy));

    Ok(grad)
}

/// Compute AI matrix
/// AI_ij = 0.5 * (Py)' G_i V^-1 G_j (Py)
/// For residual: AI_ie = 0.5 * (Py)' G_i (Py)
///               AI_ee = 0.5 * (Py)' (Py)
fn compute_ai_matrix(
    v: &Array2<f64>,
    py: &Array1<f64>,
    g_list: &[(Array2<f64>, String)],
    n_random: usize,
) -> StdResult<Array2<f64>, RemlError> {
    let n_params = n_random + 1;
    let mut ai = Array2::<f64>::zeros((n_params, n_params));

    // Precompute V^-1 G_i Py for each component
    // and Py itself for residual
    let mut vinv_gi_py: Vec<Array1<f64>> = Vec::with_capacity(n_random);

    for (g, _) in g_list.iter() {
        let g_py = g.dot(py);
        let v_inv_g_py = v.solve(&g_py)
            .map_err(|_| RemlError::SingularAI)?;
        vinv_gi_py.push(v_inv_g_py);
    }

    // V^-1 Py for residual component
    let vinv_py = v.solve(py)
        .map_err(|_| RemlError::SingularAI)?;

    // Fill AI matrix (symmetric)
    // Genetic-genetic block
    for i in 0..n_random {
        for j in 0..=i {
            // AI_ij = 0.5 * (Py)' G_i V^-1 G_j Py
            let ai_ij = 0.5 * vinv_gi_py[j].dot(
                &g_list[i].0.dot(py)
            );
            ai[[i, j]] = ai_ij;
            ai[[j, i]] = ai_ij;
        }
    }

    // Genetic-residual block
    for i in 0..n_random {
        let ai_ie = 0.5 * vinv_gi_py[i].dot(py);
        ai[[i, n_random]] = ai_ie;
        ai[[n_random, i]] = ai_ie;
    }

    // Residual-residual
    ai[[n_random, n_random]] = 0.5 * vinv_py.dot(py);

    Ok(ai)
}

/// Update sigma2 with step-halving to ensure all values remain positive
/// and log-likelihood does not decrease
fn update_sigma2_with_step_halving(
    sigma2_old: &[f64],
    delta: &Array1<f64>,
    tol: f64,
) -> StdResult<Vec<f64>, RemlError> {
    let mut step = 1.0f64;
    let max_halving = 10;

    for _ in 0..max_halving {
        let sigma2_new: Vec<f64> = sigma2_old.iter()
            .zip(delta.iter())
            .map(|(&s, &d)| s + step * d)
            .collect();

        if sigma2_new.iter().all(|&s| s > tol) {
            return Ok(sigma2_new);
        }
        step *= 0.5;
    }

    let sigma2_constrained: Vec<f64> = sigma2_old.iter()
        .zip(delta.iter())
        .map(|(&s, &d)| (s + step * d).max(tol * 10.0))
        .collect();

    Ok(sigma2_constrained)
}