//! Adaptive REML algorithm dispatcher.
//!
//! Single entry point for the R-side `run_reml(...)` call. Internally
//! cascades through the three REML algorithms in [`super`]:
//!
//! 1. **HE regression** ([`super::he_regression`]) for starting values.
//!    Always run; cheap, closed-form, robust.
//! 2. **AI-REML** ([`super::ai_reml`]) for fast Newton-style convergence
//!    near the optimum. Aborted if it produces a negative variance
//!    component or fails to converge within `max_iter`.
//! 3. **EM-REML** ([`super::em_reml`]) as the always-non-negative
//!    fallback when AI fails.
//!
//! ## Dispatcher logic
//!
//! - If the user passes `method = "ai"`, `"em"`, or `"he"` explicitly,
//!   that algorithm runs alone (no fallback). Useful for diagnostic
//!   comparisons.
//! - With `method = "auto"` (default), HE → AI → EM cascade applies.
//! - Thread count for parallel sections (HE pairwise products, parallel
//!   $G$ matrix-vector products) is set via
//!   [`crate::utils::linalg::set_num_threads`] using `n_threads` from
//!   the caller; default uses all available cores.
//!
//! ## Return value
//!
//! [`VarianceComponents`] carries the final $\sigma^2$ vector, the
//! method actually used (`"ai"`, `"em"`, or `"he"`), iteration count,
//! and the REML log-likelihood — enough for downstream R code to surface
//! a clean convergence diagnostic to the user.

use ndarray::Array2;
use extendr_api::prelude::*;
use std::result::Result as StdResult;

use super::{RemlData, RemlError, VarianceComponents};
use super::he_regression::run_he_regression;
use super::ai_reml::run_ai_reml;
use super::em_reml::run_em_reml;
use crate::utils::linalg::set_num_threads;

/// Adaptive REML algorithm selection
///
/// Strategy:
///   n < 50,000:
///     1. HE regression → starting values
///     2. AI-REML → fast convergence
///     3. AI diverge? → EM-REML fallback
///   n >= 50,000:
///     1. HE regression only (O(n^2), no iteration)
///
/// Manual override via method parameter:
///   "auto" → adaptive strategy above
///   "HE"   → HE only
///   "AI"   → AI-REML directly (uniform starting values)
///   "EM"   → EM-REML directly
///   "HI"   → HE → AI-REML (no EM fallback)
pub fn run_reml(
    y: &[f64],
    x: RMatrix<f64>,
    g_list: List,
    method: &str,
    max_iter: i32,
    tol: f64,
    n_threads: i32,
) -> List {
    // Set thread count
    let n_threads_usize = if n_threads <= 0 {
        rayon::current_num_threads()
    } else {
        n_threads as usize
    };
    set_num_threads(n_threads_usize);

    // Parse inputs
    let result = parse_and_run(y, x, g_list, method, max_iter as usize, tol);

    match result {
        Ok(vc) => variance_components_to_list(vc),
        Err(e) => list!(
            error     = e.to_string(),
            sigma2    = ().into_robj(),
            h2        = ().into_robj(),
            loglik    = ().into_robj(),
            n_iter    = ().into_robj(),
            algorithm = ().into_robj(),
            converged = false
        ),
    }
}

/// Internal: parse R inputs and dispatch to algorithm
fn parse_and_run(
    y_slice: &[f64],
    x_rmat: RMatrix<f64>,
    g_rlist: List,
    method: &str,
    max_iter: usize,
    tol: f64,
) -> StdResult<VarianceComponents, RemlError> {
    use ndarray::Array1;
    let n = y_slice.len();

    // Parse y
    let y = Array1::from_vec(y_slice.to_vec());

    // Parse X
    let x_t = Array2::from_shape_vec(
        (x_rmat.ncols(), x_rmat.nrows()),
        x_rmat.data().to_vec()
    ).map_err(|e| RemlError::InvalidInput(e.to_string()))?;
    let x = x_t.reversed_axes().to_owned();

    // Parse G list: named list of n×n matrices
    let mut g_matrices: Vec<(Array2<f64>, String)> = Vec::new();
    for (name, robj) in g_rlist.iter() {
        let g_rmat = RMatrix::<f64>::try_from(robj)
            .map_err(|_| RemlError::InvalidInput(
                format!("G matrix '{}' is not a numeric matrix", name)
            ))?;
        let g_n = g_rmat.nrows();
        let g_t = Array2::from_shape_vec(
            (g_n, g_n),
            g_rmat.data().to_vec()
        ).map_err(|e| RemlError::InvalidInput(e.to_string()))?;
        let g = g_t.reversed_axes().to_owned();

        g_matrices.push((g, name.to_string()));
    }

    // Build RemlData
    let data = RemlData::new(y, x, g_matrices);

    // Dispatch based on method and n
    dispatch_algorithm(&data, method, max_iter, tol, n)
}

/// Algorithm dispatch logic
fn dispatch_algorithm(
    data: &RemlData,
    method: &str,
    max_iter: usize,
    tol: f64,
    n: usize,
) -> StdResult<VarianceComponents, RemlError> {
    // Threshold for HE-only mode
    const N_LARGE: usize = 50_000;

    match method {
        // Force HE only
        "HE" => {
            run_he_regression(data, tol)
        },

        // Force AI-REML with uniform starting values
        "AI" => {
            let init = uniform_init(data.n_random, 0.3, 0.4);
            run_ai_reml(data, &init, max_iter, tol)
        },

        // Force EM-REML with uniform starting values
        "EM" => {
            let init = uniform_init(data.n_random, 0.3, 0.4);
            run_em_reml(data, &init, max_iter, tol)
        },

        // HE → AI-REML (no EM fallback)
        "HI" => {
            let he_result = run_he_regression(data, tol)?;
            run_ai_reml(data, &he_result.sigma2, max_iter, tol)
        },

        // Auto: adaptive strategy
        _ => {
            if n >= N_LARGE {
                // Large data: HE only
                run_he_regression(data, tol)
            } else {
                // Small/medium: HE → AI → EM fallback
                adaptive_hi_em(data, max_iter, tol)
            }
        }
    }
}

/// Adaptive HE → AI-REML → EM-REML fallback
fn adaptive_hi_em(
    data: &RemlData,
    max_iter: usize,
    tol: f64,
) -> StdResult<VarianceComponents, RemlError> {
    // Step 1: HE regression for starting values
    let he_result = run_he_regression(data, tol)?;

    // Step 2: AI-REML from HE starting values
    let ai_result = run_ai_reml(
        data, &he_result.sigma2, max_iter, tol
    );

    match ai_result {
        // AI-REML converged
        Ok(vc) if vc.converged => Ok(vc),

        // AI-REML did not converge or failed → EM fallback
        Ok(vc) => {
            // Use AI partial results as EM starting values
            run_em_reml(data, &vc.sigma2, max_iter, tol)
                .map(|mut em_vc| {
                    // Record that we used HI+EM
                    em_vc.algorithm = "HI+EM".to_string();
                    em_vc
                })
        },

        Err(_) => {
            // AI failed entirely → restart EM from HE values
            run_em_reml(data, &he_result.sigma2, max_iter, tol)
                .map(|mut em_vc| {
                    em_vc.algorithm = "HE+EM".to_string();
                    em_vc
                })
        }
    }
}

/// Generate uniform starting values
/// sigma2_i = total_var * genetic_prop / n_random
/// sigma2_e = total_var * (1 - genetic_prop)
fn uniform_init(n_random: usize, genetic_prop: f64, total_var: f64) -> Vec<f64> {
    let mut init = Vec::with_capacity(n_random + 1);
    let s2_genetic = total_var * genetic_prop / n_random as f64;
    for _ in 0..n_random {
        init.push(s2_genetic);
    }
    // Residual
    init.push(total_var * (1.0 - genetic_prop));
    init
}

/// Convert VarianceComponents to R list
fn variance_components_to_list(vc: VarianceComponents) -> List {
    list!(
        sigma2    = vc.sigma2,
        h2        = vc.h2,
        loglik    = vc.loglik,
        n_iter    = vc.n_iter as i32,
        algorithm = vc.algorithm,
        converged = vc.converged,
        error     = ().into_robj(),
    )
}