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

/// Top-level REML dispatcher, exposed to R via `r_run_reml`.
///
/// # Cascade strategy (`method = "auto"`, default)
///
/// - **Small / medium n (`n < 50 000`)**:
///   1. **HE regression** — closed-form starting values (always
///      runs; output feeds the next stages, never a final answer).
///   2. **AI-REML** — Newton-style iteration that usually converges
///      in 5–15 iterations near the optimum.
///   3. **EM-REML fallback** — engaged automatically if AI returns
///      a negative variance component or fails to converge within
///      `max_iter`. EM is slow but always non-negative.
///
/// - **Large n (`n ≥ 50 000`)**:
///   - **HE only**. AI/EM both need at least `O(n²)` work per
///     iteration and become impractical; HE is a single closed-form
///     evaluation that scales as `O(n²)`. Users who need a fully
///     converged REML at this scale should pre-filter to a tractable
///     subset or use the iterative path explicitly.
///
/// # Manual override (`method` parameter)
///
/// | Value  | Behaviour                                                |
/// |--------|----------------------------------------------------------|
/// | `"auto"` | Adaptive strategy above.                              |
/// | `"HE"`   | HE only. Returns immediately after the closed-form OLS. |
/// | `"AI"`   | HE → AI directly with no EM fallback. Fails if AI fails. |
/// | `"EM"`   | HE → EM directly (skip AI). Slower but always non-negative. |
/// | `"HI"`   | HE → AI but with EM fallback disabled. Used in diagnostic comparisons. |
///
/// # Thread pool
///
/// `n_threads = 0` (the default surfaced by the R wrapper) means
/// "use all available cores", which we resolve via
/// `rayon::current_num_threads()`. Any positive value pins Rayon
/// **and** OpenBLAS to that count via
/// [`crate::utils::linalg::set_num_threads`] to avoid the
/// oversubscription discussed there.
///
/// # Output
///
/// Returns an R list with `sigma2`, `h2`, `loglik`, `n_iter`,
/// `algorithm`, and `converged` fields on success. On failure the
/// list contains an `error` string and null placeholders for the
/// rest, so the R wrapper can detect the failure path without
/// pattern-matching on Rust types.
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

/// Adaptive cascade: HE regression → AI-REML → optional EM-REML
/// fallback.
///
/// # Decision tree
///
/// ```text
/// HE regression  →  starting values σ²_HE
///       │
///       ▼
/// AI-REML(σ²_HE)
///       │
///       ├── converged       → return AI result (algorithm = "AI")
///       │
///       ├── completed but not converged
///       │       → EM-REML(σ²_AI_partial),  algorithm = "HI+EM"
///       │
///       └── failed entirely (e.g. singular AI matrix)
///               → EM-REML(σ²_HE),          algorithm = "HE+EM"
/// ```
///
/// # Algorithm string in output
///
/// The returned `VarianceComponents.algorithm` records the actual
/// path taken so downstream R code can surface a meaningful
/// diagnostic to the user:
///
/// - `"AI"` — AI-REML converged from HE starts (the happy path).
/// - `"HI+EM"` — AI didn't converge; EM picked up AI's partial σ²
///   and finished the job.
/// - `"HE+EM"` — AI failed catastrophically (e.g. singular AI matrix
///   on degenerate data); EM ran from HE starts only.
///
/// # Why this cascade
///
/// AI-REML is the fast path (super-linear convergence near the
/// optimum) but can fail on ill-conditioned data. EM is the slow,
/// robust fallback that always converges to a non-negative
/// solution. HE is cheap enough to always run first because its
/// closed-form output gives both methods a high-quality starting
/// point — much better than uniform initialisation, which can
/// trigger 100+ extra iterations before either method finds the
/// optimum's basin.
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

/// Generate uniform starting values for AI / EM when HE is not used.
///
/// # Formula
///
/// ```text
/// σ²_k = total_var · genetic_prop / n_random,   k = 1..n_random
/// σ²_e = total_var · (1 − genetic_prop)
/// ```
///
/// Default `genetic_prop = 0.3` (h² ≈ 0.3) and `total_var = 0.4`
/// are reasonable for many livestock / aquaculture traits; they
/// are intentionally rough because AI/EM iterate to the optimum
/// regardless of starting values, just faster with HE starts than
/// with these uniform ones.
///
/// # When this is used
///
/// Only when the user explicitly passes `method = "AI"` or
/// `method = "EM"` and asks the algorithm to skip the HE
/// regression. The adaptive cascade in [`adaptive_hi_em`] always
/// prefers HE-derived starting values over these.
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