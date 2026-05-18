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

    // ================================================================
    // AI-REML iteration (Johnson & Thompson 1995).
    //
    // For variance-component vector θ = (σ²_1, …, σ²_K, σ²_e), the
    // REML log-likelihood has gradient (score) and a Newton-style
    // update using the *average information* matrix as the curvature
    // approximation:
    //
    //     V(θ) = Σ_k σ²_k G_k + σ²_e I,                            (V)
    //     P    = V⁻¹ − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹,                  (P)
    //     s_k  = −½ [ tr(P G_k) − y' P G_k P y ]   ← score for σ²_k
    //     AI_{kl} = ½  (P y)' G_k P G_l (P y)      ← AI[k, l]
    //     θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ + AI⁻¹ · s.
    //
    // AI averages observed and expected information; it stays
    // positive definite near the optimum (unlike pure Newton), giving
    // near-quadratic convergence while avoiding the curvature
    // pathologies that derail vanilla Newton-Raphson REML.
    // ================================================================
    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Build V(θ) and solve (V, X) once; both the score and AI
        // matrix below need P = V⁻¹ − V⁻¹X (X'V⁻¹X)⁻¹ X'V⁻¹ applied
        // to y. `compute_py` exploits a Cholesky factor cache.
        let v = data.build_v(&sigma2);
        let py = compute_py(&v, &data.y, &data.x)
            .map_err(|e| RemlError::LinAlgError(e.to_string()))?;

        // Score vector (gradient of REML log-likelihood w.r.t. θ).
        // Each entry takes the form
        //     s_k = −½ ( tr(P G_k) − y' P G_k P y ).
        // The trace term uses `solve_matrix` to avoid forming P explicitly.
        let grad = compute_gradient(
            &v, &data.x, &py, &data.g_list, n_random
        )?;

        // Convergence check on the score; ‖s‖_∞ < tol means we sit on
        // the REML stationary point to first order.
        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        if max_grad < tol {
            converged = true;
            loglik = compute_reml_loglik(&v, &data.y, &data.x)
                .map_err(|e| RemlError::LinAlgError(e.to_string()))?;
            break;
        }

        // AI matrix — symmetric positive (semi-)definite at the optimum.
        // Element AI[k, l] = ½ (Py)' G_k P G_l (Py).
        let ai = compute_ai_matrix(
            &v, &py, &data.g_list, n_random
        )?;

        // Solve  AI · Δ = s  for the Newton direction. If AI is singular
        // — typically meaning two variance components are non-identifiable
        // given the data — bail out so the adaptive dispatcher can fall
        // back to EM-REML (which never inverts AI).
        let delta = ai.solve(&Array1::from_vec(grad.clone()))
            .map_err(|_| RemlError::SingularAI)?;

        // Newton step with safeguards: `update_sigma2_with_step_halving`
        // halves the step until every σ²_i stays positive (variance
        // components cannot be negative under the REML model). Step
        // halving keeps AI useful in the early iterations when starting
        // values are far from the optimum.
        sigma2 = update_sigma2_with_step_halving(
            &sigma2, &delta, tol
        )?;

        // Recompute log-likelihood for monitoring / final reporting.
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

/// Compute the REML score vector `s = ∂ log L_REML / ∂ θ`.
///
/// # Mathematical form
///
/// For each genetic component `k` (one `G_k` matrix and its
/// variance `σ²_k`):
///
/// ```text
/// s_k = −½ · ( tr(P · G_k) − y' P G_k P y ).
/// ```
///
/// The last entry (residual variance) replaces `G_k` with the
/// identity `I`:
///
/// ```text
/// s_e = −½ · ( tr(P) − y' P² y ).
/// ```
///
/// `P = V⁻¹ − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹` is the REML projection
/// onto the residual subspace. We never form `P` explicitly:
/// `tr(P G) = tr(V⁻¹ G) − tr((X' V⁻¹ X)⁻¹ X' V⁻¹ G V⁻¹ X)` lets us
/// evaluate the trace via small auxiliary solves.
///
/// # Algorithm
///
/// 1. Pre-compute `V⁻¹ X` and `X' V⁻¹ X` once outside the per-G
///    loop (reused for every component).
/// 2. For each `G_k`:
///    - Solve `V⁻¹ G_k`, trace it for `tr(V⁻¹ G_k)`.
///    - Form the correction `(X' V⁻¹ X)⁻¹ · (X' V⁻¹ G_k V⁻¹ X)`,
///      trace it.
///    - Subtract to get `tr(P G_k)`.
///    - Compute `y' P G_k P y = (P y)' G_k (P y)`.
///    - Score entry `s_k = −½ · (tr_pg − ytpgipy)`.
/// 3. Repeat for the residual variance (`G_e = I`), needing
///    `V⁻¹` itself and `V⁻²` (computed as `V⁻¹ · V⁻¹ X` reused).
///
/// # Errors
///
/// `RemlError::SingularAI` on any failed linear solve — typically
/// means `V` or `(X' V⁻¹ X)` became near-singular during the
/// iteration. The adaptive dispatcher upstream catches this and
/// can fall back to EM-REML.
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

/// Compute the **average-information** (AI) matrix used as the
/// REML curvature approximation in the Newton-style update.
///
/// # Mathematical form
///
/// The exact observed-information matrix `I` is expensive (requires
/// second derivatives with traces over `P G_i P G_j`). The expected
/// information `E[I]` is cheap but only valid at the optimum.
/// Johnson & Thompson (1995) showed that the **average** of the
/// two — the AI matrix — is both cheap and yields near-Newton
/// convergence near the optimum:
///
/// ```text
/// AI_{k, l} = ½ · (P y)' G_k V⁻¹ G_l (P y)        for k, l ≤ n_random,
/// AI_{k, e} = ½ · (P y)' G_k V⁻¹ (P y),           genetic × residual,
/// AI_{e, e} = ½ · (P y)' V⁻¹ (P y)                residual × residual.
/// ```
///
/// AI is symmetric positive semi-definite at the optimum, so the
/// Newton step `Δ = AI⁻¹ · score` is well-defined whenever the
/// variance components are identifiable.
///
/// # Algorithm
///
/// 1. Pre-compute `V⁻¹ · (G_i · P y)` for every genetic component;
///    this gives the `n_random` vectors we will reuse.
/// 2. Also pre-compute `V⁻¹ · P y` for the residual entries.
/// 3. Fill the symmetric AI matrix entry by entry; we only compute
///    the lower triangle and mirror to the upper.
///
/// All inner-loop work is `O(n)` vector dots; the `O(n²)` matrix
/// operations live in step 1's solves.
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

/// Apply a Newton step `σ² ← σ² + step · Δ` with safeguards to keep
/// every variance component strictly positive.
///
/// # Why step halving
///
/// The raw Newton step `Δ = AI⁻¹ · score` can overshoot at the
/// start of REML iteration (when σ² is far from the optimum) and
/// drive one or more components negative. Negative variances are
/// inadmissible under the model, so we halve the step size
/// repeatedly until every component stays above a small floor.
///
/// # Algorithm
///
/// ```text
/// step ← 1.0
/// for at most 10 halvings:
///     candidate ← σ² + step · Δ
///     if all candidate > tol:
///         return candidate
///     step ← step / 2
/// return clipped candidate (max(., 10 · tol) element-wise)
/// ```
///
/// The 10-halving cap (`step` shrinks to 2⁻¹⁰ ≈ 1e-3 in the worst
/// case) is a defensive upper bound: in practice REML reaches a
/// positive candidate in 1–3 halvings near the optimum, and only
/// pathological starting values need more. The final clip-to-floor
/// is a last-resort safety net so the next AI iteration still has a
/// well-conditioned V.
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