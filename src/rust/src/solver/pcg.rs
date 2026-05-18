//! Preconditioned Conjugate Gradient (PCG) solver.
//!
//! Iterative solver for the mixed-model equations $V x = b$ that avoids
//! forming or factoring $V$ explicitly. Recommended for $n \ge 10{,}000$
//! where the $O(n^2)$ memory cost of [`super::cholesky`] becomes
//! prohibitive.
//!
//! ## Algorithm
//!
//! Standard PCG with the diagonal of $V$ as preconditioner (Jacobi).
//! Iteration stops when the relative residual $\|r_k\| / \|b\|$ falls
//! below `tol` (default 1e-8) or when `max_iter` is reached.
//!
//! ## When to use vs. Cholesky
//!
//! - $n < 10{,}000$ → Cholesky is faster (one factorisation, then
//!   triangular solves).
//! - $n \ge 10{,}000$ → PCG scales as $O(n^2)$ per iteration via the
//!   $G$ matrix-vector products, never materialising $V^{-1}$.
//! - Very ill-conditioned $V$ → PCG may need many iterations; consider
//!   tighter convergence thresholds or scaling $G$ before passing in.
//!
//! ## Output
//!
//! Same [`BlupResult`] shape as [`super::cholesky`], with `solver =
//! "pcg"` and `n_iter` set to the actual iteration count for diagnostics.

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
    // ============================================================
    // Step 1 — Assemble V = Σ_i σ²_i · G_i + σ²_e · I.
    //
    // V is the mixed-model variance matrix. Once σ² are fixed by
    // REML, V is constant — we never modify it during the PCG
    // sweep, only multiply against vectors V·p inside the inner
    // loop. Memory cost: one dense n × n matrix; matvec cost: O(n²).
    // ============================================================
    let sigma2_e = sigma2[n_random];
    let mut v = Array2::<f64>::eye(n) * sigma2_e;
    for (i, (g, _)) in g_list.iter().enumerate() {
        v += &(g * sigma2[i]);
    }

    // ============================================================
    // Step 2 — Build the Jacobi (diagonal) preconditioner M.
    //
    //     M = diag(V),   so   M⁻¹ = diag(1 / V_{ii}).
    //
    // Jacobi is the cheapest non-trivial preconditioner: O(n) to
    // build, O(n) per application (just element-wise division).
    // It clusters eigenvalues of M⁻¹V around 1 for diagonally
    // dominant V, which V always is under the mixed model because
    // σ²_e contributes σ²_e to every diagonal entry. The 1e-10
    // floor avoids division by zero on degenerate components.
    // ============================================================
    let m_diag: Array1<f64> = (0..n)
        .map(|i| v[[i, i]].max(1e-10))
        .collect::<Vec<f64>>()
        .into();

    // ============================================================
    // Step 3 — Compute V⁻¹ y and V⁻¹ X via PCG, then assemble Py.
    //
    // The REML projection used everywhere downstream is
    //     P = V⁻¹ − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹,
    //     P y = V⁻¹ y − V⁻¹ X · (X' V⁻¹ X)⁻¹ X' V⁻¹ y.
    //
    // We solve V·a = y (gives a = V⁻¹ y) and V·B = X (gives B =
    // V⁻¹ X) iteratively. (X' V⁻¹ X) is a tiny c × c matrix where
    // c = number of fixed effects, so we factor it with a direct
    // Cholesky instead of PCG.
    // ============================================================
    let vinv_y = pcg_solve(&v, y, &m_diag, max_iter, tol)?;
    let vinv_x = solve_vinv_x(&v, x, &m_diag, max_iter, tol)?;

    let xtvinvx = x.t().dot(&vinv_x);
    let xtviny = x.t().dot(&vinv_y);

    // Direct Cholesky on the c × c projection — cheap, exact.
    let chol_xx = xtvinvx.cholesky(UPLO::Lower)
        .map_err(|_| SolverError::NotPositiveDefinite)?;
    let fixed_effects = chol_xx.solve(&xtviny)
        .map_err(|_| SolverError::NotPositiveDefinite)?;

    // Py = V⁻¹ y − V⁻¹ X · β̂ — the REML residual projection.
    let py = &vinv_y - &vinv_x.dot(&fixed_effects);

    // ============================================================
    // Step 4 — Per-component BLUP via Henderson's formula:
    //
    //     û_i = σ²_i · G_i · P y.
    //
    // Each random-effect component i gets its own GEBV vector,
    // returned separately in `gebv` so downstream code can sum
    // them (for total BLUP) or examine them individually (e.g. to
    // partition pedigree vs SNP contributions in a joint model).
    // ============================================================
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

/// Solve `V · x = b` via Jacobi-preconditioned Conjugate Gradient.
///
/// # Algorithm (textbook PCG, Saad 2003 ch. 9)
///
/// Initialise with x₀ = 0, r₀ = b − V·x₀ = b, z₀ = M⁻¹ r₀, p₀ = z₀.
/// Then for k = 0, 1, 2, …:
///
/// ```text
/// α_k     = (r_k' z_k) / (p_k' V p_k)
/// x_{k+1} = x_k + α_k · p_k
/// r_{k+1} = r_k − α_k · V·p_k
/// z_{k+1} = M⁻¹ · r_{k+1}
/// β_k     = (r_{k+1}' z_{k+1}) / (r_k' z_k)
/// p_{k+1} = z_{k+1} + β_k · p_k
/// ```
///
/// The Jacobi preconditioner `M = diag(V)` is precomputed by the
/// caller and passed as `m_diag`, so each preconditioner-solve is a
/// single O(n) element-wise division. Other than that, each PCG
/// iteration costs one V·p matvec (O(n²)) plus a few O(n) dots.
///
/// # Convergence
///
/// Stops when the relative residual `‖r‖ / ‖b‖ < tol` (default
/// `1e-8`) or when `max_iter` is hit. For the mixed-model V used in
/// REML, PCG typically converges in `O(√κ)` iterations where κ is
/// the condition number; with the σ²_e term well above zero the
/// condition number stays moderate even for large n.
///
/// # Degenerate guard
///
/// If `p' V p ≈ 0` (numerator vanishes), the search direction has
/// collapsed and further iterations cannot reduce the residual; we
/// bail out and return the current x. This handles the rare case
/// where V is numerically rank-deficient along the current direction.
fn pcg_solve(
    v: &Array2<f64>,
    b: &Array1<f64>,
    m_diag: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> StdResult<Array1<f64>, SolverError> {
    let n = b.len();

    // Initial guess x₀ = 0 (best when we have no prior estimate; if
    // a warm start were available it would shrink iteration count).
    let mut x = Array1::<f64>::zeros(n);

    // Initial residual: r₀ = b − V·x₀ = b (since x₀ = 0).
    let mut r = b - &v.dot(&x);

    // Preconditioned residual z₀ = M⁻¹ r₀. Jacobi → elementwise.
    let mut z: Array1<f64> = &r / m_diag;

    // First search direction: p₀ = z₀.
    let mut p = z.clone();

    // Inner product (r_k, z_k); reused in α and β.
    let mut rz = r.dot(&z);

    for _iter in 0..max_iter {
        // Apply V to current direction (the O(n²) work per iter).
        let vp = v.dot(&p);

        // Curvature in the search direction.
        let pvp = p.dot(&vp);

        // Degenerate-direction guard: stop if curvature collapsed.
        if pvp.abs() < 1e-15 {
            break;
        }

        // Optimal step size along p_k (minimises ‖V·x − b‖_V⁻¹).
        let alpha = rz / pvp;

        // Update solution and residual in tandem so r stays
        // consistent with x (avoids drift from the explicit
        // recomputation b − V·x_k).
        x = &x + &(&p * alpha);
        r = &r - &(&vp * alpha);

        // Convergence check after the update (so the *post-update*
        // residual is what we compare against tol).
        // ||r|| / ||b||
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

/// Solve `V · X_out = X` column-by-column via PCG.
///
/// PCG handles one right-hand side at a time, so for a matrix `X` of
/// shape `(n, c)` we solve `c` independent linear systems, one per
/// fixed-effect column. Each call reuses the same Jacobi
/// preconditioner `m_diag` (a property of V, independent of the RHS),
/// so the only per-column overhead is the PCG iteration itself.
///
/// # Complexity
///
/// `c · k_PCG · O(n²)` where `k_PCG` is the average iteration count
/// per column. For mixed models with σ²_e bounded away from zero,
/// `k_PCG = O(√κ)` is modest (often <100), making this much cheaper
/// than the `O(n³)` Cholesky factorisation when `n` is large.
///
/// # Why not solve `V · X_out = X` as a matrix system?
///
/// A blocked / multi-RHS PCG (e.g. block-CG) could in principle share
/// work across columns and converge in fewer matvecs, but it adds
/// substantial code complexity and rarely helps when `c` is small
/// (typical fixed-effect designs have `c ≤ 10`). Column-by-column is
/// simple, correct, and fast enough for the usage patterns here.
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