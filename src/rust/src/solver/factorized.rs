// src/rust/src/solver/factorized.rs

//! Cached Cholesky factorisation of $V$.
//!
//! Many downstream operations need to solve $V x = b$ with the same $V$
//! but many different $b$:
//!
//! - [`super::cholesky`] — one solve per right-hand side for EBV.
//! - [`crate::gwas::emmax`] — one solve per marker for the per-SNP Wald
//!   statistic.
//!
//! [`FactorizedV`] packages the Cholesky factor of
//! $V = \sum_i G_i \sigma^2_i + I \sigma^2_e$ together with the dependent
//! quantities (factor of $X' V^{-1} X$, residual projection) that all
//! consumers need. Computing them once and passing the handle around
//! eliminates redundant factorisation work.
//!
//! ## Lifetime / mutability
//!
//! `FactorizedV` is constructed from owned matrices (it stores its own
//! Cholesky factors). It is `&self`-immutable on consumption, so multiple
//! threads can call `solve(...)` concurrently — needed by the parallel
//! per-marker loop in EMMAX.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Solve, UPLO, Diag};
use ndarray_linalg::Cholesky;
use ndarray_linalg::SolveTriangular;

use super::{BlupResult, SolverError, StdResult};

/// Pre-computed Cholesky factorisation of the mixed-model variance
/// matrix `V`.
///
/// # Definition
///
/// ```text
/// V = Σ_i σ²_i · G_i + σ²_e · I,
/// V = L · L'    (Cholesky, L lower triangular).
/// ```
///
/// `L` is computed once at construction (one `O(n³)` factorisation
/// via `ndarray_linalg::Cholesky`) and stored. Every subsequent
/// solve uses two `O(n²)` triangular back-substitutions on this
/// cached `L`, so reusing the same `FactorizedV` across calls is
/// the difference between `O(n³)` and `O(n²)` per right-hand side.
///
/// # Lifecycle
///
/// `FactorizedV` is constructed once per REML iteration (after the
/// σ² update) and passed by reference to the BLUP solver and the
/// EMMAX per-marker loop. It is **read-only** after construction
/// (`&self` methods), so it can be shared across threads — used by
/// the parallel `rayon` loop in EMMAX without locking.
pub struct FactorizedV {
    /// Cholesky factor L (lower triangular). `L · L' = V`.
    pub l: Array2<f64>,
    /// Number of individuals; matches `l.nrows()` and `l.ncols()`.
    pub n: usize,
    /// Variance components used to build V, kept for reference / diagnostics.
    pub sigma2: Vec<f64>,
}

impl FactorizedV {
    /// Build `V = Σ σ²_i · G_i + σ²_e · I` and Cholesky-factor it.
    ///
    /// # Errors
    ///
    /// Returns `SolverError::NotPositiveDefinite` if the Cholesky
    /// fails. In practice this happens when (a) some `σ²_i < 0` from
    /// AI-REML producing a negative variance, or (b) the relationship
    /// matrices `G_i` are near-singular (highly co-linear), making V
    /// numerically rank-deficient. The adaptive REML dispatcher
    /// catches the first case and falls back to EM-REML.
    pub fn new(
        g_list: &[(Array2<f64>, String)],
        sigma2: &[f64],
        n: usize,
    ) -> StdResult<Self, SolverError> {
        let sigma2_e = sigma2[g_list.len()];

        // Build V = sum(G_i * sigma2_i) + I * sigma2_e
        let mut v = Array2::<f64>::eye(n) * sigma2_e;
        for (i, (g, _)) in g_list.iter().enumerate() {
            v += &(g * sigma2[i]);
        }

        // Cholesky factorization
        let l = v.cholesky(UPLO::Lower)
            .map_err(|_| SolverError::NotPositiveDefinite)?;

        Ok(Self { l, n, sigma2: sigma2.to_vec() })
    }

    /// Solve `V · x = b` for a single right-hand side using the cached
    /// Cholesky factor.
    ///
    /// # Algorithm
    ///
    /// Given `V = L · L'`, the equation `V · x = b` decomposes into
    /// two triangular solves:
    ///
    /// ```text
    /// 1. Forward:  L · y = b      (lower triangular, O(n²))
    /// 2. Backward: L' · x = y     (upper triangular, O(n²))
    /// ```
    ///
    /// Total cost: 2 · n² flops per right-hand side, vs. n³ for a
    /// fresh factorisation. This is why `FactorizedV` exists.
    pub fn solve_vec(&self, b: &Array1<f64>) -> StdResult<Array1<f64>, SolverError> {
        // Step 1 — forward substitution L · y = b.
        let y = self.l.solve_triangular(UPLO::Lower, Diag::NonUnit, b)
            .map_err(|_| SolverError::NotPositiveDefinite)?;
        // Step 2 — backward substitution L' · x = y.
        self.l.t().to_owned().solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
            .map_err(|_| SolverError::NotPositiveDefinite)
    }

    /// Solve `V · X = B` for multiple right-hand sides column by column.
    ///
    /// Each column is solved independently via [`solve_vec`]; the
    /// Cholesky factor is reused so the total cost is `2 · ncols ·
    /// n²` rather than the `n³ · ncols` of a naive approach.
    ///
    /// [`solve_vec`]: Self::solve_vec
    pub fn solve_mat(&self, b: &Array2<f64>) -> StdResult<Array2<f64>, SolverError> {
        let ncols = b.ncols();
        let mut out = Array2::<f64>::zeros((self.n, ncols));
        for j in 0..ncols {
            let col = b.column(j).to_owned();
            let sol = self.solve_vec(&col)?;
            out.column_mut(j).assign(&sol);
        }
        Ok(out)
    }

    /// Compute the REML projection `P y = V⁻¹ y − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹ y`.
    ///
    /// # Why this matters
    ///
    /// `P` is the orthogonal projector onto the residual subspace
    /// after accounting for fixed effects `X`. The REML score, AI
    /// matrix, and EMMAX test statistic all evaluate `P · y` rather
    /// than `V⁻¹ · y` directly. Forming `P` explicitly costs O(n²)
    /// memory; this routine computes only `P y` and avoids that.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. φ_X = V⁻¹ X            via solve_mat (n × c)
    /// 2. φ_y = V⁻¹ y            via solve_vec (n)
    /// 3. β̂   = (X' φ_X)⁻¹ X' φ_y    (c × c symmetric solve)
    /// 4. P y = φ_y − φ_X · β̂
    /// ```
    ///
    /// The c × c system in step 3 is the small fixed-effects
    /// projection. We use a direct `ndarray_linalg::Solve` because
    /// `c` is typically O(1)–O(10) and a non-iterative solver
    /// is fastest at that scale.
    pub fn compute_py(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> StdResult<Array1<f64>, SolverError> {
        // Step 1–2: V⁻¹ X and V⁻¹ y via cached Cholesky.
        let phi1 = self.solve_mat(x)?;      // V⁻¹ X
        let phi2 = self.solve_vec(y)?;       // V⁻¹ y

        // Step 3: fixed-effect estimate β̂ from the small c × c system.
        let xtvinvx = x.t().dot(&phi1);      // X' V⁻¹ X
        let xtviny  = x.t().dot(&phi2);      // X' V⁻¹ y

        let coef = xtvinvx.solve(&xtviny)
            .map_err(|_| SolverError::NotPositiveDefinite)?;

        // Step 4: P y = V⁻¹ y − V⁻¹ X · β̂.
        Ok(phi2 - phi1.dot(&coef))
    }

    /// Solve the full BLUP system using the cached Cholesky factor.
    ///
    /// # Mixed-model equations
    ///
    /// Given fitted variance components σ², the mixed-model
    /// equations are
    ///
    /// ```text
    /// β̂   = (X' V⁻¹ X)⁻¹ X' V⁻¹ y                     (fixed effects)
    /// û_i = σ²_i · G_i · P y,    P = V⁻¹ − V⁻¹ X (X' V⁻¹ X)⁻¹ X' V⁻¹
    /// ```
    ///
    /// where `i` indexes random-effect components. `û_i` is
    /// Henderson's mixed-model-equation BLUP for component `i`;
    /// summing across `i` gives the total GEBV.
    ///
    /// # Algorithm
    ///
    /// 1. Compute `P y` via [`compute_py`].
    /// 2. Recompute `V⁻¹ X` and `V⁻¹ y` (cheap, reuses the factor)
    ///    to extract `β̂`.
    /// 3. Form `û_i = σ²_i · (G_i · P y)` for every component.
    /// 4. Package into a [`BlupResult`] with `solver = "cholesky"`
    ///    and `n_iter = 0` (Cholesky is exact, no iteration).
    ///
    /// # Performance
    ///
    /// All three steps reuse the cached Cholesky factor — total
    /// cost is dominated by the per-component matrix-vector
    /// products `G_i · P y` (each `O(n²)`). The expensive
    /// `O(n³)` factorisation happened once at `FactorizedV::new`.
    ///
    /// [`compute_py`]: Self::compute_py
    pub fn solve_blup(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        g_list: &[(Array2<f64>, String)],
        _sigma2: &[f64],
    ) -> StdResult<BlupResult, SolverError> {
        let sigma2 = &self.sigma2;

        // Compute Py using stored factor
        let py = self.compute_py(y, x)?;

        // Fixed effects: b = (X'V^-1 X)^-1 X'V^-1 y
        let phi1 = self.solve_mat(x)?;
        let phi2 = self.solve_vec(y)?;
        let xtvinvx = x.t().dot(&phi1);
        let xtviny  = x.t().dot(&phi2);
        let fixed_effects = xtvinvx.solve(&xtviny)
            .map_err(|_| SolverError::NotPositiveDefinite)?;

        // EBV: u_i = G_i * Py * sigma2_i
        let gebv: Vec<Array1<f64>> = g_list.iter()
            .enumerate()
            .map(|(i, (g, _))| g.dot(&py) * sigma2[i])
            .collect();

        let labels: Vec<String> = g_list.iter()
            .map(|(_, label)| label.clone())
            .collect();

        Ok(BlupResult::new(gebv, labels, fixed_effects, "cholesky", 0))
    }
}