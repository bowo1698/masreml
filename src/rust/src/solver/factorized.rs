// src/rust/src/solver/factorized.rs
use ndarray::{Array1, Array2};
use ndarray_linalg::{Solve, UPLO, Diag};
use ndarray_linalg::Cholesky;
use ndarray_linalg::SolveTriangular;

use super::{BlupResult, SolverError, StdResult};

/// Pre-computed Cholesky factorization of V matrix
/// V = sum_i(G_i * sigma2_i) + I * sigma2_e
/// V = L * L'  (Cholesky)
pub struct FactorizedV {
    /// Cholesky factor L (lower triangular)
    pub l: Array2<f64>,
    /// Number of individuals
    pub n: usize,
    pub sigma2: Vec<f64>,
}

impl FactorizedV {
    /// Build V and compute Cholesky factorization
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

    pub fn solve_vec(&self, b: &Array1<f64>) -> StdResult<Array1<f64>, SolverError> {
    // V = L*L', solve V*x = b via:
    // 1. solve L*y = b  (forward substitution)
    // 2. solve L'*x = y (backward substitution)

        let y = self.l.solve_triangular(UPLO::Lower, Diag::NonUnit, b)
            .map_err(|_| SolverError::NotPositiveDefinite)?;
        self.l.t().to_owned().solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
            .map_err(|_| SolverError::NotPositiveDefinite)
    }

    /// Solve V * X = B column by column using stored L factor
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

    /// Compute Py = V^-1 y - V^-1 X (X'V^-1 X)^-1 X'V^-1 y
    pub fn compute_py(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
    ) -> StdResult<Array1<f64>, SolverError> {
        let phi1 = self.solve_mat(x)?;      // V^-1 X
        let phi2 = self.solve_vec(y)?;       // V^-1 y

        let xtvinvx = x.t().dot(&phi1);      // X'V^-1 X
        let xtviny  = x.t().dot(&phi2);      // X'V^-1 y

        let coef = xtvinvx.solve(&xtviny)
            .map_err(|_| SolverError::NotPositiveDefinite)?;

        Ok(phi2 - phi1.dot(&coef))           // Py
    }

    /// Solve BLUP using pre-factorized V
    /// Reuses L factor — no re-factorization needed
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