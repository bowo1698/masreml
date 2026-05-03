/// src/utils/linalg.rs
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use rayon::ThreadPoolBuilder;
use crate::matrix::MatrixError;

pub fn solve_matrix(
    v: &Array2<f64>,
    b: &Array2<f64>,
) -> std::result::Result<Array2<f64>, MatrixError> {
    // Solve V * X = B column by column
    let ncols = b.ncols();
    let nrows = b.nrows();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for j in 0..ncols {
        let col = b.column(j).to_owned();
        let sol = v.solve(&col)
            .map_err(|_| MatrixError::NotPositiveDefinite)?;
        out.column_mut(j).assign(&sol);
    }
    Ok(out)
}

/// Compute projection matrix Py = V^-1 y - V^-1 X (X'V^-1 X)^-1 X'V^-1 y
/// Uses Cholesky factorization of V (avoid direct inversion)
///
/// Returns: Py vector (n)
pub fn compute_py(
    v: &Array2<f64>,
    y: &Array1<f64>,
    x: &Array2<f64>,
) -> std::result::Result<Array1<f64>, MatrixError> {

    // Solve V * phi1 = X (column by column)
    let phi1 = solve_matrix(v, x)?;

    // Solve V * phi2 = y
    let phi2 = v.solve(y)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;

    // X'V^-1 X = X' * phi1
    let xtvinvx = x.t().dot(&phi1);

    // X'V^-1 y = X' * phi2
    let xtviny = x.t().dot(&phi2);

    // Solve (X'V^-1 X) * coef = X'V^-1 y
    let coef = xtvinvx.solve(&xtviny)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;

    // Py = phi2 - phi1 * coef
    Ok(phi2 - phi1.dot(&coef))
}

/// Compute log determinant of V via Cholesky
/// log|V| = 2 * sum(log(diag(L)))
pub fn log_det_cholesky(v: &Array2<f64>) -> std::result::Result<f64, MatrixError> {
    use ndarray_linalg::{Cholesky, UPLO};
    let l = v.cholesky(UPLO::Lower)
        .map_err(|_| MatrixError::NotPositiveDefinite)?;
    // Diagonal of L stored in factor
    let log_det = l.diag()
        .iter()
        .map(|&x: &f64| x.ln())
        .sum::<f64>() * 2.0;
    Ok(log_det)
}

/// Set number of threads for BLAS and Rayon
/// Prevents thread oversubscription
pub fn set_num_threads(n_threads: usize) {
    // Rayon global thread pool
    let _ = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global();

    // OpenBLAS threads via env variable
    // ndarray-linalg respects OPENBLAS_NUM_THREADS
    std::env::set_var("OPENBLAS_NUM_THREADS", n_threads.to_string());
}