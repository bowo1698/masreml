// src/rust/src/gwas/smoother.rs

use super::{GwasError, StdResult};

/// Moving average of LR values over a window of SNPs/blocks
/// Window is centered: floor(window/2) to the left and right
/// Edge handling: shrink window at boundaries (no padding)
///
/// Input:
///   lr: LR values per marker/block
///   window: number of markers/blocks in moving average
/// Output: smoothed LR values (same length as input)
pub fn moving_average(lr: &[f64], window: usize) -> StdResult<Vec<f64>, GwasError> {
    if lr.is_empty() {
        return Err(GwasError::InvalidInput("LR vector is empty".to_string()));
    }
    if window == 0 {
        return Err(GwasError::InvalidInput("Window size must be > 0".to_string()));
    }

    let n = lr.len();
    let half = window / 2;
    let mut smoothed = vec![0.0f64; n];

    for i in 0..n {
        let left  = if i >= half { i - half } else { 0 };
        let right = if i + half < n { i + half } else { n - 1 };
        let count = right - left + 1;
        let sum: f64 = lr[left..=right].iter().sum();
        smoothed[i] = sum / count as f64;
    }

    Ok(smoothed)
}

/// Compute posterior probability of marker/block having non-zero effect
/// Based on Meuwissen et al. (2024) Eq. 4:
///   PP_j = π * exp(LR_j) / (π * exp(LR_j) + (1 - π))
///
/// Numerically stable implementation:
///   If LR_j large → PP_j → 1
///   If LR_j ≈ 0   → PP_j ≈ π
///
/// Input:
///   smoothed_lr: smoothed LR values per marker/block
///   pi: prior probability of non-zero effect (default 0.001)
/// Output: posterior probabilities PP_j (same length as input)
pub fn compute_pp(smoothed_lr: &[f64], pi: f64) -> StdResult<Vec<f64>, GwasError> {
    if smoothed_lr.is_empty() {
        return Err(GwasError::InvalidInput(
            "Smoothed LR vector is empty".to_string()
        ));
    }
    if pi <= 0.0 || pi >= 1.0 {
        return Err(GwasError::InvalidInput(
            format!("pi must be in (0, 1), got {}", pi)
        ));
    }

    let pp: Vec<f64> = smoothed_lr
        .iter()
        .map(|&lr| {
            // Numerically stable: avoid exp overflow
            // PP = 1 / (1 + (1-π)/π * exp(-LR))
            let log_odds = lr + (pi / (1.0 - pi)).ln();
            if log_odds > 500.0 {
                1.0
            } else if log_odds < -500.0 {
                0.0
            } else {
                let e = log_odds.exp();
                e / (1.0 + e)
            }
        })
        .collect();

    Ok(pp)
}

/// Combined: moving average then compute PP
/// Convenience function called from lib.rs extendr entry point
pub fn smooth_and_pp(
    lr: &[f64],
    window: usize,
    pi: f64,
) -> StdResult<(Vec<f64>, Vec<f64>), GwasError> {
    let smoothed = moving_average(lr, window)?;
    let pp = compute_pp(&smoothed, pi)?;
    Ok((smoothed, pp))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_basic() {
        let lr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = moving_average(&lr, 3).unwrap();
        // index 1: mean(1,2,3) = 2.0
        assert!((smoothed[1] - 2.0).abs() < 1e-10);
        // index 2: mean(2,3,4) = 3.0
        assert!((smoothed[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_edge() {
        let lr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = moving_average(&lr, 3).unwrap();
        // index 0: mean(1,2) = 1.5 (shrink window)
        assert!((smoothed[0] - 1.5).abs() < 1e-10);
        // index 4: mean(4,5) = 4.5 (shrink window)
        assert!((smoothed[4] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pp_low_lr() {
        // LR = 0, pi = 0.001 → PP ≈ pi
        let pp = compute_pp(&[0.0], 0.001).unwrap();
        assert!((pp[0] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_compute_pp_high_lr() {
        // LR very large → PP → 1
        let pp = compute_pp(&[1000.0], 0.001).unwrap();
        assert!((pp[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_pi() {
        assert!(compute_pp(&[1.0], 0.0).is_err());
        assert!(compute_pp(&[1.0], 1.0).is_err());
    }

    #[test]
    fn test_window_zero() {
        assert!(moving_average(&[1.0], 0).is_err());
    }
}