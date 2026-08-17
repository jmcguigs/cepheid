//! DFT sampling window.
//!
//! This is **not** a Lomb–Scargle periodogram. A floating-mean GLS of
//! `y ≡ 1` has residual 0 and power 0 at every frequency — that recipe
//! is a no-op and must not be used as \(W(f)\).
//!
//! \[
//! W(f) = \bigl|\sum_j w_j e^{-2\pi i f t_j}\bigr|^2 / \bigl(\sum_j w_j\bigr)^2
//! \in [0, 1]
//! \]
//!
//! \(W(0) = 1\). Peaks of \(W\) are sampling lines (pass cadence, 1 day, …).

use crate::entities::assessment::{Periodogram, ScoreKind};
use crate::functions::periodicity::gls::floating_mean_gls_power;
use rayon::prelude::*;

/// Sampling window \(W(f)\) at the given frequencies (Hz).
///
/// `weights` must be the same length as `t_s`. When `SigmaSpec` is
/// `Unknown`, pass all ones (unweighted).
pub fn spectral_window(t_s: &[f64], weights: &[f64], freqs_hz: &[f64]) -> Vec<f64> {
    debug_assert_eq!(t_s.len(), weights.len());
    let wsum: f64 = weights.iter().sum();
    if t_s.is_empty() || !wsum.is_finite() || wsum == 0.0 {
        return vec![0.0; freqs_hz.len()];
    }
    let inv_wsum2 = 1.0 / (wsum * wsum);
    freqs_hz
        .par_iter()
        .map(|&f| {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..t_s.len() {
                let ang = -std::f64::consts::TAU * f * t_s[i];
                let (s, c) = ang.sin_cos();
                re += weights[i] * c;
                im += weights[i] * s;
            }
            (re * re + im * im) * inv_wsum2
        })
        .collect()
}

/// \(W\) evaluated on a period grid, stored as a [`Periodogram`].
pub fn window_periodogram(t_s: &[f64], weights: &[f64], periods_s: &[f64]) -> Periodogram {
    let freqs: Vec<f64> = periods_s
        .iter()
        .map(|&p| {
            if p > 0.0 && p.is_finite() {
                1.0 / p
            } else {
                0.0
            }
        })
        .collect();
    let score = spectral_window(t_s, weights, &freqs);
    Periodogram {
        period_s: periods_s.to_vec(),
        score,
        score_kind: ScoreKind::SpectralWindow,
    }
}

/// Local maxima of `score` with `score > min_power`. Endpoints count if
/// they exceed their single neighbour.
pub fn local_maxima(period_s: &[f64], score: &[f64], min_power: f64) -> Vec<(f64, f64)> {
    let n = period_s.len().min(score.len());
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return if score[0] > min_power {
            vec![(period_s[0], score[0])]
        } else {
            Vec::new()
        };
    }
    let mut peaks = Vec::new();
    if score[0] > score[1] && score[0] > min_power {
        peaks.push((period_s[0], score[0]));
    }
    for i in 1..n - 1 {
        if score[i] > score[i - 1] && score[i] > score[i + 1] && score[i] > min_power {
            peaks.push((period_s[i], score[i]));
        }
    }
    if score[n - 1] > score[n - 2] && score[n - 1] > min_power {
        peaks.push((period_s[n - 1], score[n - 1]));
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_at_zero_is_one() {
        let t = vec![0.0, 1.1, 2.4, 5.0];
        let w = vec![1.0; t.len()];
        let ww = spectral_window(&t, &w, &[0.0]);
        assert!((ww[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t_w1_two_pass_comb_peaks_at_pass_cadence_and_is_not_gls_of_ones() {
        // Two 50-sample bursts, 1 s cadence, second burst starts at 1000 s.
        let mut t = Vec::new();
        for i in 0..50 {
            t.push(i as f64);
        }
        for i in 0..50 {
            t.push(1000.0 + i as f64);
        }
        let weights = vec![1.0; t.len()];
        let y_ones = vec![1.0; t.len()];

        // Period grid covering the 1000 s start-to-start cadence.
        let n = 400;
        let p_min: f64 = 100.0;
        let p_max: f64 = 4000.0;
        let periods: Vec<f64> = (0..n)
            .map(|i| {
                let u = i as f64 / (n - 1) as f64;
                (p_min.ln() + u * (p_max.ln() - p_min.ln())).exp()
            })
            .collect();
        let pgram = window_periodogram(&t, &weights, &periods);
        assert_eq!(pgram.score_kind, ScoreKind::SpectralWindow);

        let (best_p, best_w) = pgram
            .period_s
            .iter()
            .zip(pgram.score.iter())
            .filter(|(p, _)| (*p - 1000.0).abs() / 1000.0 < 0.5)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(p, s)| (*p, *s))
            .unwrap();
        assert!(
            (best_p - 1000.0).abs() / 1000.0 < 0.08,
            "W should peak near the 1000 s pass cadence, got {best_p}"
        );
        assert!(
            best_w > 0.3,
            "W at pass cadence should be large, got {best_w}"
        );

        let gls = floating_mean_gls_power(&t, &y_ones, &weights, 1.0 / 1000.0);
        assert!(
            gls.abs() < 1e-12,
            "floating-mean GLS of y≡1 must be ~0, got {gls}"
        );
        assert!(
            (best_w - gls).abs() > 0.2,
            "W(f) must not be GLS-of-ones (W={best_w}, GLS={gls})"
        );

        // GLS of ones is identically ~0 across the grid; W is not.
        let mut max_gls = 0.0_f64;
        for &p in &periods {
            max_gls = max_gls.max(floating_mean_gls_power(&t, &y_ones, &weights, 1.0 / p).abs());
        }
        assert!(
            max_gls < 1e-9,
            "GLS(y=1) must be ~0 everywhere, max={max_gls}"
        );
        let max_w = pgram.score.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_w > 0.5);
    }
}
