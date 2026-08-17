//! Generalized Lomb–Scargle (Zechmeister & Kürster 2009).
//!
//! After Auto detrend the intercept is already in `n_β`, so the default
//! path is **zero-mean** GLS. Floating-mean H=1 is kept for T-BAL1 and
//! undetrended research use.

use crate::entities::assessment::{Periodogram, ScoreKind};
use crate::functions::sampling::log_period_grid;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Zero-mean H-harmonic GLS power \(p_H = (\chi^2_0 - \chi^2_H)/\chi^2_0\).
pub fn gls_power_zero_mean(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    period: f64,
    n_harmonics: usize,
) -> f64 {
    let n = t.len();
    if n < 3 || period <= 0.0 || n_harmonics == 0 {
        return 0.0;
    }
    let chi2_0: f64 = (0..n).map(|i| w[i] * y[i] * y[i]).sum();
    if chi2_0 <= 0.0 {
        return 0.0;
    }
    let k = 2 * n_harmonics;
    let mut xtwx = DMatrix::<f64>::zeros(k, k);
    let mut xty = DVector::<f64>::zeros(k);
    let omega = std::f64::consts::TAU / period;
    for i in 0..n {
        let mut cols = vec![0.0; k];
        for h in 0..n_harmonics {
            let ang = omega * (h + 1) as f64 * t[i];
            let (s, c) = ang.sin_cos();
            cols[2 * h] = c;
            cols[2 * h + 1] = s;
        }
        let wi = w[i];
        for a in 0..k {
            xty[a] += wi * cols[a] * y[i];
            for b in 0..=a {
                let v = wi * cols[a] * cols[b];
                xtwx[(a, b)] += v;
                if a != b {
                    xtwx[(b, a)] += v;
                }
            }
        }
    }
    let beta = match xtwx.clone().lu().solve(&xty) {
        Some(b) => b,
        None => return 0.0,
    };
    let explained = beta.dot(&xty);
    let chi2_h = (chi2_0 - explained).max(0.0);
    ((chi2_0 - chi2_h) / chi2_0).clamp(0.0, 1.0)
}

/// Zechmeister & Kürster 2009 eq. 20 — floating-mean single-frequency GLS.
pub fn floating_mean_gls_power(t: &[f64], y: &[f64], w: &[f64], freq_hz: f64) -> f64 {
    let n = t.len();
    if n < 3 || y.len() != n || w.len() != n {
        return 0.0;
    }
    let wsum: f64 = w.iter().sum();
    if wsum <= 0.0 {
        return 0.0;
    }
    let ybar: f64 = (0..n).map(|i| w[i] * y[i]).sum::<f64>() / wsum;
    let omega = std::f64::consts::TAU * freq_hz;
    let mut c_sum = 0.0;
    let mut s_sum = 0.0;
    let mut yy = 0.0;
    let mut yc = 0.0;
    let mut ys = 0.0;
    let mut cc = 0.0;
    let mut ss = 0.0;
    let mut cs = 0.0;
    for i in 0..n {
        let wi = w[i];
        let yh = y[i] - ybar;
        let (s, c) = (omega * t[i]).sin_cos();
        c_sum += wi * c;
        s_sum += wi * s;
        yy += wi * yh * yh;
        yc += wi * yh * c;
        ys += wi * yh * s;
        cc += wi * c * c;
        ss += wi * s * s;
        cs += wi * c * s;
    }
    cc -= c_sum * c_sum / wsum;
    ss -= s_sum * s_sum / wsum;
    cs -= c_sum * s_sum / wsum;
    let det = cc * ss - cs * cs;
    if yy <= 0.0 || det <= 0.0 {
        return 0.0;
    }
    ((yc * yc * ss + ys * ys * cc - 2.0 * yc * ys * cs) / (yy * det)).clamp(0.0, 1.0)
}

/// Subtract the zero-mean H=1 sinusoid at `period` from `y`.
pub fn subtract_h1(t: &[f64], y: &[f64], w: &[f64], period: f64) -> Vec<f64> {
    let n = t.len();
    if n == 0 || period <= 0.0 {
        return y.to_vec();
    }
    let omega = std::f64::consts::TAU / period;
    let mut cc = 0.0;
    let mut ss = 0.0;
    let mut cs = 0.0;
    let mut yc = 0.0;
    let mut ys = 0.0;
    let mut cosv = vec![0.0; n];
    let mut sinv = vec![0.0; n];
    for i in 0..n {
        let (s, c) = (omega * t[i]).sin_cos();
        cosv[i] = c;
        sinv[i] = s;
        cc += w[i] * c * c;
        ss += w[i] * s * s;
        cs += w[i] * c * s;
        yc += w[i] * y[i] * c;
        ys += w[i] * y[i] * s;
    }
    let det = cc * ss - cs * cs;
    if det.abs() < 1e-18 {
        return y.to_vec();
    }
    let a = (yc * ss - ys * cs) / det;
    let b = (ys * cc - yc * cs) / det;
    (0..n).map(|i| y[i] - a * cosv[i] - b * sinv[i]).collect()
}

pub fn gls_periodogram(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    periods: &[f64],
    n_harmonics: usize,
    zero_mean: bool,
) -> Periodogram {
    let score: Vec<f64> = periods
        .par_iter()
        .map(|&p| {
            if zero_mean {
                gls_power_zero_mean(t, y, w, p, n_harmonics)
            } else if n_harmonics <= 1 {
                floating_mean_gls_power(t, y, w, 1.0 / p)
            } else {
                gls_power_zero_mean(t, y, w, p, n_harmonics)
            }
        })
        .collect();
    Periodogram {
        period_s: periods.to_vec(),
        score,
        score_kind: ScoreKind::GlsPower,
    }
}

/// Coarse log grid + ±5% refine around the top-`n_peaks` local maxima.
/// Does **not** shrink `f_max`.
pub fn coarse_refine_periods(
    p_min: f64,
    p_max: f64,
    span_s: f64,
    n_coarse: usize,
    max_trials: usize,
    oversample: f64,
    n_peaks: usize,
    evaluate: impl Fn(&[f64]) -> Vec<f64> + Sync,
) -> (Vec<f64>, Vec<f64>) {
    let n_coarse = n_coarse.max(8).min(max_trials);
    let coarse = log_period_grid(p_min, p_max, n_coarse);
    if coarse.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let coarse_s = evaluate(&coarse);
    let peaks = top_k_peaks(&coarse, &coarse_s, n_peaks);
    let fine_budget = max_trials.saturating_sub(coarse.len());
    let per_peak = if peaks.is_empty() {
        0
    } else {
        (fine_budget / peaks.len()).max(8)
    };
    let df = if span_s > 0.0 && oversample >= 1.0 {
        1.0 / (oversample * span_s)
    } else {
        0.0
    };
    let mut fine: Vec<f64> = Vec::new();
    for &peak in &peaks {
        fine.extend(refine_around(peak, p_min, p_max, df, per_peak));
    }
    let fine_s = if fine.is_empty() {
        Vec::new()
    } else {
        evaluate(&fine)
    };

    let mut pairs: Vec<(f64, f64)> = coarse
        .into_iter()
        .zip(coarse_s)
        .chain(fine.into_iter().zip(fine_s))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs.dedup_by(|a, b| (a.0 - b.0).abs() <= f64::EPSILON * a.0.max(1.0));
    let (p, s): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    (p, s)
}

fn top_k_peaks(periods: &[f64], scores: &[f64], k: usize) -> Vec<f64> {
    let n = periods.len().min(scores.len());
    if n == 0 {
        return Vec::new();
    }
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    if n == 1 {
        return vec![periods[0]];
    }
    if scores[0] > scores[1] {
        peaks.push((periods[0], scores[0]));
    }
    for i in 1..n - 1 {
        if scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1] {
            peaks.push((periods[i], scores[i]));
        }
    }
    if scores[n - 1] > scores[n - 2] {
        peaks.push((periods[n - 1], scores[n - 1]));
    }
    if peaks.is_empty() {
        peaks = periods
            .iter()
            .copied()
            .zip(scores.iter().copied())
            .collect();
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    peaks.truncate(k);
    peaks.into_iter().map(|(p, _)| p).collect()
}

fn refine_around(peak: f64, p_min: f64, p_max: f64, df: f64, n_target: usize) -> Vec<f64> {
    if n_target == 0 || peak <= 0.0 {
        return Vec::new();
    }
    let lo = (peak * 0.95).max(p_min);
    let hi = (peak * 1.05).min(p_max);
    if hi <= lo {
        return Vec::new();
    }
    if df > 0.0 {
        let f_lo = 1.0 / hi;
        let f_hi = 1.0 / lo;
        let mut freqs = Vec::new();
        let mut f = f_lo;
        while f <= f_hi && freqs.len() < n_target {
            freqs.push(f);
            f += df;
        }
        if freqs.len() >= 2 {
            return freqs.into_iter().rev().map(|f| 1.0 / f).collect();
        }
    }
    let step = (hi.ln() - lo.ln()) / (n_target.max(2) - 1) as f64;
    (0..n_target.max(2))
        .map(|i| (lo.ln() + i as f64 * step).exp())
        .collect()
}

/// Interior maximum of a higher-better score: not an endpoint, both neighbours lower.
pub fn is_interior_maximum(scores: &[f64], idx: usize) -> bool {
    idx > 0
        && idx + 1 < scores.len()
        && scores[idx] > scores[idx - 1]
        && scores[idx] > scores[idx + 1]
}

pub fn argmax(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

/// Parabolic interpolation of `score(log P)` around `idx`. Returns (P, half-width).
pub fn interpolate_peak(periods: &[f64], scores: &[f64], idx: usize) -> (f64, Option<f64>) {
    if idx == 0 || idx + 1 >= periods.len() {
        return (periods.get(idx).copied().unwrap_or(f64::NAN), None);
    }
    let x0 = periods[idx - 1].ln();
    let x1 = periods[idx].ln();
    let x2 = periods[idx + 1].ln();
    let y0 = scores[idx - 1];
    let y1 = scores[idx];
    let y2 = scores[idx + 1];
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denom.abs() < 1e-18 {
        return (periods[idx], None);
    }
    // Fit y = a x^2 + b x + c through three points; vertex at -b/(2a).
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a >= 0.0 {
        return (periods[idx], None);
    }
    let xv = -b / (2.0 * a);
    if !xv.is_finite() || xv < x0 || xv > x2 {
        return (periods[idx], None);
    }
    let p = xv.exp();
    // Half-width where the parabola drops by y1 / N, N ~ n_points proxy.
    let drop = (y1 / periods.len().max(8) as f64).max(1e-6);
    let c = y1 - a * x1 * x1 - b * x1;
    let disc = b * b - 4.0 * a * (c - (y1 - drop));
    let width = if disc > 0.0 {
        let dx = disc.sqrt() / (2.0 * a.abs());
        Some(0.5 * ((xv + dx).exp() - (xv - dx).exp()).abs())
    } else {
        None
    };
    (p, width)
}
