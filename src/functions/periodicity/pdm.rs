//! Stellingwerf (1978) phase dispersion minimization.
//!
//! Lower θ is better. Trials whose occupied-bin fraction is below 0.4 are
//! **rejected** (no θ), not softly penalized.

use crate::entities::assessment::{Periodogram, ScoreKind};
use crate::functions::periodicity::fold_phase;
use rayon::prelude::*;

const OCCUPANCY_FLOOR: f64 = 0.4;

/// Adaptive bin count: `min(10, max(4, n/3))`.
pub fn pdm_bin_count(n_in_scope: usize, override_m: Option<usize>) -> usize {
    if let Some(m) = override_m {
        return m.max(2);
    }
    let adapt = (n_in_scope / 3).max(4);
    adapt.min(10)
}

/// Classic PDM θ, or `None` if occupancy is below the floor.
pub fn pdm_theta(t: &[f64], y: &[f64], period: f64, m: usize) -> Option<f64> {
    if m < 2 || t.len() < 4 || !(period > 0.0) {
        return None;
    }
    let mut bins: Vec<Vec<f64>> = (0..m).map(|_| Vec::new()).collect();
    for i in 0..t.len() {
        let phi = fold_phase(t[i], period);
        if !phi.is_finite() {
            continue;
        }
        let idx = ((phi * m as f64) as usize).min(m - 1);
        bins[idx].push(y[i]);
    }
    let mut n_occ = 0usize;
    let mut n_used = 0usize;
    let mut ss_within = 0.0;
    let mut used: Vec<f64> = Vec::new();
    for b in &bins {
        if b.len() < 2 {
            continue;
        }
        n_occ += 1;
        n_used += b.len();
        let mean = b.iter().sum::<f64>() / b.len() as f64;
        let rss: f64 = b.iter().map(|v| (v - mean) * (v - mean)).sum();
        // (n_j − 1) s_j² = RSS
        ss_within += rss;
        used.extend_from_slice(b);
    }
    if (n_occ as f64) < OCCUPANCY_FLOOR * m as f64 || n_used <= n_occ {
        return None;
    }
    let mean_all = used.iter().sum::<f64>() / used.len() as f64;
    let ss_tot: f64 = used.iter().map(|v| (v - mean_all) * (v - mean_all)).sum();
    if ss_tot <= 0.0 {
        return Some(0.0);
    }
    // θ = Σ (n_j−1) s_j² / ((N_used − n_occ) s_tot²)
    // s_tot² = ss_tot / (N_used − 1)
    let s_tot2 = ss_tot / (n_used - 1) as f64;
    Some(ss_within / ((n_used - n_occ) as f64 * s_tot2))
}

pub fn pdm_periodogram(t: &[f64], y: &[f64], periods: &[f64], m: usize) -> Periodogram {
    let score: Vec<f64> = periods
        .par_iter()
        .map(|&p| pdm_theta(t, y, p, m).unwrap_or(f64::INFINITY))
        .collect();
    Periodogram {
        period_s: periods.to_vec(),
        score,
        score_kind: ScoreKind::PdmTheta,
    }
}

/// Index of the lowest finite θ. `None` if every trial was occupancy-invalid.
pub fn argmin_finite(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_reject_on_clump() {
        // All points in a tiny time window → most phase bins empty at long P.
        let t = vec![0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
        let y = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        assert!(pdm_theta(&t, &y, 1000.0, 10).is_none());
    }

    #[test]
    fn sine_prefers_true_period() {
        let p_true = 10.0;
        let t: Vec<f64> = (0..80).map(|i| i as f64 * 0.4).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|ti| (std::f64::consts::TAU * ti / p_true).sin())
            .collect();
        let m = pdm_bin_count(t.len(), None);
        let th_true = pdm_theta(&t, &y, p_true, m).unwrap();
        let th_wrong = pdm_theta(&t, &y, 6.3, m).unwrap();
        assert!(th_true < th_wrong, "{th_true} vs {th_wrong}");
    }
}
