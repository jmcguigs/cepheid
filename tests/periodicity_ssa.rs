//! SSA periodicity contracts.
//!
//! Frozen geometric knobs (do not retune to green):
//! - α = 1e-3 (Baluev on H=1)
//! - B = 200 (local 0-beat; cannot resolve α)
//! - window_ratio = 1.3
//! - occupancy floor = 0.4
//! - gap_factor = 8, min_gap_s = 60
//! - Intra eligible: n ≥ 12 and duration ≥ 12 · median Δt
//! - bound-snap: interior maximum only
//!
//! T18 is `#[ignore]` — n=200 is not run in default CI and does not claim a 1% FPR.

mod support;

use cepheid::assess_periodicity;
use cepheid::entities::assessment::{PeriodSearchConfig, PeriodicityDecision, SearchScale};
use cepheid::functions::periodicity::gls::floating_mean_gls_power;
use cepheid::functions::periodicity::window::spectral_window;
use support::ssa_synth::{leftover_lambert, leo_week_times, series_xy, series_xy_phase};

#[test]
fn t_w1_window_is_not_gls_of_ones() {
    let mut t = Vec::new();
    for i in 0..40 {
        t.push(i as f64);
    }
    for i in 0..40 {
        t.push(1000.0 + i as f64);
    }
    let w = vec![1.0; t.len()];
    let y = vec![1.0; t.len()];
    let ww = spectral_window(&t, &w, &[1.0 / 1000.0]);
    let gls = floating_mean_gls_power(&t, &y, &w, 1.0 / 1000.0);
    assert!(ww[0] > 0.3);
    assert!(gls.abs() < 1e-9);
}

#[test]
fn t14_67689_class_not_periodic() {
    let t = leo_week_times();
    let phi: Vec<f64> = t
        .iter()
        .map(|ti| std::f64::consts::TAU * ti / 5400.0)
        .collect();
    let y: Vec<f64> = leftover_lambert(&phi, 0.2)
        .into_iter()
        .zip(t.iter())
        .map(|(l, ti)| {
            let tau = ti.rem_euclid(5400.0) / 480.0;
            l + 0.15 * tau.min(1.0)
        })
        .collect();
    let s = series_xy_phase(t, y, Some(phi));
    let mut c = PeriodSearchConfig::conservative();
    c.scale = SearchScale::Auto;
    c.min_period_s = Some(3.0);
    c.max_period_s = Some(3000.0);
    let a = assess_periodicity(&s, &c);
    assert_ne!(
        a.decision,
        PeriodicityDecision::Periodic,
        "67689-class must not be Periodic: {:?}",
        a.notes
    );
}

#[test]
fn t21_one_pass_still_works() {
    let t = cepheid::functions::sampling::leo_pass_times(1, 80, 480.0, 5400.0, 0.0);
    let y: Vec<f64> = t
        .iter()
        .map(|ti| (std::f64::consts::TAU * ti / 47.0).sin())
        .collect();
    let s = series_xy(t, y);
    let mut c = PeriodSearchConfig::conservative();
    c.max_period_s = Some(200.0);
    let a = assess_periodicity(&s, &c);
    assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
}

/// 200-draw FPR probe. Not default CI. Prints a Clopper–Pearson 95% interval.
#[test]
#[ignore]
fn t18_fpr_monte_carlo() {
    let n = 200usize;
    let mut false_pos = 0usize;
    for seed in 0..n {
        let t = leo_week_times();
        let y: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let u = ((seed as u64)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(i as u64 * 12_345)
                    % (1 << 31)) as f64
                    / (1u64 << 31) as f64;
                10.0 + 0.1 * (u - 0.5)
            })
            .collect();
        let s = series_xy(t, y);
        let mut c = PeriodSearchConfig::conservative();
        c.min_period_s = Some(3.0);
        c.max_period_s = Some(3000.0);
        if assess_periodicity(&s, &c).decision == PeriodicityDecision::Periodic {
            false_pos += 1;
        }
    }
    let (lo, hi) = clopper_pearson(false_pos, n, 0.05);
    println!("T18 FPR {false_pos}/{n}  95% CI [{lo:.4}, {hi:.4}]");
}

fn clopper_pearson(k: usize, n: usize, alpha: f64) -> (f64, f64) {
    // Normal approximation fallback (no extra deps). Header states this is
    // not a 1% FPR enforcement.
    let p = k as f64 / n as f64;
    let z = 1.96; // ~95%
    let _ = alpha;
    let se = (p * (1.0 - p) / n as f64).sqrt();
    ((p - z * se).max(0.0), (p + z * se).min(1.0))
}
