//! Pass clustering, searchable bounds, and sampling diagnostics.
//!
//! The spectral window itself lives in [`crate::functions::periodicity::window`].
//! This module builds the frequency grid, clusters passes, and tags window peaks.

use crate::constants::{SIDEREAL_DAY_S, SOLAR_DAY_S};
use crate::entities::assessment::{AliasKind, Pass, SamplingDiagnostics, SearchScale, WindowPeak};
use crate::entities::series::Series;
use crate::functions::periodicity::window::{local_maxima, spectral_window, window_periodogram};

/// Default oversample for the window grid (inherited later from `PeriodSearchConfig`).
pub const DEFAULT_OVERSAMPLE: f64 = 5.0;

/// Coarse log-spaced trial count for \(W(f)\) until GLS shares this grid.
const DEFAULT_N_COARSE: usize = 2000;

/// Minimum number of cycles across the span that a searchable period must cover.
const N_CYCLES_MIN: f64 = 2.0;

/// Relative match window for tagging a \(W\) peak as a named alias.
const ALIAS_REL_TOL: f64 = 0.03;

/// Cap on `P_day / k` lines. `k = 29` is the T6b GEO harmonic (~2971 s);
/// going to `P_min = 3 s` without a cap would emit ~28k named lines.
const MAX_NAMED_HARMONIC: i32 = 48;

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Inter-pass / intra-pass Δt ratio. Default 8.
    pub gap_factor: f64,
    /// Absolute floor for a new visibility window. Default 60 s.
    pub min_gap_s: f64,
    /// How many unnamed \(W\) local maxima to keep. Default 8.
    pub n_window_peaks: usize,
    /// Caller-requested minimum period (seconds).
    pub min_period_s: f64,
    /// Caller-requested maximum period (seconds).
    pub max_period_s: f64,
    /// Known physical periods (e.g. \(P_\text{orb}\)) used to tag `Orbital` / `OrbitalHalf`.
    pub known_periods_s: Vec<f64>,
    /// Scale used to tighten searchable bounds. `Auto` uses campaign (Full) bounds.
    pub scale: SearchScale,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            gap_factor: 8.0,
            min_gap_s: 60.0,
            n_window_peaks: 8,
            min_period_s: 3.0,
            max_period_s: f64::INFINITY,
            known_periods_s: Vec::new(),
            scale: SearchScale::Auto,
        }
    }
}

impl SamplingConfig {
    pub fn new(min_period_s: f64, max_period_s: f64) -> Self {
        Self {
            min_period_s,
            max_period_s,
            ..Self::default()
        }
    }
}

/// Cluster, bound, and evaluate the DFT sampling window.
///
/// `oversample` is the search oversample (default [`DEFAULT_OVERSAMPLE`]).
/// The window is evaluated on a log-spaced coarse grid over the searchable
/// period range — the same coarse grid later PRs reuse for GLS.
pub fn diagnose_sampling(
    series: &Series,
    cfg: &SamplingConfig,
    oversample: f64,
) -> SamplingDiagnostics {
    let t = series.t_s();
    let n = t.len();
    let span_s = series.span_s().unwrap_or(0.0);
    let dts: Vec<f64> = if n >= 2 {
        t.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        Vec::new()
    };
    let min_dt_s = dts.iter().copied().fold(f64::INFINITY, f64::min);
    let min_dt_s = if min_dt_s.is_finite() { min_dt_s } else { 0.0 };
    let median_dt_s = percentile(&dts, 0.5);
    let p95_dt_s = percentile(&dts, 0.95);

    let passes = cluster_passes(t, cfg.gap_factor, cfg.min_gap_s);
    let n_passes = passes.len();
    let intrapass_dts = intrapass_deltas(t, &passes);
    let median_intrapass_dt_s = if intrapass_dts.is_empty() {
        median_dt_s
    } else {
        percentile(&intrapass_dts, 0.5)
    };

    let pass_durs: Vec<f64> = passes.iter().map(Pass::duration_s).collect();
    let median_pass_duration_s = percentile(&pass_durs, 0.5);
    let interpass_gaps = interpass_gaps(&passes);
    let median_interpass_gap_s = percentile(&interpass_gaps, 0.5);

    let duty_cycle = if span_s > 0.0 {
        pass_durs.iter().sum::<f64>() / span_s
    } else {
        0.0
    };

    let (min_searchable_period_s, max_searchable_period_s) = searchable_period_bounds(
        cfg,
        span_s,
        median_intrapass_dt_s,
        median_pass_duration_s,
        median_interpass_gap_s,
        n_passes,
    );

    let oversample = if oversample.is_finite() && oversample >= 1.0 {
        oversample
    } else {
        DEFAULT_OVERSAMPLE
    };

    let periods = if n < 2
        || min_searchable_period_s <= 0.0
        || max_searchable_period_s <= min_searchable_period_s
    {
        Vec::new()
    } else {
        log_period_grid(
            min_searchable_period_s,
            max_searchable_period_s,
            DEFAULT_N_COARSE,
        )
    };

    let weights = series.weights();
    let spectral_window = window_periodogram(t, &weights, &periods);
    let window_peaks = tag_window_peaks(
        &spectral_window.period_s,
        &spectral_window.score,
        cfg,
        min_searchable_period_s,
        max_searchable_period_s,
        median_interpass_gap_s,
        median_pass_duration_s,
        t,
        &weights,
        oversample,
        span_s,
    );

    SamplingDiagnostics {
        n,
        n_merged_duplicates: series.n_merged_duplicates(),
        span_s,
        min_dt_s,
        median_dt_s,
        median_intrapass_dt_s,
        p95_dt_s,
        duty_cycle,
        n_passes,
        passes,
        min_searchable_period_s,
        max_searchable_period_s,
        spectral_window,
        window_peaks,
    }
}

/// Practical irregular-sampling search limits (Eyer & Bartholdi / VanderPlas).
pub fn searchable_period_bounds(
    cfg: &SamplingConfig,
    span_s: f64,
    median_intrapass_dt_s: f64,
    median_pass_duration_s: f64,
    median_interpass_gap_s: f64,
    n_passes: usize,
) -> (f64, f64) {
    let mut p_min = cfg.min_period_s.max(2.0 * median_intrapass_dt_s);
    let mut p_max = cfg.max_period_s.min(span_s / N_CYCLES_MIN);
    match cfg.scale {
        SearchScale::IntraPass => {
            if median_pass_duration_s > 0.0 {
                p_max = p_max.min(0.5 * median_pass_duration_s);
            }
        }
        SearchScale::InterPass => {
            p_min = p_min.max(median_pass_duration_s);
            if n_passes >= 2 && median_interpass_gap_s > 0.0 {
                p_max =
                    p_max.min(0.5 * median_interpass_gap_s * (n_passes.saturating_sub(1) as f64));
            }
        }
        SearchScale::Auto | SearchScale::Full => {}
    }
    if !p_min.is_finite() || !p_max.is_finite() || p_min <= 0.0 || p_max <= p_min {
        return (0.0, 0.0);
    }
    (p_min, p_max)
}

pub fn cluster_passes(t: &[f64], gap_factor: f64, min_gap_s: f64) -> Vec<Pass> {
    if t.is_empty() {
        return Vec::new();
    }
    if t.len() == 1 {
        return vec![Pass {
            t_start_s: t[0],
            t_end_s: t[0],
            n: 1,
            median_dt_s: 0.0,
        }];
    }
    let dts: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    let med_dt = percentile(&dts, 0.5);
    let gap_thresh = min_gap_s.max(gap_factor * med_dt.max(0.0));

    let mut passes = Vec::new();
    let mut start = 0usize;
    for i in 0..dts.len() {
        if dts[i] > gap_thresh {
            passes.push(make_pass(&t[start..=i]));
            start = i + 1;
        }
    }
    passes.push(make_pass(&t[start..]));
    passes
}

pub fn log_period_grid(min_period: f64, max_period: f64, n: usize) -> Vec<f64> {
    if n < 2 || min_period <= 0.0 || max_period <= min_period {
        return Vec::new();
    }
    let log_min = min_period.ln();
    let log_max = max_period.ln();
    let step = (log_max - log_min) / (n - 1) as f64;
    (0..n).map(|i| (log_min + i as f64 * step).exp()).collect()
}

fn make_pass(t: &[f64]) -> Pass {
    let dts: Vec<f64> = if t.len() >= 2 {
        t.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        Vec::new()
    };
    Pass {
        t_start_s: t[0],
        t_end_s: t[t.len() - 1],
        n: t.len(),
        median_dt_s: percentile(&dts, 0.5),
    }
}

fn intrapass_deltas(t: &[f64], passes: &[Pass]) -> Vec<f64> {
    let mut out = Vec::new();
    for p in passes {
        let slice: Vec<f64> = t
            .iter()
            .copied()
            .filter(|&ti| ti >= p.t_start_s && ti <= p.t_end_s)
            .collect();
        if slice.len() >= 2 {
            out.extend(slice.windows(2).map(|w| w[1] - w[0]));
        }
    }
    out
}

fn interpass_gaps(passes: &[Pass]) -> Vec<f64> {
    if passes.len() < 2 {
        return Vec::new();
    }
    passes
        .windows(2)
        .map(|w| (w[1].t_start_s - w[0].t_end_s).max(0.0))
        .collect()
}

fn tag_window_peaks(
    periods: &[f64],
    scores: &[f64],
    cfg: &SamplingConfig,
    p_min: f64,
    p_max: f64,
    median_interpass_gap_s: f64,
    median_pass_duration_s: f64,
    t: &[f64],
    weights: &[f64],
    oversample: f64,
    span_s: f64,
) -> Vec<WindowPeak> {
    let mut named: Vec<(f64, AliasKind)> = Vec::new();
    push_harmonics(
        &mut named,
        SIDEREAL_DAY_S,
        AliasKind::SiderealDay,
        p_min,
        p_max,
    );
    push_harmonics(&mut named, SOLAR_DAY_S, AliasKind::SolarDay, p_min, p_max);
    for &p_orb in &cfg.known_periods_s {
        if in_range(p_orb, p_min, p_max) {
            named.push((p_orb, AliasKind::Orbital));
        }
        let half = p_orb / 2.0;
        if in_range(half, p_min, p_max) {
            named.push((half, AliasKind::OrbitalHalf));
        }
    }
    if in_range(median_interpass_gap_s, p_min, p_max) {
        named.push((median_interpass_gap_s, AliasKind::PassCadence));
    }
    if in_range(median_pass_duration_s, p_min, p_max) {
        named.push((median_pass_duration_s, AliasKind::PassDuration));
    }

    let df = if span_s > 0.0 {
        1.0 / (oversample * span_s)
    } else {
        0.0
    };

    let mut peaks: Vec<WindowPeak> = Vec::new();

    // Always emit named lines, with W evaluated exactly at that period.
    for &(p, kind) in &named {
        let w = if p > 0.0 {
            spectral_window(t, weights, &[1.0 / p])
                .first()
                .copied()
                .unwrap_or(0.0)
        } else {
            0.0
        };
        peaks.push(WindowPeak {
            period_s: p,
            power: w,
            kind,
        });
    }

    let mut maxima = local_maxima(periods, scores, 0.1);
    maxima.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    maxima.truncate(cfg.n_window_peaks);

    for (p, power) in maxima {
        if let Some(idx) = match_named(p, &named, df) {
            // Already emitted as a named line; keep the stronger power if the
            // grid peak is closer to the true maximum.
            if let Some(existing) = peaks.iter_mut().find(|pk| pk.kind == named[idx].1) {
                if power > existing.power {
                    existing.period_s = p;
                    existing.power = power;
                }
            }
            continue;
        }
        peaks.push(WindowPeak {
            period_s: p,
            power,
            kind: AliasKind::WindowPeak,
        });
    }

    peaks.sort_by(|a, b| {
        b.power
            .partial_cmp(&a.power)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    peaks
}

fn push_harmonics(
    out: &mut Vec<(f64, AliasKind)>,
    fundamental: f64,
    kind: AliasKind,
    p_min: f64,
    p_max: f64,
) {
    if fundamental <= 0.0 {
        return;
    }
    let k_max = (fundamental / p_min).floor() as i32;
    let k_max = k_max.clamp(1, MAX_NAMED_HARMONIC);
    for k in 1..=k_max {
        let p = fundamental / k as f64;
        if in_range(p, p_min, p_max) {
            out.push((p, kind));
        }
    }
}

fn in_range(p: f64, p_min: f64, p_max: f64) -> bool {
    p.is_finite() && p > 0.0 && p >= p_min && p <= p_max
}

fn match_named(period: f64, named: &[(f64, AliasKind)], df: f64) -> Option<usize> {
    let f = 1.0 / period;
    named.iter().enumerate().find_map(|(i, &(p, _))| {
        if p <= 0.0 {
            return None;
        }
        let rel = (period - p).abs() / p;
        let df_ok = df > 0.0 && (f - 1.0 / p).abs() < 1.5 * df;
        if rel < ALIAS_REL_TOL || df_ok {
            Some(i)
        } else {
            None
        }
    })
}

fn percentile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    v[idx.min(v.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::series::{Covariates, Modality, SeriesMeta, SigmaSpec, YUnit};

    fn series_from_t_y(t: Vec<f64>, y: Vec<f64>) -> Series {
        Series::try_new(
            t,
            y,
            SigmaSpec::Unknown,
            Covariates::default(),
            SeriesMeta {
                modality: Modality::Generic,
                y_unit: YUnit::Dimensionless,
                label: None,
            },
        )
        .unwrap()
    }

    /// LEO-week times: 18 passes, 40 pts, 480 s pass, 5400 s orbit.
    fn leo_week_times() -> Vec<f64> {
        leo_pass_times(18, 40, 480.0, 5400.0, 0.0)
    }

    fn leo_pass_times(
        n_passes: usize,
        pts_per_pass: usize,
        pass_len_s: f64,
        orbit_s: f64,
        t0: f64,
    ) -> Vec<f64> {
        let mut t = Vec::with_capacity(n_passes * pts_per_pass);
        let dt = if pts_per_pass > 1 {
            pass_len_s / (pts_per_pass - 1) as f64
        } else {
            0.0
        };
        for p in 0..n_passes {
            let start = t0 + p as f64 * orbit_s;
            for i in 0..pts_per_pass {
                t.push(start + i as f64 * dt);
            }
        }
        t
    }

    fn geo_night_times(
        n_nights: usize,
        pts_per_night: usize,
        night_len_s: f64,
        t0: f64,
    ) -> Vec<f64> {
        let mut t = Vec::with_capacity(n_nights * pts_per_night);
        let dt = if pts_per_night > 1 {
            night_len_s / (pts_per_night - 1) as f64
        } else {
            0.0
        };
        for n in 0..n_nights {
            let start = t0 + n as f64 * SOLAR_DAY_S;
            for i in 0..pts_per_night {
                t.push(start + i as f64 * dt);
            }
        }
        t
    }

    fn peak_near(peaks: &[WindowPeak], period: f64, rel: f64) -> Option<&WindowPeak> {
        peaks
            .iter()
            .find(|p| (p.period_s - period).abs() / period < rel)
    }

    fn w_near(diag: &SamplingDiagnostics, period: f64, rel: f64) -> f64 {
        diag.spectral_window
            .period_s
            .iter()
            .zip(diag.spectral_window.score.iter())
            .filter(|(p, _)| (*p - period).abs() / period < rel)
            .map(|(_, s)| *s)
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn t16_empty_and_singleton_do_not_panic() {
        let empty = series_from_t_y(vec![], vec![]);
        let d0 = diagnose_sampling(&empty, &SamplingConfig::default(), DEFAULT_OVERSAMPLE);
        assert_eq!(d0.n, 0);
        assert_eq!(d0.n_passes, 0);
        assert!(d0.spectral_window.is_empty());
        assert!(d0.is_too_sparse());

        let one = series_from_t_y(vec![0.0], vec![1.0]);
        let d1 = diagnose_sampling(&one, &SamplingConfig::default(), DEFAULT_OVERSAMPLE);
        assert_eq!(d1.n, 1);
        assert_eq!(d1.n_passes, 1);
        assert_eq!(d1.span_s, 0.0);
        assert!(d1.spectral_window.is_empty());
        assert!(d1.is_too_sparse());
    }

    #[test]
    fn t1_leo_week_constant_y_window_peaks_at_orbit() {
        let t = leo_week_times();
        let y = vec![1.0; t.len()];
        let s = series_from_t_y(t, y);
        let cfg = SamplingConfig::new(3.0, 20_000.0);
        let d = diagnose_sampling(&s, &cfg, DEFAULT_OVERSAMPLE);
        assert_eq!(d.n_passes, 18);
        assert!(d.duty_cycle > 0.0 && d.duty_cycle < 0.2);
        // Pulse-train period is the orbit (5400 s), independent of y.
        let w_orb = w_near(&d, 5400.0, 0.05);
        assert!(
            w_orb > 0.2,
            "LEO-week W should be large near the 5400 s orbit, got {w_orb}"
        );
        assert!(
            peak_near(&d.window_peaks, 4920.0, 0.1).is_some()
                || d.window_peaks
                    .iter()
                    .any(|p| p.kind == AliasKind::PassCadence),
            "expected a PassCadence tag near the inter-pass gap"
        );
    }

    #[test]
    fn t2_geo_week_constant_y_window_has_daily_line() {
        let t = geo_night_times(7, 80, 18_000.0, 0.0);
        let y = vec![1.0; t.len()];
        let s = series_from_t_y(t, y);
        let cfg = SamplingConfig::new(3.0, 200_000.0);
        let d = diagnose_sampling(&s, &cfg, DEFAULT_OVERSAMPLE);
        assert_eq!(d.n_passes, 7);
        let has_day = d.window_peaks.iter().any(|p| {
            matches!(p.kind, AliasKind::SolarDay | AliasKind::SiderealDay)
                && ((p.period_s - SOLAR_DAY_S).abs() / SOLAR_DAY_S < 0.02
                    || (p.period_s - SIDEREAL_DAY_S).abs() / SIDEREAL_DAY_S < 0.02)
        });
        let w_day = w_near(&d, SOLAR_DAY_S, 0.05).max(w_near(&d, SIDEREAL_DAY_S, 0.05));
        assert!(
            has_day || w_day > 0.15,
            "GEO-week window should show a ~1-day line (tagged or in W)"
        );
    }

    #[test]
    fn t5_zero_y_window_matches_constant_y() {
        let t = leo_week_times();
        let ones = series_from_t_y(t.clone(), vec![1.0; t.len()]);
        let zeros = series_from_t_y(t.clone(), vec![0.0; t.len()]);
        let cfg = SamplingConfig::new(3.0, 20_000.0);
        let d1 = diagnose_sampling(&ones, &cfg, DEFAULT_OVERSAMPLE);
        let d0 = diagnose_sampling(&zeros, &cfg, DEFAULT_OVERSAMPLE);
        assert_eq!(d1.n_passes, d0.n_passes);
        assert_eq!(d1.spectral_window.len(), d0.spectral_window.len());
        for (a, b) in d1
            .spectral_window
            .score
            .iter()
            .zip(d0.spectral_window.score.iter())
        {
            assert!((a - b).abs() < 1e-12, "W must not depend on y");
        }
        let w0 = w_near(&d0, 5400.0, 0.05);
        assert!(w0 > 0.2, "zero-y LEO-week still has a 5400 s window line");
    }

    #[test]
    fn intra_pass_bounds_cap_at_half_pass_duration() {
        let t = leo_pass_times(3, 40, 480.0, 5400.0, 0.0);
        let s = series_from_t_y(t, vec![0.0; 120]);
        let mut cfg = SamplingConfig::new(3.0, 3000.0);
        cfg.scale = SearchScale::IntraPass;
        let d = diagnose_sampling(&s, &cfg, DEFAULT_OVERSAMPLE);
        assert!(d.max_searchable_period_s <= 0.5 * 480.0 + 1e-9);
        assert!(d.passes.iter().all(|p| p.is_intra_eligible()));
    }

    #[test]
    fn known_period_is_tagged_orbital() {
        let t = leo_week_times();
        let s = series_from_t_y(t.clone(), vec![1.0; t.len()]);
        let mut cfg = SamplingConfig::new(3.0, 20_000.0);
        cfg.known_periods_s = vec![5400.0];
        let d = diagnose_sampling(&s, &cfg, DEFAULT_OVERSAMPLE);
        assert!(d.window_peaks.iter().any(|p| p.kind == AliasKind::Orbital));
        assert!(d.window_peaks.iter().any(
            |p| p.kind == AliasKind::OrbitalHalf && (p.period_s - 2700.0).abs() / 2700.0 < 0.02
        ));
    }
}
