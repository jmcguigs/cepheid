use crate::entities::observation::Observation;
use crate::entities::lightcurve::Lightcurve;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

/// Generate a trial-period grid bounded by `max_trials`.
///
/// Uses the adaptive step Δp = p² · frac / span when the resulting count
/// fits in `max_trials`, otherwise falls back to a log-spaced grid of
/// exactly `max_trials` points. Photometric campaigns can span days while
/// resolving periods in seconds — the adaptive formula collapses the step
/// size in that regime and would generate millions of trials.
pub fn trial_periods(
    lightcurve: &Lightcurve,
    min_period: f64,
    max_period: f64,
    max_fractional_error: f64,
    max_trials: usize,
) -> Vec<f64> {
    let span_s = lightcurve.data_span_s().max(1.0);

    let estimated = if min_period > 0.0 {
        span_s / max_fractional_error * (1.0 / min_period - 1.0 / max_period)
    } else {
        (max_trials + 1) as f64
    };

    if estimated <= max_trials as f64 {
        adaptive_grid(min_period, max_period, max_fractional_error, span_s)
    } else {
        log_spaced_grid(min_period, max_period, max_trials)
    }
}

fn adaptive_grid(min_period: f64, max_period: f64, max_frac_err: f64, span_s: f64) -> Vec<f64> {
    let mut periods = Vec::new();
    let mut p = min_period;
    while p <= max_period {
        periods.push(p);
        let n_cycles = (span_s / p).max(1.0);
        let increment = p * max_frac_err / n_cycles;
        p += increment;
    }
    periods
}

fn log_spaced_grid(min_period: f64, max_period: f64, n: usize) -> Vec<f64> {
    if n < 2 || min_period <= 0.0 || max_period <= min_period {
        return Vec::new();
    }
    let log_min = min_period.ln();
    let log_max = max_period.ln();
    let step = (log_max - log_min) / (n - 1) as f64;
    (0..n).map(|i| (log_min + i as f64 * step).exp()).collect()
}

pub struct StringLengthPeriodEstimator {}


impl StringLengthPeriodEstimator {
    fn compute_string_length(lightcurve: &Lightcurve, period: f64) -> f64 {
        // phase fold observations by period
        let mut folded_obs: Vec<(&Observation, f64)> = lightcurve.observations.iter()
            .map(|obs| {
                let timestamp_unix = obs.timestamp.timestamp() as f64;
                let phase = (timestamp_unix % period) / period;
                (obs, phase)
            })
            .collect();
        // sort by phase
        folded_obs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        // compute string length - handle wrap-around
        let mut string_length = 0.0;
        for i in 0..folded_obs.len() {
            let (obs_a, phase_a) = folded_obs[i];
            let (obs_b, phase_b) = folded_obs[(i + 1) % folded_obs.len()];
            let delta_phase = if i == folded_obs.len() - 1 {
                // wrap-around case
                (phase_b + 1.0) - phase_a
            } else {
                phase_b - phase_a
            };
            let delta_mag = obs_b.std_magnitude - obs_a.std_magnitude;
            string_length += (delta_phase.powi(2) + delta_mag.powi(2)).sqrt();
        }
        string_length
    }

    fn compute_prior_string_length(lightcurve: &Lightcurve) -> f64 {
        // compute string length using random ordering of observations (baseline to compare against)
        // randomize ordering
        let mut prior_obs: Vec<&Observation> = lightcurve.observations.iter().collect();

        let mut rng = rand::rng();
        prior_obs.shuffle(&mut rng);
        let mut string_length = 0.0;
        for i in 0..prior_obs.len() {
            let obs_a = prior_obs[i];
            let obs_b = prior_obs[(i + 1) % prior_obs.len()];
            let delta_phase = 1.0 / prior_obs.len() as f64; // uniform spacing in prior
            let delta_mag = obs_b.std_magnitude - obs_a.std_magnitude;
            string_length += (delta_phase.powi(2) + delta_mag.powi(2)).sqrt();
        }
        string_length
    }

    pub fn estimate_period(lightcurve: &Lightcurve, min_period: f64, max_period: f64, max_fractional_error: f64, threshold_odds_ratio: Option<f64>) -> Option<f64> {
        let trials = trial_periods(lightcurve, min_period, max_period, max_fractional_error, 5000);
        let prior_string_length = Self::compute_prior_string_length(lightcurve);

        let (best_period, best_string_length) = trials
            .par_iter()
            .map(|&period| (period, Self::compute_string_length(lightcurve, period)))
            .reduce(
                || (f64::NAN, f64::MAX),
                |(bp, bl), (p, l)| if l < bl { (p, l) } else { (bp, bl) },
            );
        let best_period = if best_string_length == f64::MAX { None } else { Some(best_period) };

        // Only return a period if it's significantly better than the prior
        let improvement_threshold = if let Some(ratio) = threshold_odds_ratio {
            1.0 / ratio
        } else {
            0.2 // default threshold: 80% improvement
        };
        if best_string_length < prior_string_length * improvement_threshold {
            best_period
        } else {
            None
        }
    }
}

/// Bayesian binned-phase periodogram (Gregory & Loredo 1992, ApJ 398, 146).
///
/// For a trial period `P` with `m` phase bins, the data are folded and binned.
/// The signal model is `y_ij = μ_j + ε`, `ε ~ N(0, σ²)`, with Jeffreys priors
/// `p(μ_j) = const` and `p(σ) ∝ 1/σ`. Marginalizing μ_1..μ_m and σ in closed
/// form yields the evidence
///
/// ```text
/// log Z(P, m) = -½ Σ_j log n_j  -  ((N - m) / 2) · log W
///               + log Γ((N - m) / 2)  +  const
/// ```
///
/// where `n_j` is the count in bin `j`, `W = Σ_j Σ_i (y - ȳ_j)²` is the
/// total within-bin RSS, and `N = Σ_j n_j`. The constant model (no
/// variation, `m = 1`) gives the same expression with `n_1 = N` and
/// `W₀ = Σ (y - ȳ)²`. Their ratio is the Bayes factor — log-units.
///
/// Bins with zero or one observations contribute nothing to either sum
/// (treated as absent in the model) so periods that leave gaps aren't
/// unfairly penalized.
pub struct GregoryLoredoPeriodEstimator {}

pub struct GregoryLoredoResult {
    pub period_s: Option<f64>,
    pub log_odds: f64,
    pub periodogram: Vec<(f64, f64)>,
}

impl GregoryLoredoPeriodEstimator {
    /// Run the estimator. `bin_range` defaults to (2, 12) — the m-marginalized
    /// version that lets the data choose model complexity. Pass `Some((m, m))`
    /// to fix m. `log_odds_threshold` defaults to 5.0, `max_trial_periods` to 5000.
    pub fn estimate_period(
        lightcurve: &Lightcurve,
        min_period: f64,
        max_period: f64,
        max_fractional_error: f64,
        bin_range: Option<(usize, usize)>,
        log_odds_threshold: Option<f64>,
        max_trial_periods: Option<usize>,
    ) -> GregoryLoredoResult {
        let (m_min, m_max) = bin_range.unwrap_or((2, 12));
        let log_odds_threshold = log_odds_threshold.unwrap_or(5.0);
        let max_trials = max_trial_periods.unwrap_or(5000);

        let n = lightcurve.observations.len();
        if m_min < 2 || m_max < m_min || n < m_max + 2 {
            return GregoryLoredoResult {
                period_s: None,
                log_odds: 0.0,
                periodogram: Vec::new(),
            };
        }

        let mags: Vec<f64> = lightcurve.observations.iter().map(|o| o.std_magnitude).collect();
        let log_z0 = constant_model_log_z(&mags);

        let trials = trial_periods(lightcurve, min_period, max_period, max_fractional_error, max_trials);

        // Look-elsewhere correction: a log-uniform prior over the period grid
        // gives each trial weight ~1/N_trials, so the per-trial log Bayes factor
        // against the constant model picks up a -log(N_trials) penalty.
        let n_trials = trials.len().max(1);
        let lep = (n_trials as f64).ln();

        // Marginalize over m with a uniform prior in [m_min, m_max]:
        // log Z(P) = logsumexp_m log Z(P, m) - log(m_max - m_min + 1)
        let log_p_m = -((m_max - m_min + 1) as f64).ln();

        let periodogram: Vec<(f64, f64)> = trials
            .par_iter()
            .map(|&p| {
                let log_zs: Vec<f64> = (m_min..=m_max)
                    .map(|m| binned_log_z(lightcurve, p, m))
                    .collect();
                let log_z = log_sum_exp(&log_zs) + log_p_m;
                (p, log_z - log_z0 - lep)
            })
            .collect();

        let (best_period, best_log_odds) = periodogram.par_iter().copied().reduce(
            || (f64::NAN, f64::NEG_INFINITY),
            |(bp, bl), (p, l)| if l > bl { (p, l) } else { (bp, bl) },
        );
        let best_period = if best_log_odds == f64::NEG_INFINITY { None } else { Some(best_period) };

        let period = if best_period.is_some() && best_log_odds > log_odds_threshold {
            best_period
        } else {
            None
        };

        GregoryLoredoResult {
            period_s: period,
            log_odds: best_log_odds,
            periodogram,
        }
    }
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = xs.iter().map(|x| (x - max).exp()).sum();
    max + sum.ln()
}

fn constant_model_log_z(mags: &[f64]) -> f64 {
    let n = mags.len() as f64;
    let mean = mags.iter().sum::<f64>() / n;
    let w0: f64 = mags.iter().map(|m| (m - mean).powi(2)).sum();
    -0.5 * n.ln() - (n - 1.0) / 2.0 * safe_ln(w0) + log_gamma((n - 1.0) / 2.0)
}

fn binned_log_z(lightcurve: &Lightcurve, period: f64, bins: usize) -> f64 {
    let mut bin_data: Vec<Vec<f64>> = (0..bins).map(|_| Vec::new()).collect();

    for o in &lightcurve.observations {
        let t = o.unix_seconds();
        let mut phase = (t % period) / period;
        if phase < 0.0 {
            phase += 1.0;
        }
        let idx = ((phase * bins as f64) as usize).min(bins - 1);
        bin_data[idx].push(o.std_magnitude);
    }

    let mut sum_log_n = 0.0;
    let mut w_total = 0.0;
    let mut n_used = 0usize;
    let mut m_eff = 0usize;

    for xs in &bin_data {
        let nj = xs.len();
        if nj < 2 {
            // Bins with 0 or 1 obs contribute nothing — effectively dropped.
            n_used += nj;
            continue;
        }
        let mean: f64 = xs.iter().sum::<f64>() / nj as f64;
        let rss: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum();
        sum_log_n += (nj as f64).ln();
        w_total += rss;
        n_used += nj;
        m_eff += 1;
    }

    if m_eff < 2 || n_used < m_eff + 1 {
        return safe_ln(0.0);
    }

    let dof = (n_used - m_eff) as f64;
    -0.5 * sum_log_n - dof / 2.0 * safe_ln(w_total) + log_gamma(dof / 2.0)
}

fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 { 1.0e-300_f64.ln() } else { x.ln() }
}

/// Lanczos approximation to log Γ(x); reflection for x < 0.5.
fn log_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln()
            - log_gamma(1.0 - x);
    }
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut a = COEF[0];
    for (i, &c) in COEF[1..].iter().enumerate() {
        a += c / (x + (i + 1) as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Quasi-periodic Gaussian-process marginal-likelihood periodogram.
///
/// Models the lightcurve as f(t) ~ GP(0, k(t,t')) with a quasi-periodic
/// kernel (squared-exponential envelope × periodic component, MacKay 1998):
///
/// ```text
/// k(τ) = σ_f² · exp(-τ² / (2 L²)) · exp(-2 sin²(π τ / P) / λ²)
/// ```
///
/// plus i.i.d. observation noise σ_n². The SE envelope of length scale `L`
/// lets correlation decay over time, so a tumbler whose spin rate drifts
/// across a campaign isn't unfairly penalized — unlike the binned-phase
/// G-L model, which assumes a stationary period.
///
/// For each trial period the other hyperparameters are fixed at sensible
/// defaults derived from the data (override via `hyperparams`). The score
/// is the log marginal likelihood of the QP-GP minus the same GP with no
/// periodic component (period → ∞, leaving SE × σ_f² + noise as baseline).
/// Look-elsewhere correction subtracts log(N_trials) just like G-L.
pub struct QuasiPeriodicGPPeriodEstimator {}

#[derive(Clone, Copy, Debug)]
pub struct QuasiPeriodicGPHyperparams {
    /// SE envelope length scale (seconds). Controls how quickly correlation
    /// decays — short L tolerates non-stationary periods, long L assumes
    /// the same period across the full data span.
    pub length_scale_s: f64,
    /// Periodic kernel "harmonic scale" (dimensionless). Smaller → more
    /// sinusoidal; larger → more flexible per-period shape (multi-peak,
    /// flash-like).
    pub harmonic_scale: f64,
    /// Signal variance σ_f².
    pub signal_variance: f64,
    /// Observation-noise variance σ_n².
    pub noise_variance: f64,
}

pub struct QuasiPeriodicGPResult {
    pub period_s: Option<f64>,
    pub log_odds: f64,
    pub periodogram: Vec<(f64, f64)>,
    pub hyperparams: QuasiPeriodicGPHyperparams,
}

impl QuasiPeriodicGPPeriodEstimator {
    /// Run the estimator. Defaults: hyperparams derived from data
    /// (L = span/2, λ = 1.0, σ_f² = 0.95·var(y), σ_n² = 0.05·var(y));
    /// `log_odds_threshold` = 5.0; `max_trial_periods` = 1000 (lower than
    /// G-L's 5000 because each evaluation is an O(N³) Cholesky).
    pub fn estimate_period(
        lightcurve: &Lightcurve,
        min_period: f64,
        max_period: f64,
        max_fractional_error: f64,
        hyperparams: Option<QuasiPeriodicGPHyperparams>,
        log_odds_threshold: Option<f64>,
        max_trial_periods: Option<usize>,
    ) -> QuasiPeriodicGPResult {
        let log_odds_threshold = log_odds_threshold.unwrap_or(5.0);
        let max_trials = max_trial_periods.unwrap_or(1000);

        let n = lightcurve.observations.len();
        let hp = hyperparams.unwrap_or_else(|| default_qp_hyperparams(lightcurve));

        if n < 10 {
            return QuasiPeriodicGPResult {
                period_s: None,
                log_odds: 0.0,
                periodogram: Vec::new(),
                hyperparams: hp,
            };
        }

        // Center t at the first observation for numerical stability — exp((Δt)²)
        // with raw unix seconds would silently underflow long before Cholesky.
        let t0 = lightcurve
            .observations
            .iter()
            .map(|o| o.unix_seconds())
            .fold(f64::INFINITY, f64::min);
        let t: Vec<f64> = lightcurve
            .observations
            .iter()
            .map(|o| o.unix_seconds() - t0)
            .collect();
        let mean_y: f64 =
            lightcurve.observations.iter().map(|o| o.std_magnitude).sum::<f64>() / n as f64;
        let y: Vec<f64> = lightcurve
            .observations
            .iter()
            .map(|o| o.std_magnitude - mean_y)
            .collect();
        let var_y = y.iter().map(|v| v * v).sum::<f64>() / (n - 1).max(1) as f64;

        // Baseline: i.i.d. Gaussian with variance = sample variance of y. This
        // tests "does adding QP structure beat treating the data as pure noise?"
        // — appropriate when σ_f² may be near zero (true white noise input).
        let log_ml_baseline = noise_only_log_marginal_likelihood(&y, var_y.max(1.0e-12));

        let trials = trial_periods(lightcurve, min_period, max_period, max_fractional_error, max_trials);
        let n_trials = trials.len().max(1);
        let lep = (n_trials as f64).ln();

        let periodogram: Vec<(f64, f64)> = trials
            .par_iter()
            .map(|&p| {
                let gram = build_gram_qp(&t, p, &hp);
                let log_ml = gp_log_marginal_likelihood(gram, &y);
                (p, log_ml - log_ml_baseline - lep)
            })
            .collect();

        let (best_period, best_log_odds) = periodogram.par_iter().copied().reduce(
            || (f64::NAN, f64::NEG_INFINITY),
            |(bp, bl), (p, l)| if l > bl { (p, l) } else { (bp, bl) },
        );
        let best_period = if best_log_odds == f64::NEG_INFINITY {
            None
        } else {
            Some(best_period)
        };

        let period = if best_period.is_some() && best_log_odds > log_odds_threshold {
            best_period
        } else {
            None
        };

        QuasiPeriodicGPResult {
            period_s: period,
            log_odds: best_log_odds,
            periodogram,
            hyperparams: hp,
        }
    }
}

fn default_qp_hyperparams(lc: &Lightcurve) -> QuasiPeriodicGPHyperparams {
    let n = lc.observations.len();
    let mags: Vec<f64> = lc.observations.iter().map(|o| o.std_magnitude).collect();
    let mean = if n == 0 { 0.0 } else { mags.iter().sum::<f64>() / n as f64 };
    let var = if n > 1 {
        mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        1.0
    };
    let var = var.max(1.0e-12);
    let span = lc.data_span_s().max(1.0);

    // Estimate σ_n² from consecutive-time-pair squared differences. For a
    // smooth signal nearby points are similar so the diff captures noise;
    // for white noise the diff is ~ 2·var(y), which gets us back σ_n² ≈ var(y).
    // This makes the GP gracefully collapse to noise-only on aperiodic data.
    let noise_variance = consecutive_pair_noise_variance(lc).clamp(1.0e-12, var);
    let signal_variance = (var - noise_variance).max(1.0e-6 * var);

    QuasiPeriodicGPHyperparams {
        length_scale_s: span / 2.0,
        harmonic_scale: 1.0,
        signal_variance,
        noise_variance,
    }
}

fn consecutive_pair_noise_variance(lc: &Lightcurve) -> f64 {
    let sorted = lc.observations_sorted_by_time();
    let n = sorted.len();
    if n < 2 {
        return 1.0e-12;
    }
    let mut diffs: Vec<f64> = (0..n - 1)
        .map(|i| (sorted[i + 1].std_magnitude - sorted[i].std_magnitude).powi(2) / 2.0)
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    diffs[diffs.len() / 2]
}

fn noise_only_log_marginal_likelihood(y: &[f64], variance: f64) -> f64 {
    // y is mean-subtracted; iid N(0, σ²).
    let n = y.len() as f64;
    let ss: f64 = y.iter().map(|v| v * v).sum();
    -0.5 * ss / variance - 0.5 * n * (2.0 * std::f64::consts::PI * variance).ln()
}

fn qp_kernel(dt: f64, period: f64, hp: &QuasiPeriodicGPHyperparams) -> f64 {
    let two_l2 = 2.0 * hp.length_scale_s * hp.length_scale_s;
    let lambda2 = hp.harmonic_scale * hp.harmonic_scale;
    let se = (-(dt * dt) / two_l2).exp();
    let arg = std::f64::consts::PI * dt / period;
    let per = (-2.0 * arg.sin().powi(2) / lambda2).exp();
    hp.signal_variance * se * per
}

fn build_gram_qp(t: &[f64], period: f64, hp: &QuasiPeriodicGPHyperparams) -> Vec<Vec<f64>> {
    let n = t.len();
    let mut a = vec![vec![0.0_f64; n]; n];
    let jitter = hp.noise_variance + 1.0e-8 * hp.signal_variance.max(1.0e-12);
    for i in 0..n {
        for j in 0..=i {
            let val = qp_kernel(t[i] - t[j], period, hp);
            a[i][j] = val;
            if i != j {
                a[j][i] = val;
            }
        }
        a[i][i] += jitter;
    }
    a
}

/// log p(y | K) = -½ y^T K^-1 y - ½ log|K| - N/2 log(2π).
/// Uses Cholesky in-place; returns -∞ if K isn't positive definite.
fn gp_log_marginal_likelihood(mut k: Vec<Vec<f64>>, y: &[f64]) -> f64 {
    let n = k.len();
    if cholesky_lower_in_place(&mut k).is_err() {
        return f64::NEG_INFINITY;
    }
    let log_det: f64 = (0..n).map(|i| k[i][i].ln()).sum::<f64>() * 2.0;
    let z = forward_solve_lower(&k, y);
    let quad: f64 = z.iter().map(|x| x * x).sum();
    -0.5 * quad - 0.5 * log_det - (n as f64) / 2.0 * (2.0 * std::f64::consts::PI).ln()
}

/// In-place Cholesky factorization of a symmetric positive-definite matrix:
/// overwrites the lower triangle with L such that L Lᵀ = A; upper triangle zeroed.
fn cholesky_lower_in_place(a: &mut [Vec<f64>]) -> Result<(), &'static str> {
    let n = a.len();
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err("matrix is not positive definite");
                }
                a[i][i] = s.sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
        for j in (i + 1)..n {
            a[i][j] = 0.0;
        }
    }
    Ok(())
}

/// Forward-substitute L x = b for lower-triangular L.
fn forward_solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    #[test]
    fn test_string_length_sine_wave_random_sampling() {
        // Generate a sine wave with known period, sampled at random times
        let true_period = 3600.0; // 1 hour in seconds
        let amplitude = 2.0; // magnitude units
        let mean_magnitude = 10.0;
        let num_samples = 100;

        // Generate random sampling times over 5 periods
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 5.0 * true_period;

        // Use a simple linear congruential generator for deterministic random times
        let mut observations = Vec::new();
        let mut lcg_state: u64 = 12345;

        for _ in 0..num_samples {
            // Generate pseudo-random time offset
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let random_fraction = lcg_state as f64 / (1u64 << 31) as f64;
            let time_offset_sec = random_fraction * time_span;

            let timestamp = base_time + chrono::Duration::seconds(time_offset_sec as i64);
            let t = time_offset_sec;

            // Generate sine wave: magnitude = mean + amplitude * sin(2π * t / period)
            let phase = 2.0 * std::f64::consts::PI * t / true_period;
            let magnitude = mean_magnitude + amplitude * phase.sin();

            // Create observation with dummy range and phase values
            let obs = Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            };
            observations.push(obs);
        }

        let lightcurve = Lightcurve::new(observations, Some(true), Some(true_period));

        // Estimate the period using string length method
        let min_period = 1800.0; // 0.5 hours
        let max_period = 7200.0; // 2 hours
        let max_fractional_error = 0.01;

        let estimated_period = StringLengthPeriodEstimator::estimate_period(
            &lightcurve,
            min_period,
            max_period,
            max_fractional_error,
            Some(4.0)
        );

        assert!(estimated_period.is_some(), "Period estimation should return a value");
        let estimated = estimated_period.unwrap();

        // Check that estimated period is within 5% of true period
        let relative_error = (estimated - true_period).abs() / true_period;
        assert!(
            relative_error < 0.05,
            "Estimated period {} should be within 5% of true period {}. Relative error: {:.2}%",
            estimated,
            true_period,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_string_length_non_periodic_signal() {
        // Generate a non-periodic signal (uniform random noise) and verify no period is detected
        let num_samples = 100;

        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 18000.0; // 5 hours in seconds

        // Generate completely random magnitudes using LCG for deterministic behavior
        let mut observations = Vec::new();
        let mut lcg_state: u64 = 54321;

        for i in 0..num_samples {
            let time_offset_sec = (i as f64 / num_samples as f64) * time_span;
            let timestamp = base_time + chrono::Duration::seconds(time_offset_sec as i64);

            // Generate uniform random magnitude between 8 and 12
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let magnitude = 8.0 + 4.0 * (lcg_state as f64 / (1u64 << 31) as f64);

            let obs = Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            };
            observations.push(obs);
        }

        let lightcurve = Lightcurve::new(observations, Some(false), None);

        // Try to estimate a period
        let min_period = 1800.0; // 0.5 hours
        let max_period = 7200.0; // 2 hours
        let max_fractional_error = 0.01;

        let estimated_period = StringLengthPeriodEstimator::estimate_period(
            &lightcurve,
            min_period,
            max_period,
            max_fractional_error,
            Some(4.0)
        );

        // For non-periodic data, the estimator should return None
        // because no trial period beats the random baseline
        assert!(
            estimated_period.is_none(),
            "Non-periodic signal should not return a period estimate, but got {:?}",
            estimated_period
        );
    }

    fn build_sine_lightcurve(true_period: f64, num_samples: usize, seed: u64) -> Lightcurve {
        let amplitude = 2.0;
        let mean_magnitude = 10.0;
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 5.0 * true_period;

        let mut observations = Vec::new();
        let mut lcg_state: u64 = seed;
        for _ in 0..num_samples {
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let random_fraction = lcg_state as f64 / (1u64 << 31) as f64;
            let time_offset_sec = random_fraction * time_span;
            let timestamp = base_time + chrono::Duration::milliseconds((time_offset_sec * 1000.0) as i64);
            let phase = 2.0 * std::f64::consts::PI * time_offset_sec / true_period;
            let magnitude = mean_magnitude + amplitude * phase.sin();
            observations.push(Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            });
        }
        Lightcurve::new(observations, Some(true), Some(true_period))
    }

    #[test]
    fn test_gregory_loredo_sine_wave_random_sampling() {
        let true_period = 3600.0;
        let lightcurve = build_sine_lightcurve(true_period, 200, 12345);

        let result = GregoryLoredoPeriodEstimator::estimate_period(
            &lightcurve,
            1800.0,
            7200.0,
            0.005,
            None,
            None,
            None,
        );

        assert!(
            result.period_s.is_some(),
            "Bayesian estimator should detect a period for a clean sine wave (log_odds = {})",
            result.log_odds
        );
        let estimated = result.period_s.unwrap();
        let relative_error = (estimated - true_period).abs() / true_period;
        assert!(
            relative_error < 0.05,
            "Estimated period {} should be within 5% of true period {}. Relative error: {:.2}%",
            estimated,
            true_period,
            relative_error * 100.0
        );
        assert!(!result.periodogram.is_empty(), "periodogram must be populated");
    }

    #[test]
    fn test_gregory_loredo_non_periodic_signal() {
        let num_samples = 200;
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 18000.0;

        let mut observations = Vec::new();
        let mut lcg_state: u64 = 54321;
        for i in 0..num_samples {
            let time_offset_sec = (i as f64 / num_samples as f64) * time_span;
            let timestamp = base_time + chrono::Duration::milliseconds((time_offset_sec * 1000.0) as i64);
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let magnitude = 8.0 + 4.0 * (lcg_state as f64 / (1u64 << 31) as f64);
            observations.push(Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            });
        }
        let lightcurve = Lightcurve::new(observations, Some(false), None);

        let result = GregoryLoredoPeriodEstimator::estimate_period(
            &lightcurve,
            1800.0,
            7200.0,
            0.005,
            None,
            None,
            None,
        );

        assert!(
            result.period_s.is_none(),
            "Non-periodic signal should not yield a period (log_odds = {})",
            result.log_odds
        );
    }

    #[test]
    fn test_gregory_loredo_double_peak_tumbler() {
        // Two specular flashes per rotation — a fixed m=2 model would prefer
        // the half-period alias; m-marginalization should recover P_true.
        let true_period = 60.0;
        let num_samples = 300;
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 20.0 * true_period;

        let mut observations = Vec::new();
        let mut lcg_state: u64 = 9001;
        for _ in 0..num_samples {
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let t = (lcg_state as f64 / (1u64 << 31) as f64) * time_span;
            let timestamp = base_time + chrono::Duration::milliseconds((t * 1000.0) as i64);
            // Two narrow Gaussian flashes per rotation, asymmetric brightness so
            // P/2 doesn't fold the two peaks onto each other identically.
            let phi = (t / true_period).fract();
            let g = |center: f64, amp: f64| {
                let mut d = phi - center;
                if d > 0.5 { d -= 1.0; }
                if d < -0.5 { d += 1.0; }
                -amp * (-d * d / (2.0 * 0.04 * 0.04)).exp()
            };
            let magnitude = 12.0 + g(0.20, 3.0) + g(0.70, 1.5);
            observations.push(Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            });
        }
        let lightcurve = Lightcurve::new(observations, Some(true), Some(true_period));

        let result = GregoryLoredoPeriodEstimator::estimate_period(
            &lightcurve,
            20.0,
            120.0,
            0.005,
            None,
            None,
            None,
        );

        assert!(result.period_s.is_some(), "double-peaked tumbler should be detected");
        let estimated = result.period_s.unwrap();
        let rel_err_full = (estimated - true_period).abs() / true_period;
        let rel_err_half = (estimated - true_period / 2.0).abs() / (true_period / 2.0);
        assert!(
            rel_err_full < 0.02,
            "should prefer true period {} over half-period alias; got {} (full err {:.2}%, half err {:.2}%)",
            true_period,
            estimated,
            rel_err_full * 100.0,
            rel_err_half * 100.0
        );
    }

    #[test]
    fn test_log_sum_exp_basic() {
        // logsumexp([0, 0]) = log 2
        assert!((log_sum_exp(&[0.0, 0.0]) - 2.0_f64.ln()).abs() < 1.0e-12);
        // logsumexp([1000, 1000]) = 1000 + log 2 (numerically stable)
        assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2.0_f64.ln())).abs() < 1.0e-9);
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_gamma_known_values() {
        // log Γ(1) = 0, log Γ(2) = 0, log Γ(5) = log(24) ≈ 3.17805
        assert!((log_gamma(1.0)).abs() < 1.0e-9);
        assert!((log_gamma(2.0)).abs() < 1.0e-9);
        assert!((log_gamma(5.0) - 24.0_f64.ln()).abs() < 1.0e-9);
        // log Γ(0.5) = ½ log π
        assert!((log_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1.0e-9);
    }

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        let mut a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        cholesky_lower_in_place(&mut a).unwrap();
        assert!((a[0][0] - 2.0).abs() < 1.0e-12);
        assert!((a[1][0] - 1.0).abs() < 1.0e-12);
        assert!((a[1][1] - 2.0_f64.sqrt()).abs() < 1.0e-12);
        assert_eq!(a[0][1], 0.0);
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        // K = LLᵀ identity-like check: solve K x = b, verify K x ≈ b.
        let k = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 3.0];
        let mut l = k.clone();
        cholesky_lower_in_place(&mut l).unwrap();
        // Solve L z = b, then Lᵀ x = z
        let z = forward_solve_lower(&l, &b);
        // back-substitute Lᵀ x = z manually
        let n = 3;
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = z[i];
            for j in (i + 1)..n {
                s -= l[j][i] * x[j];
            }
            x[i] = s / l[i][i];
        }
        // Verify K x ≈ b
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += k[i][j] * x[j];
            }
            assert!((s - b[i]).abs() < 1.0e-10, "K x_{} = {} but b = {}", i, s, b[i]);
        }
    }

    #[test]
    fn test_quasi_periodic_gp_sine_wave() {
        let true_period = 3600.0;
        let lightcurve = build_sine_lightcurve(true_period, 100, 12345);

        let result = QuasiPeriodicGPPeriodEstimator::estimate_period(
            &lightcurve,
            1800.0,
            7200.0,
            0.01,
            None,
            None,
            None,
        );

        assert!(
            result.period_s.is_some(),
            "QP-GP should detect a period for clean sine wave (log_odds = {})",
            result.log_odds
        );
        let estimated = result.period_s.unwrap();
        let relative_error = (estimated - true_period).abs() / true_period;
        assert!(
            relative_error < 0.05,
            "Estimated period {} should be within 5% of true period {} (rel err {:.2}%)",
            estimated,
            true_period,
            relative_error * 100.0
        );
        assert!(!result.periodogram.is_empty());
    }

    #[test]
    fn test_quasi_periodic_gp_noisy_sine_wave() {
        // Sine wave + ~5% noise. Tests that the consecutive-pair noise
        // estimator gracefully handles non-zero noise floors.
        let true_period = 3600.0;
        let amplitude = 2.0;
        let mean_magnitude = 10.0;
        let num_samples = 120;
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 5.0 * true_period;

        let mut observations = Vec::new();
        let mut lcg_state: u64 = 24680;
        for _ in 0..num_samples {
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let frac = lcg_state as f64 / (1u64 << 31) as f64;
            let t = frac * time_span;
            let timestamp = base_time + chrono::Duration::milliseconds((t * 1000.0) as i64);

            // Box-Muller-ish noise from another LCG draw — small amplitude.
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let u = lcg_state as f64 / (1u64 << 31) as f64;
            let noise = 0.1 * (u - 0.5);

            let phase = 2.0 * std::f64::consts::PI * t / true_period;
            let magnitude = mean_magnitude + amplitude * phase.sin() + noise;
            observations.push(Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            });
        }
        let lightcurve = Lightcurve::new(observations, Some(true), Some(true_period));

        let result = QuasiPeriodicGPPeriodEstimator::estimate_period(
            &lightcurve,
            1800.0,
            7200.0,
            0.01,
            None,
            None,
            None,
        );

        assert!(
            result.period_s.is_some(),
            "QP-GP should detect a period in noisy sine (log_odds = {})",
            result.log_odds
        );
        let estimated = result.period_s.unwrap();
        let rel_err = (estimated - true_period).abs() / true_period;
        assert!(
            rel_err < 0.05,
            "Estimated {} should be within 5% of true {} (err {:.2}%)",
            estimated,
            true_period,
            rel_err * 100.0
        );
        assert!(
            result.hyperparams.noise_variance > 0.0,
            "noise_variance should be positive for noisy data"
        );
    }

    #[test]
    fn test_quasi_periodic_gp_non_periodic_signal() {
        let num_samples = 100;
        let base_time = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let time_span = 18000.0;

        let mut observations = Vec::new();
        let mut lcg_state: u64 = 54321;
        for i in 0..num_samples {
            let time_offset_sec = (i as f64 / num_samples as f64) * time_span;
            let timestamp = base_time + chrono::Duration::milliseconds((time_offset_sec * 1000.0) as i64);
            lcg_state = (lcg_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
            let magnitude = 8.0 + 4.0 * (lcg_state as f64 / (1u64 << 31) as f64);
            observations.push(Observation {
                vismag: magnitude,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: magnitude,
                timestamp,
                fractional_period: None,
            });
        }
        let lightcurve = Lightcurve::new(observations, Some(false), None);

        let result = QuasiPeriodicGPPeriodEstimator::estimate_period(
            &lightcurve,
            1800.0,
            7200.0,
            0.01,
            None,
            None,
            None,
        );

        assert!(
            result.period_s.is_none(),
            "Non-periodic signal should not yield a period (log_odds = {})",
            result.log_odds
        );
    }
}
