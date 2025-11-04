use crate::entities::observation::Observation;
use crate::entities::lightcurve::Lightcurve;
use rand::prelude::SliceRandom;

pub struct StringLengthPeriodEstimator {}


impl StringLengthPeriodEstimator {
    // generate trial periods between min and max with given max fractional error
    fn generate_trial_periods(lightcurve: &Lightcurve, min_period: f64, max_period: f64, max_fractional_error: f64) -> Vec<f64> {
        let mut trial_periods = Vec::new();
        let mut period = min_period;
        let obs_by_time = lightcurve.observations_sorted_by_time();
        let data_span_s = obs_by_time.last().unwrap().timestamp.timestamp() as f64 - obs_by_time.first().unwrap().timestamp.timestamp() as f64;
        while period <= max_period {
            trial_periods.push(period);
            let fractional_error = period / data_span_s;
            // increment period based on max fractional error
            period += period * max_fractional_error / fractional_error;
        }
        trial_periods
    }

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
        // compute string length
        let mut string_length = 0.0;
        for i in 0..folded_obs.len() {
            let (obs_a, phase_a) = folded_obs[i];
            let (obs_b, phase_b) = folded_obs[(i + 1) % folded_obs.len()];
            let delta_phase = phase_b - phase_a;
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
        let trial_periods = Self::generate_trial_periods(lightcurve, min_period, max_period, max_fractional_error);
        let prior_string_length = Self::compute_prior_string_length(lightcurve);
        let mut best_period = None;
        let mut best_string_length = std::f64::MAX;
        
        for period in trial_periods {
            let string_length = Self::compute_string_length(lightcurve, period);
            if string_length < best_string_length {
                best_string_length = string_length;
                best_period = Some(period);
            }
        }
        
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
}