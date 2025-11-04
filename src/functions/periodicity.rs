use crate::entities::observation::Observation;
use crate::entities::lightcurve::Lightcurve;

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

    pub fn estimate_period(lightcurve: &Lightcurve, min_period: f64, max_period: f64, max_fractional_error: f64) -> Option<f64> {
        let trial_periods = Self::generate_trial_periods(lightcurve, min_period, max_period, max_fractional_error);
        let mut best_period = None;
        let mut best_string_length = std::f64::MAX;
        for period in trial_periods {
            let string_length = Self::compute_string_length(lightcurve, period);
            if string_length < best_string_length {
                best_string_length = string_length;
                best_period = Some(period);
            }
        }
        best_period
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
            max_fractional_error
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
}