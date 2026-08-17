use crate::entities::assessment::{PeriodicityAssessment, PeriodicityDecision};
use crate::entities::observation::Observation;
use crate::entities::series::{Series, SeriesError};

#[derive(Clone, Debug)]
pub struct Lightcurve {
    pub observations: Vec<Observation>,
    pub is_periodic: Option<bool>, // Optional: indicates if the lightcurve is periodic
    pub period_sec: Option<f64>,   // Optional: period in seconds if periodic
}

impl Lightcurve {
    pub fn new(
        observations: Vec<Observation>,
        is_periodic: Option<bool>,
        period_sec: Option<f64>,
    ) -> Self {
        Lightcurve {
            observations,
            is_periodic,
            period_sec,
        }
    }

    pub fn add_observation(&mut self, observation: Observation) {
        self.observations.push(observation);
    }

    pub fn observations_sorted_by_time(&self) -> Vec<&Observation> {
        let mut obs_refs: Vec<&Observation> = self.observations.iter().collect();
        obs_refs.sort_by_key(|obs| obs.timestamp);
        obs_refs
    }

    pub fn observations_by_fractional_period(&self) -> Vec<&Observation> {
        // Returns observations sorted by fractional period if available
        let mut obs_refs: Vec<&Observation> = self.observations.iter().collect();
        obs_refs.sort_by(|a, b| {
            a.fractional_period
                .partial_cmp(&b.fractional_period)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        obs_refs
    }

    /// Campaign span in seconds. Returns `0.0` when there are fewer than two
    /// observations — never panics.
    pub fn data_span_s(&self) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let obs_by_time = self.observations_sorted_by_time();
        obs_by_time.last().unwrap().unix_seconds() - obs_by_time.first().unwrap().unix_seconds()
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn update_period(&mut self, period_sec: Option<f64>) {
        self.period_sec = period_sec;
        self.is_periodic = period_sec.map(|_| true);

        // update fractional periods for observations - if None, set to None
        if let Some(period) = period_sec {
            for obs in &mut self.observations {
                obs.fractional_period = Some(obs.unix_seconds().rem_euclid(period) / period);
            }
        } else {
            for obs in &mut self.observations {
                obs.fractional_period = None;
            }
        }
    }

    pub fn to_series(&self) -> Result<Series, SeriesError> {
        Series::from_lightcurve(self)
    }

    pub fn apply_assessment(&mut self, a: &PeriodicityAssessment) {
        match a.decision {
            PeriodicityDecision::Periodic => self.update_period(a.period_s),
            PeriodicityDecision::NotPeriodic | PeriodicityDecision::Inconclusive => {
                self.update_period(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::observation::Observation;
    use chrono::{DateTime, Utc};

    fn obs_at(offset_s: f64) -> Observation {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        Observation {
            vismag: 10.0,
            range_m: 1.0e6,
            phase_rad: 0.0,
            std_magnitude: 10.0,
            timestamp: base + chrono::Duration::milliseconds((offset_s * 1000.0) as i64),
            fractional_period: None,
        }
    }

    #[test]
    fn empty_and_singleton_span_is_zero() {
        let empty = Lightcurve::new(vec![], None, None);
        assert_eq!(empty.data_span_s(), 0.0);
        let one = Lightcurve::new(vec![obs_at(0.0)], None, None);
        assert_eq!(one.data_span_s(), 0.0);
    }

    #[test]
    fn update_period_uses_rem_euclid() {
        let mut lc = Lightcurve::new(vec![obs_at(7.5)], None, None);
        lc.update_period(Some(5.0));
        let phi = lc.observations[0].fractional_period.unwrap();
        assert!((phi - 0.5).abs() < 1e-12);
    }

    #[test]
    fn apply_assessment_maps_non_periodic_to_none() {
        let mut lc = Lightcurve::new(vec![obs_at(0.0), obs_at(1.0)], Some(true), Some(10.0));
        lc.apply_assessment(&PeriodicityAssessment::new(
            PeriodicityDecision::NotPeriodic,
            Some(10.0),
        ));
        assert!(lc.period_sec.is_none());
        assert!(lc.is_periodic.is_none());
        assert!(lc.observations[0].fractional_period.is_none());
    }

    #[test]
    fn to_series_round_trip_unknown_sigma() {
        let lc = Lightcurve::new(vec![obs_at(0.0), obs_at(10.0)], None, None);
        let s = lc.to_series().unwrap();
        assert!(s.sigma_spec().is_unknown());
        assert_eq!(s.len(), 2);
        assert!((s.span_s().unwrap() - 10.0).abs() < 1e-9);
    }
}
