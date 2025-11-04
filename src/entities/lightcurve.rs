use crate::entities::observation::Observation;

pub struct Lightcurve {
    pub observations: Vec<Observation>,
    pub is_periodic: Option<bool>,  // Optional: indicates if the lightcurve is periodic
    pub period_sec: Option<f64>,    // Optional: period in seconds if periodic
}

impl Lightcurve {
    pub fn new(observations: Vec<Observation>, is_periodic: Option<bool>, period_sec: Option<f64>) -> Self {
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
}