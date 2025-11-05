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

    pub fn data_span_s(&self) -> f64 {
        let obs_by_time = self.observations_sorted_by_time();
        (obs_by_time.last().unwrap().timestamp.timestamp_micros() as f64 
            - obs_by_time.first().unwrap().timestamp.timestamp_micros() as f64) / 1_000_000.0
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
                let timestamp_unix = obs.timestamp.timestamp_micros() as f64 / 1_000_000.0;
                obs.fractional_period = Some((timestamp_unix % period) / period);
            }
        }
        else {
            for obs in &mut self.observations {
                obs.fractional_period = None;
            }
        }
    }
}