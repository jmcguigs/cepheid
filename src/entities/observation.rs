use chrono::{DateTime, Utc};


pub struct Observation {
    pub vismag: f64,
    pub range_m: f64,
    pub phase_rad: f64,
    pub std_magnitude: f64,
    pub timestamp: DateTime<Utc>,
    pub fractional_period: Option<f64> // Optional: phase within a periodic cycle
}


impl Observation {
    pub fn new(
        vismag: f64,
        range_m: f64,
        phase_rad: f64,
        timestamp: DateTime<Utc>,
        std_range: f64,
        std_phase: f64,
    ) -> Self {
        let std_magnitude = crate::functions::normalization::normalize_vismag(
            vismag,
            range_m,
            std_range,
            phase_rad,
            std_phase,
        );
        Observation {
            vismag,
            range_m,
            phase_rad,
            std_magnitude,
            timestamp,
            fractional_period: None, // default to None, compute later if needed
        }
    }

    pub fn new_default_normalization(vismag: f64, range_m: f64, phase_rad: f64, timestamp: DateTime<Utc>) -> Self {
        // new with default std range 1000 km and std phase 90 deg
        let std_magnitude = crate::functions::normalization::normalize_vismag(
            vismag,
            range_m,
            1000.0e3, // default std range 1000 km
            phase_rad,
            90.0_f64.to_radians(), // default std phase 90 deg
        );
        Observation {
            vismag,
            range_m,
            phase_rad,
            std_magnitude,
            timestamp,
            fractional_period: None,
        }
    }
}