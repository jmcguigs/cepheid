use chrono::{DateTime, Utc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarization {
    H,
    V,
    Lhcp,
    Rhcp,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct RfPowerObservation {
    pub timestamp: DateTime<Utc>,
    pub power_dbm: f64,
    pub sigma_db: Option<f64>,
    pub range_m: Option<f64>,
    pub frequency_hz: Option<f64>,
    pub elevation_rad: Option<f64>,
    pub saturated: bool,
    pub agc_engaged: Option<bool>,
    pub sensor_id: Option<String>,
    pub polarization: Option<Polarization>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RfYDomain {
    Decibel,
    Linear,
}

#[derive(Clone, Debug)]
pub struct RfNormConfig {
    pub path_exponent: f64,
    pub ref_range_m: f64,
    pub drop_saturated: bool,
    pub drop_agc_engaged: bool,
    pub y_domain: RfYDomain,
}

impl Default for RfNormConfig {
    fn default() -> Self {
        Self {
            path_exponent: 2.0,
            ref_range_m: 1.0e6,
            drop_saturated: true,
            drop_agc_engaged: true,
            y_domain: RfYDomain::Decibel,
        }
    }
}
