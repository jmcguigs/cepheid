//! Frozen SSA sampling generators for integration tests.

use cepheid::entities::series::{Covariates, Modality, Series, SeriesMeta, SigmaSpec, YUnit};
use cepheid::functions::sampling::{geo_night_times, leo_pass_times};

pub const LEO_N_PASSES: usize = 18;
pub const LEO_PTS: usize = 40;
pub const LEO_PASS_LEN: f64 = 480.0;
pub const LEO_ORBIT: f64 = 5400.0;

pub fn leo_week_times() -> Vec<f64> {
    leo_pass_times(LEO_N_PASSES, LEO_PTS, LEO_PASS_LEN, LEO_ORBIT, 0.0)
}

#[allow(dead_code)]
pub fn geo_week_times() -> Vec<f64> {
    geo_night_times(7, 80, 18_000.0, 0.0)
}

pub fn series_xy(t: Vec<f64>, y: Vec<f64>) -> Series {
    series_xy_phase(t, y, None)
}

pub fn series_xy_phase(t: Vec<f64>, y: Vec<f64>, phase: Option<Vec<f64>>) -> Series {
    Series::try_new(
        t,
        y,
        SigmaSpec::Unknown,
        Covariates {
            solar_phase_rad: phase,
            ..Covariates::default()
        },
        SeriesMeta {
            modality: Modality::OpticalPhotometry,
            y_unit: YUnit::Magnitude,
            label: None,
        },
    )
    .expect("series")
}

pub fn leftover_lambert(phase: &[f64], c: f64) -> Vec<f64> {
    phase.iter().map(|ph| c * (1.0 + ph.cos()) / 2.0).collect()
}
