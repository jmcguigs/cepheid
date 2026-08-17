//! Reduce RF downlink samples to a `Series` for `assess_periodicity`.

use crate::entities::rf::{RfNormConfig, RfPowerObservation, RfYDomain};
use crate::entities::series::{
    Covariates, Modality, Series, SeriesError, SeriesMeta, SigmaSpec, YUnit,
};

/// Range-correct (when range is present) and pack into a [`Series`].
///
/// Default stays in dBm: `y = P_dBm + 10 n log10(r/r0)` with `n = 2`.
pub fn standardize_rf(
    samples: &[RfPowerObservation],
    cfg: &RfNormConfig,
) -> Result<Series, SeriesError> {
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut sig = Vec::new();
    let mut elev = Vec::new();
    let mut have_sigma = true;
    let mut have_elev = false;

    for s in samples {
        if cfg.drop_saturated && s.saturated {
            continue;
        }
        if cfg.drop_agc_engaged && s.agc_engaged == Some(true) {
            continue;
        }
        if !s.power_dbm.is_finite() {
            continue;
        }
        let mut y_db = s.power_dbm;
        if let Some(r) = s.range_m {
            if r > 0.0 && r.is_finite() && cfg.ref_range_m > 0.0 {
                y_db += 10.0 * cfg.path_exponent * (r / cfg.ref_range_m).log10();
            }
        }
        let (yi, si) = match cfg.y_domain {
            RfYDomain::Decibel => (y_db, s.sigma_db),
            RfYDomain::Linear => {
                let y_lin = 10.0_f64.powf(y_db / 10.0);
                let sig_lin = s.sigma_db.map(|sd| y_lin * std::f64::consts::LN_10 / 10.0 * sd);
                (y_lin, sig_lin)
            }
        };
        t.push(s.timestamp.timestamp_micros() as f64 / 1.0e6);
        y.push(yi);
        match si {
            Some(v) if v.is_finite() && v > 0.0 => sig.push(v),
            _ => {
                have_sigma = false;
                sig.push(f64::NAN);
            }
        }
        if let Some(el) = s.elevation_rad {
            have_elev = true;
            elev.push(el);
        } else {
            elev.push(f64::NAN);
        }
    }

    if t.len() < 20 {
        return Err(SeriesError::EmptyAfterFilter);
    }

    let sigma = if have_sigma {
        let first = sig[0];
        if sig.iter().all(|v| (*v - first).abs() < 1e-15) {
            SigmaSpec::Homoscedastic(first)
        } else {
            SigmaSpec::PerPoint(sig)
        }
    } else {
        SigmaSpec::Unknown
    };

    let y_unit = match cfg.y_domain {
        RfYDomain::Decibel => YUnit::Decibels,
        RfYDomain::Linear => YUnit::LinearPower,
    };

    Series::try_new(
        t,
        y,
        sigma,
        Covariates {
            elevation_rad: have_elev.then_some(elev),
            ..Covariates::default()
        },
        SeriesMeta {
            modality: Modality::RfPower,
            y_unit,
            label: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assess_periodicity;
    use crate::entities::assessment::{PeriodSearchConfig, PeriodicityDecision, SearchScale};
    use crate::entities::rf::RfPowerObservation;
    use chrono::{DateTime, Utc};

    fn ts(sec: f64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + sec as i64, 0).unwrap()
    }

    fn sample(t: f64, pdbm: f64, range: Option<f64>, agc: Option<bool>) -> RfPowerObservation {
        RfPowerObservation {
            timestamp: ts(t),
            power_dbm: pdbm,
            sigma_db: None,
            range_m: range,
            frequency_hz: None,
            elevation_rad: None,
            saturated: false,
            agc_engaged: agc,
            sensor_id: None,
            polarization: None,
        }
    }

    #[test]
    fn t11_range_law_only_not_periodic() {
        let mut samples = Vec::new();
        for i in 0..80 {
            let t = i as f64 * 10.0;
            let r = 8.0e5 + 4.0e5 * (t / 800.0);
            let pdbm = -20.0 * (r / 1.0e6).log10() - 80.0;
            samples.push(sample(t, pdbm, Some(r), None));
        }
        let s = standardize_rf(&samples, &RfNormConfig::default()).unwrap();
        let mut c = PeriodSearchConfig::conservative();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(40.0);
        c.max_period_s = Some(400.0);
        let a = assess_periodicity(&s, &c);
        assert_ne!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
    }

    #[test]
    fn t11b_missing_range_does_not_invent_spin() {
        let mut samples = Vec::new();
        for i in 0..80 {
            let t = i as f64 * 10.0;
            let r = 8.0e5 + 4.0e5 * (t / 800.0);
            let pdbm = -20.0 * (r / 1.0e6).log10() - 80.0;
            samples.push(sample(t, pdbm, None, None));
        }
        let s = standardize_rf(&samples, &RfNormConfig::default()).unwrap();
        let mut c = PeriodSearchConfig::conservative();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(40.0);
        c.max_period_s = Some(400.0);
        let a = assess_periodicity(&s, &c);
        assert_ne!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
    }

    #[test]
    fn t11c_agc_empty_is_error() {
        let samples: Vec<_> = (0..30)
            .map(|i| sample(i as f64, -90.0, Some(1e6), Some(true)))
            .collect();
        let err = standardize_rf(&samples, &RfNormConfig::default()).unwrap_err();
        assert_eq!(err, SeriesError::EmptyAfterFilter);
    }

    #[test]
    fn t12_rf_sine_period_not_half() {
        let p = 120.0;
        let mut samples = Vec::new();
        for i in 0..150 {
            let t = i as f64 * 8.0;
            let r = 1.0e6;
            let pdbm = -80.0 + 2.0 * (std::f64::consts::TAU * t / p).sin();
            samples.push(sample(t, pdbm, Some(r), None));
        }
        let s = Series::from_rf_power(&samples, &RfNormConfig::default()).unwrap();
        let mut c = PeriodSearchConfig::conservative();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(40.0);
        c.max_period_s = Some(300.0);
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let got = a.period_s.unwrap();
        assert!((got - p).abs() / p < 0.08, "got {got}, want {p} not P/2");
        assert!((got - p / 2.0).abs() / (p / 2.0) > 0.1);
    }
}
