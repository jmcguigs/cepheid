use crate::entities::lightcurve::Lightcurve;

/// Duplicate-time quantum: 1 microsecond, matching [`Observation::unix_seconds`](crate::entities::observation::Observation::unix_seconds).
pub const T_DUP_S: f64 = 1.0e-6;

/// Irregular, already-standardized 1-D series.
///
/// Constructors enforce: finite `t`/`y` after filtering; `σ > 0` when present;
/// sorted `t`; pairs with `|t_i − t_j| ≤ T_DUP_S` merged; all vectors the same length.
#[derive(Clone, Debug)]
pub struct Series {
    t_s: Vec<f64>,
    y: Vec<f64>,
    sigma: SigmaSpec,
    covariates: Covariates,
    meta: SeriesMeta,
    n_merged_duplicates: usize,
}

/// Distinguish “caller did not measure errors” from “errors really are 1”.
///
/// Never overwrite [`Homoscedastic`](SigmaSpec::Homoscedastic) / [`PerPoint`](SigmaSpec::PerPoint)
/// because the values happen to be 1.
#[derive(Clone, Debug, PartialEq)]
pub enum SigmaSpec {
    /// No measured errors. Weighted methods are *unweighted*. Scale-needing
    /// methods (G-L, QP-GP, PDM scatter, pair-noise) may estimate σ from
    /// the data — only in this variant.
    Unknown,
    /// One known σ for every point (including the legitimate value 1.0).
    Homoscedastic(f64),
    /// Per-point σ, same length as `t_s`. All entries must be finite and `> 0`.
    PerPoint(Vec<f64>),
}

impl SigmaSpec {
    /// `true` only when the caller did not supply errors. Pair-noise / scale
    /// estimators must run only in this case.
    pub fn is_unknown(&self) -> bool {
        matches!(self, SigmaSpec::Unknown)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Covariates {
    pub solar_phase_rad: Option<Vec<f64>>,
    pub range_m: Option<Vec<f64>>,
    pub elevation_rad: Option<Vec<f64>>,
    pub sensor_key: Option<Vec<u16>>,
    pub keep: Option<Vec<bool>>,
}

#[derive(Clone, Debug)]
pub struct SeriesMeta {
    pub modality: Modality,
    pub y_unit: YUnit,
    pub label: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    OpticalPhotometry,
    RfPower,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YUnit {
    Magnitude,
    Decibels,
    LinearPower,
    Dimensionless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeriesError {
    LengthMismatch,
    TooFewPoints { n: usize, min: usize },
    NonFinite,
    NonPositiveSigma,
    EmptyAfterFilter,
}

impl std::fmt::Display for SeriesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeriesError::LengthMismatch => {
                write!(f, "series vectors have inconsistent lengths")
            }
            SeriesError::TooFewPoints { n, min } => {
                write!(f, "series has {n} points; need at least {min}")
            }
            SeriesError::NonFinite => write!(f, "series contains non-finite t or y"),
            SeriesError::NonPositiveSigma => {
                write!(f, "sigma must be finite and strictly positive")
            }
            SeriesError::EmptyAfterFilter => {
                write!(f, "no points remain after keep / finite-value filter")
            }
        }
    }
}

impl std::error::Error for SeriesError {}

struct Row {
    t: f64,
    y: f64,
    inv_var: f64,
    phase: Option<f64>,
    range: Option<f64>,
    elev: Option<f64>,
    sensor: Option<u16>,
}

impl Row {
    fn absorb(&mut self, other: &Row) {
        let w = self.inv_var + other.inv_var;
        self.t = (self.t * self.inv_var + other.t * other.inv_var) / w;
        self.y = (self.y * self.inv_var + other.y * other.inv_var) / w;
        self.phase = weighted_opt(self.phase, self.inv_var, other.phase, other.inv_var, w);
        self.range = weighted_opt(self.range, self.inv_var, other.range, other.inv_var, w);
        self.elev = weighted_opt(self.elev, self.inv_var, other.elev, other.inv_var, w);
        if self.sensor.is_none() {
            self.sensor = other.sensor;
        }
        self.inv_var = w;
    }
}

fn weighted_opt(a: Option<f64>, wa: f64, b: Option<f64>, wb: f64, w: f64) -> Option<f64> {
    match (a, b) {
        (Some(av), Some(bv)) if av.is_finite() && bv.is_finite() => Some((av * wa + bv * wb) / w),
        (Some(av), _) if av.is_finite() => Some(av),
        (_, Some(bv)) if bv.is_finite() => Some(bv),
        _ => None,
    }
}

fn inv_var_of(sigma: &SigmaSpec, i: usize) -> f64 {
    match sigma {
        SigmaSpec::Unknown => 1.0,
        SigmaSpec::Homoscedastic(s) => 1.0 / (s * s),
        SigmaSpec::PerPoint(s) => 1.0 / (s[i] * s[i]),
    }
}

impl Series {
    pub fn try_new(
        t_s: Vec<f64>,
        y: Vec<f64>,
        sigma: SigmaSpec,
        covariates: Covariates,
        meta: SeriesMeta,
    ) -> Result<Self, SeriesError> {
        let n = t_s.len();
        if y.len() != n {
            return Err(SeriesError::LengthMismatch);
        }
        match &sigma {
            SigmaSpec::PerPoint(s) if s.len() != n => {
                return Err(SeriesError::LengthMismatch);
            }
            SigmaSpec::Homoscedastic(s) if !s.is_finite() || *s <= 0.0 => {
                return Err(SeriesError::NonPositiveSigma);
            }
            SigmaSpec::PerPoint(s) if s.iter().any(|v| !v.is_finite() || *v <= 0.0) => {
                return Err(SeriesError::NonPositiveSigma);
            }
            _ => {}
        }
        for opt_f in [
            &covariates.solar_phase_rad,
            &covariates.range_m,
            &covariates.elevation_rad,
        ] {
            if let Some(v) = opt_f {
                if v.len() != n {
                    return Err(SeriesError::LengthMismatch);
                }
            }
        }
        if let Some(v) = &covariates.sensor_key {
            if v.len() != n {
                return Err(SeriesError::LengthMismatch);
            }
        }
        if let Some(v) = &covariates.keep {
            if v.len() != n {
                return Err(SeriesError::LengthMismatch);
            }
        }

        if n == 0 {
            return Ok(Self {
                t_s,
                y,
                sigma,
                covariates,
                meta,
                n_merged_duplicates: 0,
            });
        }

        let mut idx: Vec<usize> = (0..n)
            .filter(|&i| {
                let keep_ok = covariates.keep.as_ref().map(|k| k[i]).unwrap_or(true);
                keep_ok && t_s[i].is_finite() && y[i].is_finite()
            })
            .collect();

        if idx.is_empty() {
            return Err(SeriesError::EmptyAfterFilter);
        }

        idx.sort_by(|&a, &b| t_s[a].total_cmp(&t_s[b]));

        let mut rows: Vec<Row> = Vec::with_capacity(idx.len());
        for &i in &idx {
            rows.push(Row {
                t: t_s[i],
                y: y[i],
                inv_var: inv_var_of(&sigma, i),
                phase: covariates.solar_phase_rad.as_ref().map(|v| v[i]),
                range: covariates.range_m.as_ref().map(|v| v[i]),
                elev: covariates.elevation_rad.as_ref().map(|v| v[i]),
                sensor: covariates.sensor_key.as_ref().map(|v| v[i]),
            });
        }

        let mut merged: Vec<Row> = Vec::with_capacity(rows.len());
        let mut n_merged_duplicates = 0usize;
        for row in rows {
            if let Some(last) = merged.last_mut() {
                if (row.t - last.t).abs() <= T_DUP_S {
                    last.absorb(&row);
                    n_merged_duplicates += 1;
                    continue;
                }
            }
            merged.push(row);
        }

        let m = merged.len();
        let mut t_out = Vec::with_capacity(m);
        let mut y_out = Vec::with_capacity(m);
        let mut phase_out = Vec::with_capacity(m);
        let mut range_out = Vec::with_capacity(m);
        let mut elev_out = Vec::with_capacity(m);
        let mut sensor_out = Vec::with_capacity(m);
        let mut sigma_out = Vec::with_capacity(m);
        let has_phase = covariates.solar_phase_rad.is_some();
        let has_range = covariates.range_m.is_some();
        let has_elev = covariates.elevation_rad.is_some();
        let has_sensor = covariates.sensor_key.is_some();

        for r in &merged {
            t_out.push(r.t);
            y_out.push(r.y);
            if has_phase {
                phase_out.push(r.phase.unwrap_or(f64::NAN));
            }
            if has_range {
                range_out.push(r.range.unwrap_or(f64::NAN));
            }
            if has_elev {
                elev_out.push(r.elev.unwrap_or(f64::NAN));
            }
            if has_sensor {
                sensor_out.push(r.sensor.unwrap_or(0));
            }
            sigma_out.push(1.0 / r.inv_var.sqrt());
        }

        let sigma = match sigma {
            SigmaSpec::Unknown => SigmaSpec::Unknown,
            SigmaSpec::Homoscedastic(s) => SigmaSpec::Homoscedastic(s),
            SigmaSpec::PerPoint(_) => SigmaSpec::PerPoint(sigma_out),
        };

        Ok(Self {
            t_s: t_out,
            y: y_out,
            sigma,
            covariates: Covariates {
                solar_phase_rad: has_phase.then_some(phase_out),
                range_m: has_range.then_some(range_out),
                elevation_rad: has_elev.then_some(elev_out),
                sensor_key: has_sensor.then_some(sensor_out),
                keep: None,
            },
            meta,
            n_merged_duplicates,
        })
    }

    /// Optical ingest: `t = unix_seconds()`, `y = std_magnitude`, `sigma = Unknown`.
    pub fn from_lightcurve(lc: &Lightcurve) -> Result<Self, SeriesError> {
        let n = lc.observations.len();
        let mut t_s = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut phases = Vec::with_capacity(n);
        let mut ranges = Vec::with_capacity(n);
        for o in &lc.observations {
            t_s.push(o.unix_seconds());
            y.push(o.std_magnitude);
            phases.push(o.phase_rad);
            ranges.push(o.range_m);
        }
        Self::try_new(
            t_s,
            y,
            SigmaSpec::Unknown,
            Covariates {
                solar_phase_rad: (n > 0).then_some(phases),
                range_m: (n > 0).then_some(ranges),
                ..Covariates::default()
            },
            SeriesMeta {
                modality: Modality::OpticalPhotometry,
                y_unit: YUnit::Magnitude,
                label: None,
            },
        )
    }

    pub fn from_rf_power(
        samples: &[crate::entities::rf::RfPowerObservation],
        cfg: &crate::entities::rf::RfNormConfig,
    ) -> Result<Self, SeriesError> {
        crate::functions::rf_normalization::standardize_rf(samples, cfg)
    }

    pub fn t_s(&self) -> &[f64] {
        &self.t_s
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn sigma_spec(&self) -> &SigmaSpec {
        &self.sigma
    }

    /// Alias of [`sigma_spec`](Self::sigma_spec).
    pub fn sigma(&self) -> &SigmaSpec {
        &self.sigma
    }

    pub fn covariates(&self) -> &Covariates {
        &self.covariates
    }

    pub fn meta(&self) -> &SeriesMeta {
        &self.meta
    }

    pub fn n_merged_duplicates(&self) -> usize {
        self.n_merged_duplicates
    }

    pub fn len(&self) -> usize {
        self.t_s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t_s.is_empty()
    }

    /// `None` if fewer than two points — never panics.
    pub fn span_s(&self) -> Option<f64> {
        if self.t_s.len() < 2 {
            return None;
        }
        Some(self.t_s[self.t_s.len() - 1] - self.t_s[0])
    }

    /// GLS weights: `1/σ²` when σ is known, **all ones** for [`SigmaSpec::Unknown`].
    pub fn weights(&self) -> Vec<f64> {
        match &self.sigma {
            SigmaSpec::Unknown => vec![1.0; self.t_s.len()],
            SigmaSpec::Homoscedastic(s) => {
                let w = 1.0 / (s * s);
                vec![w; self.t_s.len()]
            }
            SigmaSpec::PerPoint(s) => s.iter().map(|v| 1.0 / (v * v)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::observation::Observation;
    use chrono::{DateTime, Utc};

    fn meta() -> SeriesMeta {
        SeriesMeta {
            modality: Modality::Generic,
            y_unit: YUnit::Dimensionless,
            label: None,
        }
    }

    #[test]
    fn empty_series_is_ok_and_does_not_panic() {
        let s = Series::try_new(
            vec![],
            vec![],
            SigmaSpec::Unknown,
            Covariates::default(),
            meta(),
        )
        .unwrap();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.span_s().is_none());
        assert!(s.weights().is_empty());
    }

    #[test]
    fn single_point_span_is_none() {
        let s = Series::try_new(
            vec![1.0],
            vec![2.0],
            SigmaSpec::Unknown,
            Covariates::default(),
            meta(),
        )
        .unwrap();
        assert_eq!(s.len(), 1);
        assert!(s.span_s().is_none());
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let err = Series::try_new(
            vec![0.0, 1.0],
            vec![0.0],
            SigmaSpec::Unknown,
            Covariates::default(),
            meta(),
        )
        .unwrap_err();
        assert_eq!(err, SeriesError::LengthMismatch);
    }

    #[test]
    fn non_positive_sigma_is_rejected() {
        let err = Series::try_new(
            vec![0.0],
            vec![1.0],
            SigmaSpec::Homoscedastic(0.0),
            Covariates::default(),
            meta(),
        )
        .unwrap_err();
        assert_eq!(err, SeriesError::NonPositiveSigma);
    }

    #[test]
    fn t17_duplicate_timestamps_within_t_dup_are_merged() {
        let s = Series::try_new(
            vec![0.0, 0.5e-6, 1.0],
            vec![1.0, 3.0, 2.0],
            SigmaSpec::Unknown,
            Covariates::default(),
            meta(),
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.n_merged_duplicates(), 1);
        assert!((s.y()[0] - 2.0).abs() < 1.0e-12);
        assert!((s.y()[1] - 2.0).abs() < 1.0e-12);
        assert!(s.t_s()[0] <= T_DUP_S);
        assert!((s.t_s()[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn t17b_unknown_vs_homoscedastic_one_are_distinct() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let unknown = Series::try_new(
            t.clone(),
            y.clone(),
            SigmaSpec::Unknown,
            Covariates::default(),
            meta(),
        )
        .unwrap();
        let known_one = Series::try_new(
            t.clone(),
            y.clone(),
            SigmaSpec::Homoscedastic(1.0),
            Covariates::default(),
            meta(),
        )
        .unwrap();
        let known_two = Series::try_new(
            t,
            y,
            SigmaSpec::Homoscedastic(2.0),
            Covariates::default(),
            meta(),
        )
        .unwrap();

        assert!(unknown.sigma_spec().is_unknown());
        assert!(!known_one.sigma_spec().is_unknown());
        assert!(!matches!(unknown.sigma_spec(), SigmaSpec::Homoscedastic(_)));
        assert!(matches!(
            known_one.sigma_spec(),
            SigmaSpec::Homoscedastic(s) if *s == 1.0
        ));

        // Weights are numerically 1 for both Unknown and Homoscedastic(1),
        // but only Unknown may run a pair-noise / scale estimator later.
        assert!(unknown.weights().iter().all(|w| (*w - 1.0).abs() < 1e-15));
        assert!(known_one.weights().iter().all(|w| (*w - 1.0).abs() < 1e-15));
        assert!(
            known_two
                .weights()
                .iter()
                .all(|w| (*w - 0.25).abs() < 1e-15)
        );
    }

    #[test]
    fn per_point_weights_are_inverse_variance() {
        let s = Series::try_new(
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            SigmaSpec::PerPoint(vec![1.0, 2.0]),
            Covariates::default(),
            meta(),
        )
        .unwrap();
        let w = s.weights();
        assert!((w[0] - 1.0).abs() < 1e-15);
        assert!((w[1] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn keep_filter_drops_points() {
        let s = Series::try_new(
            vec![0.0, 1.0, 2.0],
            vec![4.0, 5.0, 6.0],
            SigmaSpec::Unknown,
            Covariates {
                keep: Some(vec![true, false, true]),
                ..Covariates::default()
            },
            meta(),
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.y(), &[4.0, 6.0]);
        assert!(s.covariates().keep.is_none());
    }

    #[test]
    fn from_lightcurve_uses_unknown_sigma_and_copies_covariates() {
        let ts = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let obs = Observation {
            vismag: 10.0,
            range_m: 1.0e6,
            phase_rad: 0.5,
            std_magnitude: 9.5,
            timestamp: ts,
            fractional_period: None,
        };
        let lc = Lightcurve::new(vec![obs], None, None);
        let s = Series::from_lightcurve(&lc).unwrap();
        assert!(s.sigma_spec().is_unknown());
        assert_eq!(s.meta().modality, Modality::OpticalPhotometry);
        assert_eq!(s.meta().y_unit, YUnit::Magnitude);
        assert_eq!(s.y(), &[9.5]);
        assert_eq!(s.covariates().range_m.as_deref(), Some(&[1.0e6][..]));
        assert_eq!(s.covariates().solar_phase_rad.as_deref(), Some(&[0.5][..]));
        assert!(s.span_s().is_none());
    }

    #[test]
    fn from_empty_lightcurve_is_ok() {
        let lc = Lightcurve::new(vec![], None, None);
        let s = Series::from_lightcurve(&lc).unwrap();
        assert!(s.is_empty());
        assert!(s.span_s().is_none());
    }
}
