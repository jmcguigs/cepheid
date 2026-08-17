/// Three-way product decision. Fields on [`PeriodicityAssessment`] will grow
/// in a later PR; the type is `non_exhaustive` so struct literals stay
/// crate-internal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodicityDecision {
    Periodic,
    NotPeriodic,
    Inconclusive,
}

/// Minimal assessment record so [`Lightcurve::apply_assessment`](crate::entities::lightcurve::Lightcurve::apply_assessment)
/// can land with the Series IR. Later PRs add FAPs, aliases, and diagnostics.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PeriodicityAssessment {
    pub decision: PeriodicityDecision,
    pub period_s: Option<f64>,
}

impl PeriodicityAssessment {
    pub fn new(decision: PeriodicityDecision, period_s: Option<f64>) -> Self {
        Self { decision, period_s }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreKind {
    GlsPower,
    PdmTheta,
    LogOdds,
    StringLengthRatio,
    /// DFT sampling window \(W(f)\). Not a data periodogram.
    SpectralWindow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AliasKind {
    Harmonic,
    HalfPeriod,
    SiderealDay,
    SolarDay,
    Orbital,
    OrbitalHalf,
    PassCadence,
    PassDuration,
    WindowPeak,
    OtherPeak,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchScale {
    #[default]
    Auto,
    IntraPass,
    InterPass,
    Full,
}

#[derive(Clone, Debug)]
pub struct Periodogram {
    pub period_s: Vec<f64>,
    pub score: Vec<f64>,
    pub score_kind: ScoreKind,
}

impl Periodogram {
    pub fn is_empty(&self) -> bool {
        self.period_s.is_empty()
    }

    pub fn len(&self) -> usize {
        self.period_s.len()
    }
}

#[derive(Clone, Debug)]
pub struct Pass {
    pub t_start_s: f64,
    pub t_end_s: f64,
    pub n: usize,
    pub median_dt_s: f64,
}

impl Pass {
    pub fn duration_s(&self) -> f64 {
        (self.t_end_s - self.t_start_s).max(0.0)
    }

    /// Enough points and duration for an intra-pass short-period search.
    pub fn is_intra_eligible(&self) -> bool {
        self.n >= 12 && self.duration_s() >= 12.0 * self.median_dt_s
    }
}

#[derive(Clone, Debug)]
pub struct WindowPeak {
    pub period_s: f64,
    pub power: f64,
    pub kind: AliasKind,
}

/// Sampling QC computed before any period is reported.
#[derive(Clone, Debug)]
pub struct SamplingDiagnostics {
    pub n: usize,
    pub n_merged_duplicates: usize,
    pub span_s: f64,
    pub min_dt_s: f64,
    pub median_dt_s: f64,
    pub median_intrapass_dt_s: f64,
    pub p95_dt_s: f64,
    /// Σ pass_duration / span.
    pub duty_cycle: f64,
    pub n_passes: usize,
    pub passes: Vec<Pass>,
    pub min_searchable_period_s: f64,
    pub max_searchable_period_s: f64,
    pub spectral_window: Periodogram,
    pub window_peaks: Vec<WindowPeak>,
}

impl SamplingDiagnostics {
    /// QC gate used later by `assess_periodicity`: too few points or too short a span.
    pub fn is_too_sparse(&self) -> bool {
        self.n < 12 || self.span_s < 2.0 * self.min_searchable_period_s
    }
}
