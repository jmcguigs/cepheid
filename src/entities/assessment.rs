//! Periodicity assessment types, search config, and sampling diagnostics.

/// Three-way product decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodicityDecision {
    Periodic,
    NotPeriodic,
    Inconclusive,
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
pub enum MethodId {
    Gls,
    Pdm,
    GregoryLoredo,
    QuasiPeriodicGp,
    StringLength,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DetrendMode {
    #[default]
    Auto,
    LinearTime,
    PhaseFunction,
    Elevation,
    None,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FapMode {
    #[default]
    BaluevH1,
    PermMax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetrendColumn {
    PassMean { pass: usize },
    GlobalMean,
    Tau,
    PhaseLambert,
    Elevation,
    GlobalTime,
}

#[derive(Clone, Debug)]
pub struct Periodogram {
    pub period_s: Vec<f64>,
    pub score: Vec<f64>,
    pub score_kind: ScoreKind,
}

impl Periodogram {
    pub fn empty(kind: ScoreKind) -> Self {
        Self {
            period_s: Vec::new(),
            score: Vec::new(),
            score_kind: kind,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.period_s.is_empty()
    }

    pub fn len(&self) -> usize {
        self.period_s.len()
    }
}

impl Default for Periodogram {
    fn default() -> Self {
        Self::empty(ScoreKind::GlsPower)
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

#[derive(Clone, Debug)]
pub struct SamplingDiagnostics {
    pub n: usize,
    pub n_merged_duplicates: usize,
    pub span_s: f64,
    pub min_dt_s: f64,
    pub median_dt_s: f64,
    pub median_intrapass_dt_s: f64,
    pub p95_dt_s: f64,
    pub duty_cycle: f64,
    pub n_passes: usize,
    pub passes: Vec<Pass>,
    pub min_searchable_period_s: f64,
    pub max_searchable_period_s: f64,
    pub spectral_window: Periodogram,
    pub window_peaks: Vec<WindowPeak>,
}

impl SamplingDiagnostics {
    pub fn is_too_sparse(&self) -> bool {
        self.n < 12 || self.span_s < 2.0 * self.min_searchable_period_s
    }
}

impl Default for SamplingDiagnostics {
    fn default() -> Self {
        Self {
            n: 0,
            n_merged_duplicates: 0,
            span_s: 0.0,
            min_dt_s: 0.0,
            median_dt_s: 0.0,
            median_intrapass_dt_s: 0.0,
            p95_dt_s: 0.0,
            duty_cycle: 0.0,
            n_passes: 0,
            passes: Vec::new(),
            min_searchable_period_s: 0.0,
            max_searchable_period_s: 0.0,
            spectral_window: Periodogram::empty(ScoreKind::SpectralWindow),
            window_peaks: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Alias {
    pub period_s: f64,
    pub score: f64,
    pub kind: AliasKind,
    pub relative_delta: f64,
    pub vetoed: bool,
}

#[derive(Clone, Debug, Default)]
pub struct QualityFlags {
    pub n: usize,
    pub n_passes: usize,
    pub duty_cycle: f64,
    pub window_contaminated: bool,
    pub undersampled: bool,
    pub detrended: bool,
    pub estimator_agreement: Option<bool>,
    pub bound_snap: bool,
}

#[derive(Clone, Debug)]
pub struct Confirmation {
    pub method: MethodId,
    pub period_s: f64,
    pub score: f64,
    pub agrees: bool,
}

#[derive(Clone, Debug)]
pub struct DetrendReport {
    pub mode: DetrendMode,
    pub scale: SearchScale,
    pub columns: Vec<DetrendColumn>,
    pub coeffs: Vec<f64>,
    pub cond: f64,
    pub n_beta: usize,
    pub floating_mean_gls: bool,
    pub fallback: bool,
    pub pass_mean: Vec<f64>,
}

impl Default for DetrendReport {
    fn default() -> Self {
        Self {
            mode: DetrendMode::None,
            scale: SearchScale::Full,
            columns: Vec::new(),
            coeffs: Vec::new(),
            cond: 1.0,
            n_beta: 0,
            floating_mean_gls: false,
            fallback: false,
            pass_mean: Vec::new(),
        }
    }
}

/// Product result. `non_exhaustive` so later PRs can add fields.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PeriodicityAssessment {
    pub decision: PeriodicityDecision,
    pub period_s: Option<f64>,
    pub period_unc_s: Option<f64>,
    pub fap: Option<f64>,
    pub fap_perm: Option<f64>,
    pub fap_block: Option<f64>,
    pub fap_baluev: Option<f64>,
    pub score: f64,
    pub score_kind: ScoreKind,
    pub aliases: Vec<Alias>,
    pub quality: QualityFlags,
    pub periodogram: Periodogram,
    pub periodogram_h2: Option<Periodogram>,
    pub confirmation: Option<Confirmation>,
    pub sampling: SamplingDiagnostics,
    pub detrend: DetrendReport,
    pub method: MethodId,
    pub notes: Vec<String>,
}

impl PeriodicityAssessment {
    pub fn new(decision: PeriodicityDecision, period_s: Option<f64>) -> Self {
        Self {
            decision,
            period_s,
            period_unc_s: None,
            fap: None,
            fap_perm: None,
            fap_block: None,
            fap_baluev: None,
            score: 0.0,
            score_kind: ScoreKind::GlsPower,
            aliases: Vec::new(),
            quality: QualityFlags::default(),
            periodogram: Periodogram::default(),
            periodogram_h2: None,
            confirmation: None,
            sampling: SamplingDiagnostics::default(),
            detrend: DetrendReport::default(),
            method: MethodId::Gls,
            notes: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GlsOptions {}

#[derive(Clone, Debug)]
pub struct PdmOptions {
    pub m_bins: Option<usize>,
}

impl Default for PdmOptions {
    fn default() -> Self {
        Self { m_bins: None }
    }
}

#[derive(Clone, Debug)]
pub struct GlOptions {
    pub bin_range: (usize, usize),
}

impl Default for GlOptions {
    fn default() -> Self {
        Self { bin_range: (2, 12) }
    }
}

#[derive(Clone, Debug)]
pub struct QpGpOptions {
    pub max_obs: usize,
}

impl Default for QpGpOptions {
    fn default() -> Self {
        Self { max_obs: 200 }
    }
}

#[derive(Clone, Debug)]
pub enum ConfigError {
    EmptyMethods,
    ZeroHarmonics,
    Oversample,
    WindowRatio,
    PermMaxBudget { n_permutations: usize, min: usize },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::EmptyMethods => write!(f, "methods must be non-empty"),
            ConfigError::ZeroHarmonics => write!(f, "n_harmonics must be ≥ 1"),
            ConfigError::Oversample => write!(f, "oversample must be ≥ 1"),
            ConfigError::WindowRatio => write!(f, "window_ratio must be > 0"),
            ConfigError::PermMaxBudget {
                n_permutations,
                min,
            } => write!(
                f,
                "PermMax needs n_permutations ≥ {min} to resolve α (got {n_permutations})"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug)]
pub struct PeriodSearchConfig {
    pub min_period_s: Option<f64>,
    pub max_period_s: Option<f64>,
    pub scale: SearchScale,
    pub fap_threshold: f64,
    pub n_permutations: usize,
    pub fap_mode: FapMode,
    pub n_harmonics: Option<usize>,
    pub detrend: DetrendMode,
    pub methods: Vec<MethodId>,
    pub require_method_agreement: bool,
    pub window_veto: bool,
    pub window_ratio: f64,
    pub known_periods_s: Vec<f64>,
    pub oversample: f64,
    pub max_freq_trials: usize,
    pub n_coarse: usize,
    pub rng_seed: u64,
    pub gls: GlsOptions,
    pub pdm: PdmOptions,
    pub gregory_loredo: GlOptions,
    pub qp_gp: QpGpOptions,
}

impl PeriodSearchConfig {
    pub fn conservative() -> Self {
        Self {
            min_period_s: None,
            max_period_s: None,
            scale: SearchScale::Auto,
            fap_threshold: 1e-3,
            n_permutations: 200,
            fap_mode: FapMode::BaluevH1,
            n_harmonics: None,
            detrend: DetrendMode::Auto,
            methods: vec![MethodId::Gls],
            require_method_agreement: false,
            window_veto: true,
            window_ratio: 1.3,
            known_periods_s: Vec::new(),
            oversample: 5.0,
            max_freq_trials: 8000,
            n_coarse: 2000,
            rng_seed: 0x00C0_FFEE,
            gls: GlsOptions::default(),
            pdm: PdmOptions::default(),
            gregory_loredo: GlOptions::default(),
            qp_gp: QpGpOptions::default(),
        }
    }

    pub fn sensitive() -> Self {
        let mut c = Self::conservative();
        c.fap_threshold = 1e-2;
        c.require_method_agreement = false;
        c.window_ratio = 1.1;
        c
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.methods.is_empty() {
            return Err(ConfigError::EmptyMethods);
        }
        if self.n_harmonics == Some(0) {
            return Err(ConfigError::ZeroHarmonics);
        }
        if !(self.oversample >= 1.0) {
            return Err(ConfigError::Oversample);
        }
        if !(self.window_ratio > 0.0) {
            return Err(ConfigError::WindowRatio);
        }
        if self.fap_mode == FapMode::PermMax {
            let min = ((1.0 / self.fap_threshold).ceil() as usize).saturating_sub(1);
            if self.n_permutations < min {
                return Err(ConfigError::PermMaxBudget {
                    n_permutations: self.n_permutations,
                    min,
                });
            }
        }
        Ok(())
    }
}

impl Default for PeriodSearchConfig {
    fn default() -> Self {
        Self::conservative()
    }
}
