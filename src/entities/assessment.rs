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
