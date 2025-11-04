// Functions to normalize vismag data

// normalize vismag by range
pub fn normalize_to_range(vismag: f64, obs_range: f64, std_range: f64) -> f64 {
    // range correction: 5log10(obs_range / std_range)
    let range_correction = 5.0 * (obs_range / std_range).log10();
    vismag - range_correction
}

// normalize vismag by phase
pub fn normalize_to_phase(vismag: f64, obs_phase: f64, std_phase: f64) -> f64 {
    // phase correction relative to lambertian sphere
    let phase_factor_obs = (1.0 + obs_phase.cos()) / 2.0;
    let phase_factor_std = (1.0 + std_phase.cos()) / 2.0;
    let phase_correction = -2.5 * (phase_factor_obs / phase_factor_std).log10();
    vismag - phase_correction
}

// normalize vismag by both range and phase
pub fn normalize_vismag(
    vismag: f64,
    obs_range: f64,
    std_range: f64,
    obs_phase: f64,
    std_phase: f64,
) -> f64 {
    let vismag_range_normalized = normalize_to_range(vismag, obs_range, std_range);
    normalize_to_phase(vismag_range_normalized, obs_phase, std_phase)
}

// bulk normalization of vismag data
pub fn bulk_normalize_vismag(
    vismags: &[f64],
    obs_ranges: &[f64],
    std_range: f64,
    obs_phases: &[f64],
    std_phase: f64,
) -> Vec<f64> {
    // TODO: optimize to avoid redundant trig calcs on std values
    vismags
        .iter()
        .zip(obs_ranges.iter())
        .zip(obs_phases.iter())
        .map(|((&vismag, &obs_range), &obs_phase)| {
            normalize_vismag(vismag, obs_range, std_range, obs_phase, std_phase)
        })
        .collect()
}