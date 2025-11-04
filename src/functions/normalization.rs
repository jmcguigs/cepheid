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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vismag_to_standard_conditions() {
        // Use data taken from MMT9 (http://mmt.favor2.info/satellites) to verify
        // Test case: normalize vismag 9.14874 at range 37473.121 km and phase 81.836 deg
        // to standard conditions: range 1000 km and phase 0 deg
        // Expected result: 1.422 standard magnitude
        let vismag = 9.14874; // observed visual magnitude
        let obs_range = 37473.121e3; // km - GEO satellite
        let std_range = 1000.0e3; // 1Mm std range for MMT9 data
        let obs_phase = 81.836_f64.to_radians(); // convert degrees to radians
        let std_phase = 90.0_f64.to_radians(); // half-illuminated standard phase

        let result = normalize_vismag(vismag, obs_range, std_range, obs_phase, std_phase);

        // Allow small tolerance for uncertainties in range and phase estimates
        let expected = 1.422;
        let tolerance = 0.01;
        assert!(
            (result - expected).abs() < tolerance,
            "Expected {}, got {}",
            expected,
            result
        );
    }
}