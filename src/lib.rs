pub mod functions;
pub mod constants;
pub mod entities;

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

        let result = functions::normalization::normalize_vismag(vismag, obs_range, std_range, obs_phase, std_phase);

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
