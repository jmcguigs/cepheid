pub fn vismag_from_relative_flux(reference_vismag: f64, reference_flux: f64, target_flux: f64) -> f64 {
    // vismag = reference_vismag - 2.5 * log10(target_flux / reference_flux)
    reference_vismag - 2.5 * (target_flux / reference_flux).log10()
}