//! Baluev 2008 FAP and local 0-beat permutation.

use crate::entities::assessment::ScoreKind;
use crate::functions::periodicity::gls::gls_power_zero_mean;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// \(z = (\nu/2)\, p_1 / (1-p_1)\) — Baluev 2008 eqs. 5–6.
pub fn p1_to_z(p1: f64, nu: f64) -> f64 {
    let p = p1.clamp(0.0, 1.0 - 1e-15);
    if nu <= 0.0 {
        return 0.0;
    }
    (nu / 2.0) * p / (1.0 - p)
}

/// Single-trial survival \(P_1 = (1-p_1)^{\nu/2}\) — Baluev eq. 7.
pub fn p1_survival(p1: f64, nu: f64) -> f64 {
    let p = p1.clamp(0.0, 1.0 - 1e-15);
    if nu <= 0.0 {
        return 1.0;
    }
    (1.0 - p).powf(nu / 2.0)
}

/// Effective baseline \(T_\text{eff} = \sqrt{4\pi \sigma_t^2}\) — Baluev eq. 31.
pub fn teff(t: &[f64], w: &[f64]) -> f64 {
    let wsum: f64 = w.iter().sum();
    if t.len() < 2 || wsum <= 0.0 {
        return 0.0;
    }
    let tbar: f64 = t.iter().zip(w.iter()).map(|(ti, wi)| wi * ti).sum::<f64>() / wsum;
    let var: f64 = t
        .iter()
        .zip(w.iter())
        .map(|(ti, wi)| wi * (ti - tbar) * (ti - tbar))
        .sum::<f64>()
        / wsum;
    (4.0 * std::f64::consts::PI * var.max(0.0)).sqrt()
}

/// Expected upcrossings \(\tau\) — Baluev eq. 8.
pub fn tau_upcrossing(z: f64, p1: f64, nu: f64, w_baluev: f64) -> f64 {
    if z <= 0.0 || nu <= 1.0 || w_baluev <= 0.0 {
        return 0.0;
    }
    let p = p1.clamp(0.0, 1.0 - 1e-15);
    w_baluev * (1.0 - p).powf((nu - 1.0) / 2.0) * z.sqrt()
}

/// Alias-free FAP of the maximum \(p_1\) — Baluev eq. 6.
pub fn fap_baluev(p1_max: f64, nu: f64, teff_s: f64, f_max_hz: f64) -> f64 {
    if nu < 2.0 || !p1_max.is_finite() {
        return 1.0;
    }
    let z = p1_to_z(p1_max, nu);
    let p1 = p1_survival(p1_max, nu);
    let w = f_max_hz.max(0.0) * teff_s.max(0.0);
    let tau = tau_upcrossing(z, p1_max, nu, w);
    (1.0 - (1.0 - p1) * (-tau).exp()).clamp(0.0, 1.0)
}

pub fn perm_seed(rng_seed: u64, i: u64) -> u64 {
    rng_seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

pub fn beats(kind: ScoreKind, null_score: f64, data_score: f64) -> bool {
    match kind {
        ScoreKind::GlsPower | ScoreKind::LogOdds => null_score >= data_score,
        ScoreKind::PdmTheta | ScoreKind::StringLengthRatio => null_score <= data_score,
        ScoreKind::SpectralWindow => null_score >= data_score,
    }
}

/// Shuffle `y` (and optional per-point σ) at fixed `t`. Returns n_beat.
pub fn local_zero_beat(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    periods: &[f64],
    data_score: f64,
    n_harmonics: usize,
    n_perm: usize,
    rng_seed: u64,
) -> usize {
    if n_perm == 0 || y.len() < 3 {
        return 0;
    }
    let mut n_beat = 0usize;
    for i in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(perm_seed(rng_seed, i as u64));
        let mut y_star = y.to_vec();
        y_star.shuffle(&mut rng);
        let mut w_star = w.to_vec();
        // Joint shuffle of weights would require shuffling indices; for Unknown
        // all w=1 so this is a no-op. For known σ we shuffle in lockstep:
        if w.iter().any(|wi| (*wi - w[0]).abs() > 1e-15) {
            let mut idx: Vec<usize> = (0..y.len()).collect();
            idx.shuffle(&mut rng);
            y_star = idx.iter().map(|&j| y[j]).collect();
            w_star = idx.iter().map(|&j| w[j]).collect();
        }
        let null = periods
            .iter()
            .map(|&p| gls_power_zero_mean(t, &y_star, &w_star, p, n_harmonics))
            .fold(0.0_f64, f64::max);
        if beats(ScoreKind::GlsPower, null, data_score) {
            n_beat += 1;
        }
    }
    n_beat
}

pub fn fap_from_beats(n_beat: usize, b: usize) -> f64 {
    (n_beat + 1) as f64 / (b + 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_bal1_p1_half_even_sampling() {
        // N=64, t=0..63, w=1, p_1=0.5 forced, floating-mean, n_β=0, ν=61, f_max=0.4
        let n = 64usize;
        let nu = (n - 3) as f64; // 61
        assert!((nu - 61.0).abs() < 1e-12);
        let p1 = 0.5;
        let z = p1_to_z(p1, nu);
        assert!((z - 30.5).abs() / 30.5 < 1e-6, "z={z}");
        let p1s = p1_survival(p1, nu);
        let p1s_ref = 0.5_f64.powf(30.5);
        assert!((p1s - p1s_ref).abs() / p1s_ref < 1e-6);

        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let w = vec![1.0; n];
        let te = teff(&t, &w);
        // σ_t² = 341.25, T_eff = √(4π · 341.25)
        let te_ref = (4.0 * std::f64::consts::PI * 341.25).sqrt();
        assert!(
            (te - te_ref).abs() / te_ref < 1e-6,
            "T_eff={te} ref={te_ref}"
        );

        let fap = fap_baluev(p1, nu, te, 0.4);
        let wbal = 0.4 * te_ref;
        let tau = tau_upcrossing(30.5, p1, nu, wbal);
        let fap_ref = 1.0 - (1.0 - p1s_ref) * (-tau).exp();
        assert!(
            (fap - fap_ref).abs() / fap_ref.max(1e-30) < 1e-3,
            "FAP={fap} ref={fap_ref}"
        );

        let t_in: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let w_in = vec![1.0; 16];
        let te_in = teff(&t_in, &w_in);
        assert!(
            te_in < te,
            "Intra T_eff must be smaller than campaign T_eff ({te_in} vs {te})"
        );
    }
}
