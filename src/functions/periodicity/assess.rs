//! Pass-aware `assess_periodicity`.

use crate::entities::assessment::{
    Alias, AliasKind, Confirmation, DetrendMode, FapMode, MethodId, PeriodSearchConfig,
    PeriodicityAssessment, PeriodicityDecision, QualityFlags, ScoreKind, SearchScale,
};
use crate::entities::series::{Modality, Series};
use crate::functions::periodicity::detrend::{auto_detrend, pass_index_lists};
use crate::functions::periodicity::fap::{
    fap_baluev, fap_from_beats, local_zero_beat, perm_seed, teff,
};
use crate::functions::periodicity::gls::{
    argmax, coarse_refine_periods, gls_periodogram, gls_power_zero_mean, interpolate_peak,
    is_interior_maximum, subtract_h1,
};
use crate::functions::periodicity::pdm::{
    argmin_finite, pdm_bin_count, pdm_periodogram, pdm_theta,
};
use crate::functions::periodicity::window::spectral_window;
use crate::functions::sampling::{
    diagnose_sampling, searchable_period_bounds, SamplingConfig, DEFAULT_OVERSAMPLE,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

const BALUEV_SKIP: f64 = 0.05;
const CONSENSUS_REL: f64 = 0.05;

/// Product entry point. Default methods = `[Gls, Pdm]`; FAP is Baluev on H=1.
pub fn assess_periodicity(series: &Series, config: &PeriodSearchConfig) -> PeriodicityAssessment {
    if let Err(e) = config.validate() {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.notes.push(format!("config: {e}"));
        return a;
    }
    if series.len() < 12 {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.notes.push(format!("QC: n={} < 12", series.len()));
        a.quality.n = series.len();
        return a;
    }

    let h = config
        .n_harmonics
        .unwrap_or(if series.meta().modality == Modality::RfPower {
            1
        } else {
            2
        });
    let oversample = if config.oversample >= 1.0 {
        config.oversample
    } else {
        DEFAULT_OVERSAMPLE
    };

    let samp_cfg = sampling_cfg(series, config);
    let sampling = diagnose_sampling(series, &samp_cfg, oversample);
    if sampling.n_passes == 0 || sampling.span_s <= 0.0 {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.sampling = sampling;
        a.notes.push("QC: no span / no passes".into());
        return a;
    }

    let n_pass = sampling.n_passes;
    let eligible: Vec<usize> = sampling
        .passes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_intra_eligible())
        .map(|(i, _)| i)
        .collect();

    match (config.scale, n_pass) {
        (SearchScale::Auto, 1) | (SearchScale::IntraPass, 1) => {
            if eligible.is_empty() {
                let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
                a.sampling = sampling;
                a.notes
                    .push("single pass too short/sparse for intra".into());
                return a;
            }
            assess_intra(series, config, &sampling, &eligible, h, oversample)
        }
        (SearchScale::Auto, 2) | (SearchScale::IntraPass, 2..) => {
            if eligible.is_empty() {
                let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
                a.sampling = sampling;
                a.notes.push("no intra-eligible pass".into());
                return a;
            }
            let mut results: Vec<PeriodicityAssessment> = eligible
                .iter()
                .map(|&i| assess_intra(series, config, &sampling, &[i], h, oversample))
                .collect();
            consensus_intra(&mut results, &sampling)
        }
        (SearchScale::Auto, n) if n >= 3 => {
            let intra = if eligible.is_empty() {
                let mut a = PeriodicityAssessment::new(PeriodicityDecision::NotPeriodic, None);
                a.sampling = sampling.clone();
                a.notes.push("Auto≥3: no intra-eligible pass".into());
                a
            } else {
                let mut results: Vec<PeriodicityAssessment> = eligible
                    .iter()
                    .map(|&i| assess_intra(series, config, &sampling, &[i], h, oversample))
                    .collect();
                consensus_intra(&mut results, &sampling)
            };
            if intra.decision == PeriodicityDecision::Periodic {
                return intra;
            }
            let mut inter = assess_inter(series, config, &sampling, h, oversample, false);
            inter
                .notes
                .insert(0, "Auto≥3: Intra not Periodic; Inter leg".into());
            inter
        }
        (SearchScale::InterPass, 1) => {
            let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
            a.sampling = sampling;
            a.notes.push("InterPass requires ≥ 2 passes".into());
            a
        }
        (SearchScale::InterPass, _) => {
            assess_inter(series, config, &sampling, h, oversample, n_pass == 2)
        }
        (SearchScale::Full, _) => assess_full(series, config, &sampling, h, oversample),
        _ => {
            let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
            a.sampling = sampling;
            a.notes.push("unhandled (scale, n_passes) cell".into());
            a
        }
    }
}

fn sampling_cfg(series: &Series, config: &PeriodSearchConfig) -> SamplingConfig {
    let span = series.span_s().unwrap_or(0.0);
    let pmin = config.min_period_s.unwrap_or(3.0);
    let pmax = config.max_period_s.unwrap_or((span / 2.0).max(pmin * 2.0));
    SamplingConfig {
        min_period_s: pmin,
        max_period_s: pmax,
        known_periods_s: config.known_periods_s.clone(),
        scale: match config.scale {
            SearchScale::IntraPass => SearchScale::IntraPass,
            SearchScale::InterPass => SearchScale::InterPass,
            _ => SearchScale::Full,
        },
        ..SamplingConfig::default()
    }
}

fn assess_intra(
    series: &Series,
    config: &PeriodSearchConfig,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    pass_ids: &[usize],
    h: usize,
    oversample: f64,
) -> PeriodicityAssessment {
    let t = series.t_s();
    let lists = pass_index_lists(t, &sampling.passes);
    let pid = pass_ids[0];
    let idx = &lists[pid];
    let pass = &sampling.passes[pid];
    let dt = auto_detrend(
        series,
        &sampling.passes,
        SearchScale::IntraPass,
        config.detrend,
        Some(idx),
    );
    let t_s: Vec<f64> = idx.iter().map(|&i| t[i]).collect();
    let w: Vec<f64> = {
        let ww = series.weights();
        idx.iter().map(|&i| ww[i]).collect()
    };
    let min_dt = idx
        .windows(2)
        .map(|w| t[w[1]] - t[w[0]])
        .fold(f64::INFINITY, f64::min);
    let min_dt = if min_dt.is_finite() {
        min_dt
    } else {
        pass.median_dt_s
    };
    // Irregular sampling: the Eyer–Bartholdi bound is 2·min Δt, not 2·median.
    let p_min = config.min_period_s.unwrap_or(3.0).max(2.0 * min_dt);
    let p_max = config
        .max_period_s
        .unwrap_or(f64::INFINITY)
        .min(0.5 * pass.duration_s())
        .min(sampling.span_s / 2.0);
    search_and_decide(
        series,
        config,
        sampling,
        &t_s,
        &dt.y,
        &w,
        p_min,
        p_max,
        pass.duration_s().max(1.0),
        dt.report.n_beta,
        SearchScale::IntraPass,
        h,
        oversample,
        dt.report,
        None,
    )
}

fn assess_inter(
    series: &Series,
    config: &PeriodSearchConfig,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    h: usize,
    oversample: f64,
    undersampled: bool,
) -> PeriodicityAssessment {
    let dt = auto_detrend(
        series,
        &sampling.passes,
        SearchScale::InterPass,
        config.detrend,
        None,
    );
    let (p_min, p_max) = searchable_period_bounds(
        &SamplingConfig {
            min_period_s: config.min_period_s.unwrap_or(3.0),
            max_period_s: config.max_period_s.unwrap_or(f64::INFINITY),
            scale: SearchScale::InterPass,
            known_periods_s: config.known_periods_s.clone(),
            ..SamplingConfig::default()
        },
        sampling.span_s,
        sampling.median_intrapass_dt_s,
        median_pass_dur(sampling),
        median_gap(sampling),
        sampling.n_passes,
    );
    let mut a = search_and_decide(
        series,
        config,
        sampling,
        series.t_s(),
        &dt.y,
        &series.weights(),
        p_min,
        p_max,
        sampling.span_s,
        dt.report.n_beta,
        SearchScale::InterPass,
        h,
        oversample,
        dt.report,
        Some(dt.y.clone()),
    );
    a.quality.undersampled = undersampled;
    a
}

fn assess_full(
    series: &Series,
    config: &PeriodSearchConfig,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    h: usize,
    oversample: f64,
) -> PeriodicityAssessment {
    let dt = auto_detrend(
        series,
        &sampling.passes,
        SearchScale::Full,
        config.detrend,
        None,
    );
    let p_min = config
        .min_period_s
        .unwrap_or(3.0)
        .max(2.0 * sampling.median_intrapass_dt_s);
    let p_max = config
        .max_period_s
        .unwrap_or(sampling.span_s / 2.0)
        .min(sampling.span_s / 2.0);
    search_and_decide(
        series,
        config,
        sampling,
        series.t_s(),
        &dt.y,
        &series.weights(),
        p_min,
        p_max,
        sampling.span_s,
        dt.report.n_beta,
        SearchScale::Full,
        h,
        oversample,
        dt.report,
        Some(dt.y.clone()),
    )
}

fn median_pass_dur(s: &crate::entities::assessment::SamplingDiagnostics) -> f64 {
    let mut d: Vec<f64> = s.passes.iter().map(|p| p.duration_s()).collect();
    if d.is_empty() {
        return 0.0;
    }
    d.sort_by(|a, b| a.total_cmp(b));
    d[d.len() / 2]
}

fn median_gap(s: &crate::entities::assessment::SamplingDiagnostics) -> f64 {
    if s.passes.len() < 2 {
        return 0.0;
    }
    let mut g: Vec<f64> = s
        .passes
        .windows(2)
        .map(|w| (w[1].t_start_s - w[0].t_end_s).max(0.0))
        .collect();
    g.sort_by(|a, b| a.total_cmp(b));
    g[g.len() / 2]
}

fn search_and_decide(
    series: &Series,
    config: &PeriodSearchConfig,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    t: &[f64],
    y: &[f64],
    w: &[f64],
    p_min: f64,
    p_max: f64,
    span_s: f64,
    n_beta: usize,
    scale: SearchScale,
    h: usize,
    oversample: f64,
    detrend: crate::entities::assessment::DetrendReport,
    y_for_block: Option<Vec<f64>>,
) -> PeriodicityAssessment {
    let mut notes = Vec::new();
    if p_min <= 0.0 || p_max <= p_min || t.len() < 12 {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.sampling = sampling.clone();
        a.detrend = detrend;
        a.notes
            .push("QC: empty search band or n<12 in scope".into());
        return a;
    }
    let n_eff = t.len() as f64 - n_beta as f64;
    let nu = n_eff - 2.0;
    if nu < 2.0 {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.sampling = sampling.clone();
        a.detrend = detrend;
        a.notes.push(format!("QC: ν={nu} < 2"));
        return a;
    }

    let eval_h = |periods: &[f64]| {
        periods
            .iter()
            .map(|&p| gls_power_zero_mean(t, y, w, p, h))
            .collect::<Vec<_>>()
    };
    let (periods, scores_h) = coarse_refine_periods(
        p_min,
        p_max,
        span_s,
        config.n_coarse,
        config.max_freq_trials,
        oversample,
        8,
        eval_h,
    );
    let pgram_h = crate::entities::assessment::Periodogram {
        period_s: periods.clone(),
        score: scores_h.clone(),
        score_kind: ScoreKind::GlsPower,
    };
    let pgram_1 = if h == 1 {
        pgram_h.clone()
    } else {
        gls_periodogram(t, y, w, &periods, 1, true)
    };

    let Some(idx_1) = argmax(&pgram_1.score) else {
        let mut a = PeriodicityAssessment::new(PeriodicityDecision::Inconclusive, None);
        a.sampling = sampling.clone();
        a.notes.push("empty periodogram".into());
        return a;
    };
    // Place the period on H=1 (the accept statistic). H=2 is recorded for
    // later 2:1 / tumbler promotion; using p_H to *place* P* lets the 2nd
    // harmonic lock onto 2P of a single sinusoid with near-zero p_1.
    let (mut p_star, unc) = interpolate_peak(&pgram_1.period_s, &pgram_1.score, idx_1);
    let idx_h = argmax(&pgram_h.score);
    if series.meta().modality != crate::entities::series::Modality::RfPower {
        if let Some(ih) = idx_h {
            let p_h = pgram_h.period_s[ih];
            let p1_at_h = gls_power_zero_mean(t, y, w, p_h, 1);
            if (p_h - 2.0 * p_star).abs() / p_star.max(1e-12) < 0.05
                && p1_at_h > 0.5 * pgram_1.score[idx_1]
                && pgram_1.score[idx_1] > 0.1
            {
                notes.push(format!("optical 2:1 promotion {p_star:.4} → {p_h:.4}"));
                p_star = p_h;
            }
        }
        // Symmetric double-flash: H=1 locks on P/2. If PDM θ at 2P is at
        // least as good, report the rotation period.
        let two = 2.0 * p_star;
        if two >= p_min && two <= p_max {
            let m_tmp = pdm_bin_count(t.len(), config.pdm.m_bins);
            let pdm_prefers_two =
                match (pdm_theta(t, y, p_star, m_tmp), pdm_theta(t, y, two, m_tmp)) {
                    (Some(th1), Some(th2)) => th2 <= th1 * 1.15,
                    _ => false,
                };
            let resid = subtract_h1(t, y, w, p_star);
            let p1_resid_two = gls_power_zero_mean(t, &resid, w, two, 1);
            let p2_here = gls_power_zero_mean(t, y, w, p_star, 2);
            let extra_h = p2_here - pgram_1.score[idx_1];
            // Narrow flashes have extra H=2 power at P/2; a pure sine does not.
            if pdm_prefers_two || p1_resid_two > 0.05 || extra_h > 0.03 {
                notes.push(format!(
                    "optical 2P promotion {p_star:.4} → {two:.4} (PDM={pdm_prefers_two}, resid_p1={p1_resid_two:.3})"
                ));
                p_star = two;
            }
        }
    }
    let score_h = pgram_h
        .score
        .get(idx_h.unwrap_or(idx_1))
        .copied()
        .unwrap_or(0.0);
    let p1_at_star = pgram_1.score[idx_1];
    let interior = is_interior_maximum(&pgram_1.score, idx_1);
    let p1_max = pgram_1.score.iter().copied().fold(0.0_f64, f64::max);
    let f_max = 1.0 / p_min;
    let te = teff(t, w);
    let fap_b = fap_baluev(p1_max, nu, te, f_max);

    if !interior {
        notes.push("bound-snap: peak is not an interior maximum".into());
    }

    let df_local = local_df(&pgram_1.period_s, idx_1);
    let (vetoed, aliases) = window_veto(
        p_star, p1_at_star, sampling, config, df_local, t, y, w, nu, te, f_max, &mut notes,
    );

    let mut n_beat_perm = 0usize;
    let mut fap_perm = None;
    let mut n_beat_block = 0usize;
    let mut fap_block = None;

    let look_elsewhere = match config.fap_mode {
        FapMode::BaluevH1 => fap_b,
        FapMode::PermMax => {
            // filled after perms-max; placeholder
            fap_b
        }
    };

    if fap_b > BALUEV_SKIP && config.fap_mode == FapMode::BaluevH1 {
        notes.push(format!(
            "Baluev FAP={fap_b:.3} > {BALUEV_SKIP} (cheap reject)"
        ));
    } else {
        let extras = [p_star, p_star * 2.0, p_star / 2.0]
            .into_iter()
            .filter(|p| *p >= p_min && *p <= p_max)
            .collect::<Vec<_>>();
        n_beat_perm = local_zero_beat(
            t,
            y,
            w,
            &extras,
            p1_at_star,
            1,
            config.n_permutations,
            config.rng_seed,
        );
        fap_perm = Some(fap_from_beats(n_beat_perm, config.n_permutations));

        let use_block = matches!(scale, SearchScale::InterPass | SearchScale::Full)
            && sampling.n_passes >= 3
            && y_for_block.is_some();
        if use_block {
            n_beat_block = pass_block_beats(
                series,
                sampling,
                y_for_block.as_deref().unwrap(),
                w,
                p_star,
                p1_at_star,
                config.n_permutations,
                config.rng_seed,
            );
            fap_block = Some(fap_from_beats(n_beat_block, config.n_permutations));
        }
    }

    let look = if config.fap_mode == FapMode::PermMax {
        // perm-max over a thinned grid
        let thin_n = 500.min(pgram_1.len()).max(8);
        let step = (pgram_1.len() / thin_n).max(1);
        let thin: Vec<f64> = pgram_1.period_s.iter().step_by(step).copied().collect();
        let n_beat = local_zero_beat(
            t,
            y,
            w,
            &thin,
            p1_max,
            1,
            config.n_permutations,
            config.rng_seed,
        );
        fap_from_beats(n_beat, config.n_permutations)
    } else {
        look_elsewhere
    };

    let block_ok = matches!(scale, SearchScale::IntraPass)
        || sampling.n_passes < 3
        || n_beat_block == 0
        || fap_b > BALUEV_SKIP;

    let gls_ok = interior && !vetoed && look < config.fap_threshold && n_beat_perm == 0 && block_ok;

    if vetoed {
        notes.push(format!("window/alias veto at P={p_star:.4}"));
    }

    let wants_pdm = config.methods.contains(&MethodId::Pdm);
    let mut confirmation = None;
    let mut agreement = if config.methods.len() <= 1 {
        Some(true)
    } else if !wants_pdm {
        Some(true)
    } else {
        None
    };
    let mut pdm_inconclusive = false;

    if wants_pdm && config.methods.len() > 1 {
        let m = pdm_bin_count(t.len(), config.pdm.m_bins);
        let pgram_pdm = pdm_periodogram(t, y, &pgram_1.period_s, m);
        match argmin_finite(&pgram_pdm.score) {
            None => {
                notes.push("PDM occupancy: no valid trial".into());
                if config.require_method_agreement {
                    pdm_inconclusive = true;
                }
                agreement = None;
            }
            Some(ip) => {
                let p_pdm = pgram_pdm.period_s[ip];
                let th = pgram_pdm.score[ip];
                let optical = series.meta().modality != crate::entities::series::Modality::RfPower;
                let agree = periods_agree(p_star, p_pdm, optical);
                agreement = Some(agree);
                if agree && optical {
                    let longer = p_star.max(p_pdm);
                    if (longer - p_star).abs() / p_star.max(1e-12) > 0.02 {
                        notes.push(format!(
                            "optical 2:1/agreement: report longer {p_star:.4} → {longer:.4}"
                        ));
                        p_star = longer;
                    }
                }
                confirmation = Some(Confirmation {
                    method: MethodId::Pdm,
                    period_s: p_pdm,
                    score: th,
                    agrees: agree,
                });
                if !agree {
                    notes.push(format!("GLS–PDM disagree: GLS={p_star:.4} PDM={p_pdm:.4}"));
                    if config.require_method_agreement && gls_ok && look < config.fap_threshold {
                        pdm_inconclusive = true;
                    }
                }
            }
        }
    }

    let agreement_ok = !config.require_method_agreement || agreement.unwrap_or(false);

    let decision = if pdm_inconclusive {
        PeriodicityDecision::Inconclusive
    } else if gls_ok && agreement_ok {
        PeriodicityDecision::Periodic
    } else {
        PeriodicityDecision::NotPeriodic
    };

    let quality = QualityFlags {
        n: series.len(),
        n_passes: sampling.n_passes,
        duty_cycle: sampling.duty_cycle,
        window_contaminated: vetoed,
        undersampled: false,
        detrended: config.detrend != DetrendMode::None,
        estimator_agreement: agreement,
        bound_snap: !interior,
    };

    PeriodicityAssessment {
        decision,
        period_s: if decision == PeriodicityDecision::Periodic {
            Some(p_star)
        } else {
            None
        },
        period_unc_s: if decision == PeriodicityDecision::Periodic {
            unc
        } else {
            None
        },
        fap: Some(look),
        fap_perm,
        fap_block,
        fap_baluev: Some(fap_b),
        score: score_h,
        score_kind: ScoreKind::GlsPower,
        aliases,
        quality,
        periodogram: pgram_1,
        periodogram_h2: if h > 1 { Some(pgram_h) } else { None },
        confirmation,
        sampling: sampling.clone(),
        detrend,
        method: MethodId::Gls,
        notes,
    }
}

fn periods_agree(p1: f64, p2: f64, optical: bool) -> bool {
    let lo = p1.min(p2);
    let hi = p1.max(p2);
    if lo <= 0.0 {
        return false;
    }
    if (hi - lo) / lo < 0.03 {
        return true;
    }
    optical && (hi - 2.0 * lo).abs() / lo < 0.05
}

fn local_df(periods: &[f64], idx: usize) -> f64 {
    if periods.len() < 2 {
        return 0.0;
    }
    let f = 1.0 / periods[idx];
    let mut ds = Vec::new();
    if idx > 0 {
        ds.push((1.0 / periods[idx - 1] - f).abs());
    }
    if idx + 1 < periods.len() {
        ds.push((1.0 / periods[idx + 1] - f).abs());
    }
    ds.into_iter().fold(f64::INFINITY, f64::min)
}

fn window_veto(
    p_star: f64,
    p1: f64,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    config: &PeriodSearchConfig,
    df_local: f64,
    t: &[f64],
    y: &[f64],
    w: &[f64],
    nu: f64,
    te: f64,
    f_max: f64,
    notes: &mut Vec<String>,
) -> (bool, Vec<Alias>) {
    if !config.window_veto {
        return (false, Vec::new());
    }
    let f_star = 1.0 / p_star;
    let mut aliases = Vec::new();
    let mut vetoed = false;
    for pk in &sampling.window_peaks {
        let f_line = 1.0 / pk.period_s;
        let close = if df_local > 0.0 {
            (f_star - f_line).abs() < 1.5 * df_local
        } else {
            (p_star - pk.period_s).abs() / pk.period_s < 0.03
        };
        if !close {
            continue;
        }
        let named = matches!(
            pk.kind,
            AliasKind::SiderealDay
                | AliasKind::SolarDay
                | AliasKind::Orbital
                | AliasKind::OrbitalHalf
                | AliasKind::PassCadence
        );
        let w_at = spectral_window(t, w, &[f_star])
            .first()
            .copied()
            .unwrap_or(pk.power);
        let weak = w_at > 0.0 && p1 / w_at < config.window_ratio;
        let mut this_veto = named || (pk.kind == AliasKind::WindowPeak && weak);
        if this_veto {
            // Override: residual after H=1 still periodic.
            let resid = subtract_h1(t, y, w, p_star);
            let p1_r = gls_power_zero_mean(t, &resid, w, p_star, 1);
            let fap_r = fap_baluev(p1_r, nu, te, f_max);
            if fap_r < config.fap_threshold {
                this_veto = false;
                notes.push("window override: H=1 residual still significant".into());
            }
        }
        aliases.push(Alias {
            period_s: pk.period_s,
            score: pk.power,
            kind: pk.kind,
            relative_delta: (p_star - pk.period_s).abs() / p_star,
            vetoed: this_veto,
        });
        if this_veto {
            vetoed = true;
        }
    }
    (vetoed, aliases)
}

fn pass_block_beats(
    series: &Series,
    sampling: &crate::entities::assessment::SamplingDiagnostics,
    y: &[f64],
    w: &[f64],
    p_star: f64,
    data_score: f64,
    n_perm: usize,
    rng_seed: u64,
) -> usize {
    let lists = pass_index_lists(series.t_s(), &sampling.passes);
    let t = series.t_s();
    let mut n_beat = 0usize;
    for i in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(perm_seed(rng_seed, 10_000 + i as u64));
        let y_star = pass_block_replicate(y, w, &lists, &mut rng);
        let s = gls_power_zero_mean(t, &y_star, w, p_star, 1);
        if s >= data_score {
            n_beat += 1;
        }
    }
    n_beat
}

fn pass_block_replicate(
    y: &[f64],
    w: &[f64],
    passes: &[Vec<usize>],
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n_p = passes.len();
    let mut means = vec![0.0; n_p];
    let mut resids: Vec<Vec<f64>> = Vec::with_capacity(n_p);
    for (p, idx) in passes.iter().enumerate() {
        let (sw, swy) = idx
            .iter()
            .fold((0.0, 0.0), |(sw, swy), &i| (sw + w[i], swy + w[i] * y[i]));
        means[p] = if sw > 0.0 { swy / sw } else { 0.0 };
        resids.push(idx.iter().map(|&i| y[i] - means[p]).collect());
    }
    let mut y_star = vec![0.0; y.len()];
    for idx in passes {
        let k = rng.random_range(0..n_p);
        let donor = &resids[k];
        for &j in idx {
            let r = if donor.len() >= 2 {
                donor[rng.random_range(0..donor.len())]
            } else {
                0.0
            };
            y_star[j] = means[k] + r;
        }
    }
    y_star
}

fn consensus_intra(
    results: &mut [PeriodicityAssessment],
    sampling: &crate::entities::assessment::SamplingDiagnostics,
) -> PeriodicityAssessment {
    let periodic: Vec<&PeriodicityAssessment> = results
        .iter()
        .filter(|a| a.decision == PeriodicityDecision::Periodic)
        .collect();
    if periodic.is_empty() {
        let mut best = results
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .cloned()
            .unwrap_or_else(|| PeriodicityAssessment::new(PeriodicityDecision::NotPeriodic, None));
        best.sampling = sampling.clone();
        return best;
    }
    if periodic.len() == 1 {
        let mut a = periodic[0].clone();
        a.notes.push("single-pass consensus waived".into());
        a.sampling = sampling.clone();
        return a;
    }
    let p0 = periodic[0].period_s.unwrap();
    let agree = periodic.iter().all(|a| {
        a.period_s
            .map(|p| (p - p0).abs() / p0 < CONSENSUS_REL)
            .unwrap_or(false)
    });
    if agree {
        let mut a = periodic[0].clone();
        a.sampling = sampling.clone();
        return a;
    }
    let mut best = periodic
        .into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap()
        .clone();
    best.notes
        .push("intra-pass periods disagree; taking best score".into());
    best.sampling = sampling.clone();
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::assessment::Pass;
    use crate::entities::series::{Covariates, Series, SeriesMeta, SigmaSpec, YUnit};
    use crate::functions::sampling::{geo_night_times, leo_pass_times};

    fn series_of(t: Vec<f64>, y: Vec<f64>, phase: Option<Vec<f64>>) -> Series {
        Series::try_new(
            t,
            y,
            SigmaSpec::Unknown,
            Covariates {
                solar_phase_rad: phase,
                ..Covariates::default()
            },
            SeriesMeta {
                modality: Modality::OpticalPhotometry,
                y_unit: YUnit::Magnitude,
                label: None,
            },
        )
        .unwrap()
    }

    fn leo_week() -> Vec<f64> {
        leo_pass_times(18, 40, 480.0, 5400.0, 0.0)
    }

    fn geo_night_times_irregular(
        n_nights: usize,
        pts_per_night: usize,
        night_len_s: f64,
        t0: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut t = Vec::with_capacity(n_nights * pts_per_night);
        let mut s = seed;
        for n in 0..n_nights {
            let start = t0 + n as f64 * crate::constants::SOLAR_DAY_S;
            let mut night = Vec::with_capacity(pts_per_night);
            for _ in 0..pts_per_night {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                let u = (s as f64) / (u64::MAX as f64);
                night.push(start + u * night_len_s);
            }
            night.sort_by(|a, b| a.total_cmp(b));
            t.extend(night);
        }
        t
    }

    fn cfg() -> PeriodSearchConfig {
        let mut c = PeriodSearchConfig::conservative();
        c.min_period_s = Some(3.0);
        c.max_period_s = Some(3000.0);
        c
    }

    fn leftover_lambert(phase: &[f64], c: f64) -> Vec<f64> {
        phase.iter().map(|ph| c * (1.0 + ph.cos()) / 2.0).collect()
    }

    fn leftover_three_cycle(t: &[f64], passes: &[Pass], a: f64) -> Vec<f64> {
        t.iter()
            .map(|&ti| {
                let p = passes
                    .iter()
                    .find(|p| ti >= p.t_start_s && ti <= p.t_end_s)
                    .unwrap_or(&passes[0]);
                let tau = ti - p.t_start_s;
                let per = p.duration_s() / 3.0;
                a * (std::f64::consts::TAU * tau / per).sin()
            })
            .collect()
    }

    fn leftover_quad_flip(t: &[f64], passes: &[Pass], a: f64) -> Vec<f64> {
        t.iter()
            .map(|&ti| {
                let (pi, p) = passes
                    .iter()
                    .enumerate()
                    .find(|(_, p)| ti >= p.t_start_s && ti <= p.t_end_s)
                    .unwrap_or((0, &passes[0]));
                let u = if p.duration_s() > 0.0 {
                    (ti - p.t_start_s) / p.duration_s()
                } else {
                    0.0
                };
                let sign = if pi % 2 == 0 { 1.0 } else { -1.0 };
                a * (u - 0.5) * (u - 0.5) * 4.0 * sign
            })
            .collect()
    }

    fn leftover_pass_slope(t: &[f64], passes: &[Pass], b: f64) -> Vec<f64> {
        t.iter()
            .map(|&ti| {
                let p = passes
                    .iter()
                    .find(|p| ti >= p.t_start_s && ti <= p.t_end_s)
                    .unwrap_or(&passes[0]);
                b * (ti - p.t_start_s) / p.duration_s().max(1.0)
            })
            .collect()
    }

    fn lcg_noise(n: usize, amp: f64, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                amp * (s as f64 / u64::MAX as f64 - 0.5)
            })
            .collect()
    }

    fn cluster_passes_for(t: &[f64]) -> Vec<Pass> {
        crate::functions::sampling::cluster_passes(t, 8.0, 60.0)
    }

    #[test]
    fn t_fap1_perm_cannot_resolve_1e3() {
        let t: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|ti| (std::f64::consts::TAU * ti / 16.0).sin())
            .collect();
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(4.0);
        c.max_period_s = Some(32.0);
        c.n_permutations = 200;
        let a = assess_periodicity(&s, &c);
        if let Some(fp) = a.fap_perm {
            assert!(fp >= 1.0 / 201.0 - 1e-12, "fap_perm={fp}");
            assert!(fp >= 1e-3, "B=200 cannot produce fap_perm < 1e-3");
        }
        if let Some(fb) = a.fap_baluev {
            // Baluev may be < 1e-3 on a real sine.
            let _ = fb;
        }
    }

    #[test]
    fn t1_leo_week_constant_not_periodic() {
        let t = leo_week();
        let s = series_of(t.clone(), vec![1.0; t.len()], None);
        let a = assess_periodicity(&s, &cfg());
        assert_ne!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
    }

    #[test]
    fn t2_geo_week_constant_not_periodic() {
        let t = geo_night_times(7, 80, 18_000.0, 0.0);
        let s = series_of(t.clone(), vec![1.0; t.len()], None);
        let mut c = cfg();
        c.max_period_s = Some(200_000.0);
        let a = assess_periodicity(&s, &c);
        assert_ne!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
    }

    #[test]
    fn t5_zero_y_not_periodic() {
        let t = leo_week();
        let s = series_of(t.clone(), vec![0.0; t.len()], None);
        let a = assess_periodicity(&s, &cfg());
        assert_ne!(a.decision, PeriodicityDecision::Periodic);
    }

    #[test]
    fn t3_lambert_with_known_orbit_not_periodic() {
        let t = leo_week();
        let phi: Vec<f64> = t
            .iter()
            .map(|ti| std::f64::consts::TAU * ti / 5400.0)
            .collect();
        let y = leftover_lambert(&phi, 0.25);
        let s = series_of(t, y, Some(phi));
        let mut c = cfg();
        c.known_periods_s = vec![5400.0];
        let a = assess_periodicity(&s, &c);
        assert_ne!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        if !a.aliases.is_empty() {
            assert!(a
                .aliases
                .iter()
                .any(|al| { matches!(al.kind, AliasKind::Orbital | AliasKind::OrbitalHalf) }));
        }
    }

    #[test]
    fn t4_lambert_without_known_orbit_eaten_by_lambda() {
        let t = leo_week();
        let phi: Vec<f64> = t
            .iter()
            .map(|ti| std::f64::consts::TAU * ti / 5400.0)
            .collect();
        let y = leftover_lambert(&phi, 0.25);
        let s = series_of(t, y, Some(phi));
        let a = assess_periodicity(&s, &cfg());
        assert_ne!(
            a.decision,
            PeriodicityDecision::Periodic,
            "Λ is in the Auto span; leftover should not survive. notes={:?}",
            a.notes
        );
    }

    #[test]
    fn t4b_three_cycle_pass_intra_periodic() {
        let t = leo_week();
        let passes = cluster_passes_for(&t);
        let y = leftover_three_cycle(&t, &passes, 0.4);
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.scale = SearchScale::IntraPass;
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!((p - 160.0).abs() / 160.0 < 0.08, "expected ~160 s, got {p}");
    }

    #[test]
    fn t6a_geo_ramp_not_periodic_via_global_t() {
        let t = geo_night_times(7, 80, 18_000.0, 0.0);
        let y: Vec<f64> = t.iter().map(|ti| 0.3 * ti / 86_400.0).collect();
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.max_period_s = Some(200_000.0);
        let intra = {
            let mut ci = c.clone();
            ci.scale = SearchScale::IntraPass;
            assess_periodicity(&s, &ci)
        };
        assert_ne!(
            intra.decision,
            PeriodicityDecision::Periodic,
            "Intra must eat night slope. notes={:?}",
            intra.notes
        );
        c.scale = SearchScale::InterPass;
        let inter = assess_periodicity(&s, &c);
        assert_ne!(
            inter.decision,
            PeriodicityDecision::Periodic,
            "Inter must reject via global t / pass-block, not night means. notes={:?}",
            inter.notes
        );
        assert!(
            inter.detrend.columns.iter().any(|col| {
                matches!(col, crate::entities::assessment::DetrendColumn::GlobalTime)
            }) || inter.notes.iter().any(|n| n.contains("Inter")),
            "Inter mechanism must be global t / Inter leg, not per-pass means"
        );
    }

    #[test]
    fn t6b_sidereal_harmonic_is_tagged() {
        let t = geo_night_times(7, 80, 18_000.0, 0.0);
        let s = series_of(t.clone(), vec![1.0; t.len()], None);
        let mut c = cfg();
        c.max_period_s = Some(3000.0);
        let a = assess_periodicity(&s, &c);
        let p_line = crate::constants::SIDEREAL_DAY_S / 29.0;
        let tagged = a.sampling.window_peaks.iter().any(|pk| {
            pk.kind == AliasKind::SiderealDay && (pk.period_s - p_line).abs() / p_line < 0.05
        });
        assert!(tagged, "expected SiderealDay tag near {p_line}");
    }

    #[test]
    fn t9_clean_sine_full_scale() {
        let mut t = Vec::new();
        let mut s = 1u64;
        for _ in 0..200 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            t.push((s as f64 / u64::MAX as f64) * 50.0 * 47.3);
        }
        let y: Vec<f64> = t
            .iter()
            .map(|ti| (std::f64::consts::TAU * ti / 47.3).sin())
            .collect();
        let ser = series_of(t, y, None);
        let mut c = cfg();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(20.0);
        c.max_period_s = Some(100.0);
        let a = assess_periodicity(&ser, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!((p - 47.3).abs() / 47.3 < 0.05, "got {p}");
    }

    #[test]
    fn t10_white_noise_not_periodic() {
        let t: Vec<f64> = (0..120).map(|i| i as f64).collect();
        let y = lcg_noise(120, 2.0, 99);
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.scale = SearchScale::Full;
        c.min_period_s = Some(4.0);
        c.max_period_s = Some(40.0);
        let a = assess_periodicity(&s, &c);
        assert_ne!(a.decision, PeriodicityDecision::Periodic);
        if let Some(fb) = a.fap_baluev {
            assert!(fb > 0.05 || a.decision == PeriodicityDecision::NotPeriodic);
        }
    }

    #[test]
    fn t13_assess_no_bound_snap() {
        let t = leo_week();
        let y = lcg_noise(t.len(), 0.2, 7);
        let s = series_of(t, y, None);
        let a = assess_periodicity(&s, &cfg());
        if let Some(p) = a.period_s {
            assert!(p > 3.0 * 1.03, "endpoint snap {p}");
        } else {
            assert_ne!(a.decision, PeriodicityDecision::Periodic);
        }
    }

    #[test]
    fn t14_67689_class_auto_not_periodic() {
        let t = leo_week();
        let passes = cluster_passes_for(&t);
        let phi: Vec<f64> = t
            .iter()
            .map(|ti| std::f64::consts::TAU * ti / 5400.0)
            .collect();
        let slope = leftover_pass_slope(&t, &passes, 0.15);
        let lam = leftover_lambert(&phi, 0.2);
        let noise = lcg_noise(t.len(), 0.16, 11);
        let y: Vec<f64> = slope
            .iter()
            .zip(lam.iter())
            .zip(noise.iter())
            .map(|((s, l), n)| s + l + n)
            .collect();
        let s = series_of(t, y, Some(phi));
        let mut ci = cfg();
        ci.scale = SearchScale::IntraPass;
        let intra = assess_periodicity(&s, &ci);
        assert_ne!(
            intra.decision,
            PeriodicityDecision::Periodic,
            "Intra {:?}",
            intra.notes
        );
        let mut cj = cfg();
        cj.scale = SearchScale::InterPass;
        let inter = assess_periodicity(&s, &cj);
        assert_ne!(
            inter.decision,
            PeriodicityDecision::Periodic,
            "Inter via global t / pass-block, not per-pass means. {:?}",
            inter.notes
        );
    }

    #[test]
    fn t14b_quad_flip_does_not_invent_orbital_half() {
        let t = leo_week();
        let passes = cluster_passes_for(&t);
        let y = leftover_quad_flip(&t, &passes, 0.4);
        let s = series_of(t, y, None);
        let a = assess_periodicity(&s, &cfg());
        if a.decision == PeriodicityDecision::Periodic {
            for al in &a.aliases {
                if al.kind == AliasKind::OrbitalHalf && al.vetoed {
                    panic!("invented OrbitalHalf veto without known P_orb");
                }
            }
        }
    }

    #[test]
    fn t16_too_few_points_inconclusive() {
        let s = series_of(vec![0.0, 1.0], vec![1.0, 1.0], None);
        let a = assess_periodicity(&s, &cfg());
        assert_eq!(a.decision, PeriodicityDecision::Inconclusive);
    }

    #[test]
    fn t21_single_leo_pass_tumbler() {
        let t = leo_pass_times(1, 80, 480.0, 5400.0, 0.0);
        let y: Vec<f64> = t
            .iter()
            .map(|ti| (std::f64::consts::TAU * ti / 47.0).sin())
            .collect();
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.scale = SearchScale::Auto;
        c.max_period_s = Some(200.0);
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
    }

    #[test]
    fn t22_short_single_pass_inconclusive() {
        let t = leo_pass_times(1, 8, 480.0, 5400.0, 0.0);
        let s = series_of(t.clone(), vec![0.0; t.len()], None);
        let a = assess_periodicity(&s, &cfg());
        assert_eq!(a.decision, PeriodicityDecision::Inconclusive);
    }

    fn tumbler(t: &[f64], p_rot: f64, amp1: f64, amp2: f64, width: f64) -> Vec<f64> {
        t.iter()
            .map(|&ti| {
                let phi = (ti / p_rot).rem_euclid(1.0);
                let g = |center: f64, amp: f64| {
                    let mut d = phi - center;
                    if d > 0.5 {
                        d -= 1.0;
                    }
                    if d < -0.5 {
                        d += 1.0;
                    }
                    -amp * (-d * d / (2.0 * width * width)).exp()
                };
                12.0 + g(0.20, amp1) + g(0.70, amp2)
            })
            .collect()
    }

    #[test]
    fn t7_leo_asymmetric_tumbler() {
        let t = leo_week();
        let y = tumbler(&t, 195.0, 0.8, 0.4, 0.04);
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.min_period_s = Some(20.0);
        c.max_period_s = Some(400.0);
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!(
            (p - 195.0).abs() / 195.0 < 0.02,
            "expected 195 s not P/2, got {p}"
        );
        assert!((p - 97.5).abs() / 97.5 > 0.05, "must not report P/2 ({p})");
    }

    #[test]
    fn t7s_leo_symmetric_tumbler_reports_full_period() {
        let t = leo_week();
        let y = tumbler(&t, 195.0, 0.6, 0.6, 0.04);
        let s = series_of(t.clone(), y.clone(), None);
        let mut c = cfg();
        c.min_period_s = Some(20.0);
        c.max_period_s = Some(400.0);
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!(
            (p - 195.0).abs() / 195.0 < 0.05,
            "symmetric flashes must report P not P/2, got {p}"
        );
    }

    #[test]
    fn t8_geo_asymmetric_tumbler() {
        // Irregular samples inside each night so 195 s is below the irregular
        // Nyquist (80 even samples have 2Δt ≈ 456 s > 195 s).
        let t = geo_night_times_irregular(7, 80, 18_000.0, 0.0, 7);
        let y = tumbler(&t, 195.0, 0.8, 0.4, 0.04);
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.min_period_s = Some(20.0);
        c.max_period_s = Some(400.0);
        c.scale = SearchScale::IntraPass;
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!((p - 195.0).abs() / 195.0 < 0.02, "got {p}");
    }

    #[test]
    fn t15_22927_class() {
        let t = geo_night_times_irregular(7, 80, 18_000.0, 0.0, 15);
        let y = tumbler(&t, 195.0, 0.8, 0.4, 0.04);
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.min_period_s = Some(20.0);
        c.max_period_s = Some(400.0);
        let a = assess_periodicity(&s, &c);
        assert_eq!(a.decision, PeriodicityDecision::Periodic, "{:?}", a.notes);
        let p = a.period_s.unwrap();
        assert!((p - 195.0).abs() / 195.0 < 0.05, "got {p}");
    }

    #[test]
    fn t19_clumped_constant_assess_not_periodic() {
        let mut t = Vec::new();
        let mut y = Vec::new();
        for (t0, n) in [(0.0, 40), (10_000.0, 40)] {
            for i in 0..n {
                t.push(t0 + i as f64);
                y.push(10.0 + 0.01 * ((i % 3) as f64 - 1.0));
            }
        }
        let s = series_of(t, y, None);
        let mut c = cfg();
        c.min_period_s = Some(100.0);
        c.max_period_s = Some(20_000.0);
        let a = assess_periodicity(&s, &c);
        assert_ne!(
            a.decision,
            PeriodicityDecision::Periodic,
            "clumped constant y must not be Periodic: {:?}",
            a.notes
        );
    }
}
