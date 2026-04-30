//! Run all three period estimators on a randomly-sampled noisy periodic signal
//! and produce SVG plots of the raw lightcurve plus phase-folded curves at the
//! period each estimator recovers.
//!
//! Run with:
//!     cargo run --release --example period_finder_demo
//!
//! Outputs in the current directory:
//!     lightcurve_raw.svg
//!     lightcurve_folded_string_length.svg
//!     lightcurve_folded_gregory_loredo.svg
//!     lightcurve_folded_qp_gp.svg

use bland::{Figure, Marker, PaperSize};
use cepheid::entities::lightcurve::Lightcurve;
use cepheid::entities::observation::Observation;
use cepheid::functions::periodicity::{
    GregoryLoredoPeriodEstimator, QuasiPeriodicGPPeriodEstimator, StringLengthPeriodEstimator,
};
use chrono::{DateTime, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let true_period = 47.3_f64;
    let amplitude = 1.5;
    let mean_mag = 11.0;
    let noise_amp = 0.05;
    let num_samples = 200;
    let span = 50.0 * true_period;

    let mut rng = StdRng::seed_from_u64(20251209);
    let base_time = DateTime::parse_from_rfc3339("2025-04-29T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc);

    let observations: Vec<Observation> = (0..num_samples)
        .map(|_| {
            let t: f64 = rng.random::<f64>() * span;
            let timestamp = base_time + chrono::Duration::milliseconds((t * 1000.0) as i64);
            // Two harmonics → tumbler-like double-peak shape.
            let phase = 2.0 * std::f64::consts::PI * t / true_period;
            let signal = amplitude * phase.sin() + 0.4 * (2.0 * phase).sin();
            let noise = noise_amp * (rng.random::<f64>() * 2.0 - 1.0);
            let mag = mean_mag + signal + noise;
            Observation {
                vismag: mag,
                range_m: 1000.0e3,
                phase_rad: 0.0,
                std_magnitude: mag,
                timestamp,
                fractional_period: None,
            }
        })
        .collect();

    let lightcurve = Lightcurve::new(observations, Some(true), Some(true_period));

    let min_p = 20.0;
    let max_p = 100.0;
    let max_frac_err = 0.005;

    println!("True period: {:.4} s", true_period);
    println!("Samples: {}, span: {:.1} s\n", num_samples, span);

    let sl_period = StringLengthPeriodEstimator::estimate_period(
        &lightcurve, min_p, max_p, max_frac_err, None,
    );
    println!("String length         : {:?}", sl_period);

    let gl = GregoryLoredoPeriodEstimator::estimate_period(
        &lightcurve, min_p, max_p, max_frac_err, None, None, None,
    );
    println!(
        "Gregory-Loredo (var-m): {:?} (log odds = {:.2})",
        gl.period_s, gl.log_odds
    );

    let gp = QuasiPeriodicGPPeriodEstimator::estimate_period(
        &lightcurve, min_p, max_p, max_frac_err, None, None, None,
    );
    println!(
        "Quasi-periodic GP     : {:?} (log odds = {:.2})",
        gp.period_s, gp.log_odds
    );

    save_raw_plot("lightcurve_raw.svg", &lightcurve);

    save_folded_plot(
        "lightcurve_folded_string_length.svg",
        "String length",
        sl_period.unwrap_or(true_period),
        &lightcurve,
    );
    save_folded_plot(
        "lightcurve_folded_gregory_loredo.svg",
        "Gregory-Loredo (variable-m)",
        gl.period_s.unwrap_or(true_period),
        &lightcurve,
    );
    save_folded_plot(
        "lightcurve_folded_qp_gp.svg",
        "Quasi-periodic GP",
        gp.period_s.unwrap_or(true_period),
        &lightcurve,
    );

    println!("\nWrote 4 SVGs to current directory.");
}

fn save_raw_plot(filename: &str, lc: &Lightcurve) {
    let t0 = lc
        .observations
        .iter()
        .map(|o| o.unix_seconds())
        .fold(f64::INFINITY, f64::min);
    let times: Vec<f64> = lc
        .observations
        .iter()
        .map(|o| o.unix_seconds() - t0)
        .collect();
    let mags: Vec<f64> = lc.observations.iter().map(|o| o.std_magnitude).collect();

    let fig = Figure::new()
        .size(PaperSize::A5Landscape)
        .title("Raw lightcurve (random sampling)")
        .xlabel("time [s]")
        .ylabel("std magnitude")
        .scatter(&times, &mags, |s| {
            s.label("observations").marker(Marker::CircleFilled)
        });
    std::fs::write(filename, fig.to_svg()).expect("write svg");
}

fn save_folded_plot(filename: &str, name: &str, period: f64, lc: &Lightcurve) {
    let mut pairs: Vec<(f64, f64)> = lc
        .observations
        .iter()
        .map(|o| {
            let t = o.unix_seconds();
            let phase = (t.rem_euclid(period)) / period;
            (phase, o.std_magnitude)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let phis: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
    let ms: Vec<f64> = pairs.iter().map(|(_, m)| *m).collect();

    let title = format!("{} — folded at P = {:.4} s", name, period);
    let fig = Figure::new()
        .size(PaperSize::A5Landscape)
        .title(&title)
        .xlabel("phase")
        .ylabel("std magnitude")
        .scatter(&phis, &ms, |s| s.label("folded").marker(Marker::CircleFilled));
    std::fs::write(filename, fig.to_svg()).expect("write svg");
}
