//! Pulls real optical observations from the Citra Space API for two satellites
//! and runs all three period estimators on each:
//!
//!   - NORAD 67689 — stable LEO       (no period expected; estimators should reject)
//!   - NORAD 22927 — tumbling GEO     (period expected; tumbler-style double peak)
//!
//! Set `CITRA_PAT` to a Citra personal access token, then:
//!
//!     cargo run --release --example citra_period_finder
//!
//! For each satellite the example writes:
//!     <prefix>_raw.svg                       (magnitude vs. time)
//!     <prefix>_folded_string_length.svg      (only if string-length detects)
//!     <prefix>_folded_gregory_loredo.svg     (only if G-L detects)
//!     <prefix>_folded_qp_gp.svg              (only if QP-GP detects)
//!
//! Magnitudes are normalized to standard observing conditions (1000 km
//! range, 90° solar phase) using SGP4 propagation of the latest
//! Sgp4-with-Kozai-mean-motion elset for each target. SGP4-XP elsets are
//! skipped because the standard `sgp4` crate can't propagate them.

use bland::{Figure, Marker, PaperSize};
use cepheid::entities::lightcurve::Lightcurve;
use cepheid::entities::observation::Observation;
use cepheid::functions::periodicity::{
    GregoryLoredoPeriodEstimator, QuasiPeriodicGPPeriodEstimator, StringLengthPeriodEstimator,
};
use chrono::{DateTime, Duration, Utc};
use lemonaid::types::CitraElsetType;
use lemonaid::CitraClient;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use uuid::Uuid;

const CITRA_DEV_BASE_URL: &str = "https://dev.api.citra.space";
const CITRA_API_VERSION: &str = "0.1.0";

const STD_RANGE_M: f64 = 1_000_000.0; // 1000 km
const STD_PHASE_DEG: f64 = 90.0;

struct Target {
    norad: i64,
    label: &'static str,
    file_prefix: &'static str,
}

const TARGETS: &[Target] = &[
    Target {
        norad: 67689,
        label: "NORAD 67689 (LEO, stable)",
        file_prefix: "leo_67689",
    },
    Target {
        norad: 22927,
        label: "NORAD 22927 (GEO, tumbling)",
        file_prefix: "geo_22927",
    },
];

const LOOKBACK_DAYS: i64 = 7;
const MAX_OBSERVATIONS: usize = 5000;
const MIN_PERIOD_S: f64 = 3.0;
const MAX_PERIOD_S: f64 = 3000.0;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pat = std::env::var("CITRA_PAT")
        .map_err(|_| "CITRA_PAT environment variable must be set")?;

    // Hit dev.api.citra.space. Pass `false` here to target prod instead.
    let client = CitraClient::new(&pat, true);
    let http = reqwest::Client::new();

    for target in TARGETS {
        println!("=== {} ===", target.label);
        if let Err(e) = analyze(&client, &http, &pat, target).await {
            eprintln!("  failed: {}", e);
        }
        println!();
    }

    Ok(())
}

async fn analyze(
    client: &lemonaid::Client,
    http: &reqwest::Client,
    pat: &str,
    target: &Target,
) -> Result<(), Box<dyn std::error::Error>> {
    let sat_uuid = lookup_satellite_uuid(client, target.norad).await?;
    println!("  satellite UUID: {}", sat_uuid);

    // Pull the most recent SGP4-with-Kozai-mean-motion elset and build the
    // propagator once for this target. Range computation per observation is
    // a single SGP4 call + a topocentric-rotation arithmetic.
    let propagator = fetch_latest_sgp4_propagator(client, sat_uuid).await?;
    println!(
        "  elset epoch: {} ({:.2} d before now)",
        propagator.epoch.to_rfc3339(),
        (Utc::now() - propagator.epoch).num_seconds() as f64 / 86_400.0
    );

    // Fetch optical observations via raw HTTP. lemonaid 0.4 generates a typed
    // call here, but the Citra OpenAPI spec has two unrelated `Status` schemas
    // (sensor health: online/offline, query status: success/running/error)
    // that progenitor merged into a single Rust enum, so the typed deserializer
    // chokes on the `"success"` value the API actually returns. The
    // `dataPoints` payload itself is fine — pull the JSON directly.
    let mut observations = fetch_optical_observations(http, pat, sat_uuid, &propagator).await?;
    if observations.len() > MAX_OBSERVATIONS {
        let raw = observations.len();
        let mut rng = StdRng::seed_from_u64(target.norad as u64);
        observations.shuffle(&mut rng);
        observations.truncate(MAX_OBSERVATIONS);
        println!("  decimated {} → {} (random subsample)", raw, MAX_OBSERVATIONS);
    }

    let n = observations.len();
    if n < 20 {
        println!("  too few usable points ({}); skipping estimators", n);
        return Ok(());
    }

    let lc = Lightcurve::new(observations, None, None);
    let span_s = lc.data_span_s();
    println!(
        "  {} usable points, span {:.1} s ({:.2} h)",
        n,
        span_s,
        span_s / 3600.0
    );

    // Don't trial-search beyond half the data span — periods longer than that
    // can't be confidently distinguished from a slow trend.
    let max_p = MAX_PERIOD_S.min(span_s / 2.0);
    if max_p <= MIN_PERIOD_S {
        println!("  data span too short for requested period range; skipping");
        return Ok(());
    }
    println!("  searching periods {:.1}–{:.1} s", MIN_PERIOD_S, max_p);

    let sl_period = StringLengthPeriodEstimator::estimate_period(
        &lc,
        MIN_PERIOD_S,
        max_p,
        0.005,
        None,
    );
    println!("    string length         : {:?}", sl_period);

    let gl = GregoryLoredoPeriodEstimator::estimate_period(
        &lc,
        MIN_PERIOD_S,
        max_p,
        0.005,
        None,
        None,
        None,
    );
    println!(
        "    Gregory-Loredo (var-m): {:?} (log odds = {:.2})",
        gl.period_s, gl.log_odds
    );

    let gp = QuasiPeriodicGPPeriodEstimator::estimate_period(
        &lc,
        MIN_PERIOD_S,
        max_p,
        0.005,
        None,
        None,
        None,
        None,
    );
    println!(
        "    quasi-periodic GP     : {:?} (log odds = {:.2})",
        gp.period_s, gp.log_odds
    );

    save_raw_plot(
        &format!("{}_raw.svg", target.file_prefix),
        target.label,
        &lc,
    );
    if let Some(p) = sl_period {
        save_folded_plot(
            &format!("{}_folded_string_length.svg", target.file_prefix),
            target.label,
            "string length",
            p,
            &lc,
        );
    }
    if let Some(p) = gl.period_s {
        save_folded_plot(
            &format!("{}_folded_gregory_loredo.svg", target.file_prefix),
            target.label,
            "Gregory-Loredo",
            p,
            &lc,
        );
    }
    if let Some(p) = gp.period_s {
        save_folded_plot(
            &format!("{}_folded_qp_gp.svg", target.file_prefix),
            target.label,
            "quasi-periodic GP",
            p,
            &lc,
        );
    }

    Ok(())
}

async fn fetch_optical_observations(
    http: &reqwest::Client,
    pat: &str,
    sat_uuid: Uuid,
    propagator: &Sgp4Propagator,
) -> Result<Vec<Observation>, Box<dyn std::error::Error>> {
    let start = Utc::now() - Duration::days(LOOKBACK_DAYS);
    let url = format!("{}/observations/optical", CITRA_DEV_BASE_URL);
    let body: serde_json::Value = http
        .get(url)
        .bearer_auth(pat)
        .header("api-version", CITRA_API_VERSION)
        .header("accept", "application/json")
        .query(&[
            ("satelliteId", sat_uuid.to_string()),
            ("startAfter", start.to_rfc3339()),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let points = body
        .get("dataPoints")
        .and_then(|v| v.as_array())
        .ok_or("response missing dataPoints array")?;

    let raw_count = points.len();
    let mut observations = Vec::with_capacity(raw_count);
    let mut dropped = 0usize;
    for p in points {
        let Some(epoch_str) = p.get("epoch").and_then(|v| v.as_str()) else {
            dropped += 1;
            continue;
        };
        let Some(vmag) = p.get("visualMagnitude").and_then(|v| v.as_f64()) else {
            dropped += 1;
            continue;
        };
        let Some(phase_deg) = p.get("solarPhaseAngle").and_then(|v| v.as_f64()) else {
            dropped += 1;
            continue;
        };
        let Some(sensor_lat) = p.get("sensorLatitude").and_then(|v| v.as_f64()) else {
            dropped += 1;
            continue;
        };
        let Some(sensor_lon) = p.get("sensorLongitude").and_then(|v| v.as_f64()) else {
            dropped += 1;
            continue;
        };
        let Some(sensor_alt_km) = p.get("sensorAltitude").and_then(|v| v.as_f64()) else {
            dropped += 1;
            continue;
        };
        let Ok(timestamp) = DateTime::parse_from_rfc3339(epoch_str) else {
            dropped += 1;
            continue;
        };
        let timestamp = timestamp.with_timezone(&Utc);

        let range_m = match propagator.range_to_sensor_m(
            timestamp,
            sensor_lat,
            sensor_lon,
            sensor_alt_km,
        ) {
            Some(r) => r,
            None => {
                dropped += 1;
                continue;
            }
        };

        observations.push(Observation::new(
            vmag,
            range_m,
            phase_deg.to_radians(),
            timestamp,
            STD_RANGE_M,
            STD_PHASE_DEG.to_radians(),
        ));
    }

    println!(
        "  pulled {} observations (last {} days); {} kept after propagation, {} dropped",
        raw_count,
        LOOKBACK_DAYS,
        observations.len(),
        dropped
    );

    Ok(observations)
}

async fn fetch_latest_sgp4_propagator(
    client: &lemonaid::Client,
    sat_uuid: Uuid,
) -> Result<Sgp4Propagator, Box<dyn std::error::Error>> {
    // SGP4-XP and SP elsets aren't propagatable by the standard sgp4 crate, so
    // restrict to the two SGP4 mean-motion variants. Citra returns these
    // newest-first when a `limit` is given.
    let allowed_types = vec![
        CitraElsetType::Sgp4WithKozaiMeanMotion,
        CitraElsetType::Sgp4WithBrouwerMeanMotion,
    ];
    let limit = std::num::NonZeroU64::new(10).unwrap();
    let resp = client
        .get_satellite_elsets_satellites_satellite_id_elsets_get(
            &sat_uuid,
            Some(limit),
            Some(&allowed_types),
        )
        .await?;
    let elsets = resp.into_inner();

    // Find the newest elset whose two-line text we can actually parse.
    let mut last_err: Option<String> = None;
    for elset in elsets.0.into_iter() {
        if !matches!(
            elset.type_,
            CitraElsetType::Sgp4WithKozaiMeanMotion | CitraElsetType::Sgp4WithBrouwerMeanMotion
        ) {
            continue;
        }
        let Some(tle_arr) = &elset.tle else {
            last_err = Some("elset missing TLE strings".into());
            continue;
        };
        let line1 = tle_arr[0].as_str();
        let line2 = tle_arr[1].as_str();
        let (Some(line1), Some(line2)) = (line1, line2) else {
            last_err = Some("TLE entries are not strings".into());
            continue;
        };
        let tle_text = format!("{}\n{}", line1, line2);
        let elements = match sgp4::parse_2les(&tle_text) {
            Ok(mut v) if !v.is_empty() => v.remove(0),
            Ok(_) => {
                last_err = Some("parse_2les returned no elements".into());
                continue;
            }
            Err(e) => {
                last_err = Some(format!("parse_2les failed: {}", e));
                continue;
            }
        };
        let constants = match sgp4::Constants::from_elements(&elements) {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(format!("Constants::from_elements failed: {}", e));
                continue;
            }
        };
        let epoch = DateTime::parse_from_rfc3339(&elset.epoch)?.with_timezone(&Utc);
        return Ok(Sgp4Propagator { constants, epoch });
    }

    Err(last_err
        .unwrap_or_else(|| "no SGP4 elsets available for this satellite".into())
        .into())
}

struct Sgp4Propagator {
    constants: sgp4::Constants,
    epoch: DateTime<Utc>,
}

impl Sgp4Propagator {
    fn range_to_sensor_m(
        &self,
        when: DateTime<Utc>,
        sensor_lat_deg: f64,
        sensor_lon_deg: f64,
        sensor_alt_km: f64,
    ) -> Option<f64> {
        let dt_min = (when - self.epoch).num_milliseconds() as f64 / 60_000.0;
        let pred = self
            .constants
            .propagate(sgp4::MinutesSinceEpoch(dt_min))
            .ok()?;
        let sat_teme_km = pred.position; // [x, y, z] in km

        // Convert sensor geodetic → ECEF → TEME (about Z by GMST).
        let sensor_ecef_km = geodetic_to_ecef_km(sensor_lat_deg, sensor_lon_deg, sensor_alt_km);
        let gmst = gmst_rad(when);
        let sensor_teme_km = rotate_z(sensor_ecef_km, gmst);

        let dx = sat_teme_km[0] - sensor_teme_km[0];
        let dy = sat_teme_km[1] - sensor_teme_km[1];
        let dz = sat_teme_km[2] - sensor_teme_km[2];
        Some((dx * dx + dy * dy + dz * dz).sqrt() * 1000.0)
    }
}

fn geodetic_to_ecef_km(lat_deg: f64, lon_deg: f64, alt_km: f64) -> [f64; 3] {
    // WGS-84 ellipsoid.
    const A_KM: f64 = 6378.137;
    const F: f64 = 1.0 / 298.257_223_563;
    let e2 = 2.0 * F - F * F;
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let sin_lat = lat.sin();
    let n = A_KM / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    [
        (n + alt_km) * lat.cos() * lon.cos(),
        (n + alt_km) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + alt_km) * sin_lat,
    ]
}

fn gmst_rad(when: DateTime<Utc>) -> f64 {
    // Vallado §3.5.1: GMST in seconds, then to radians. Skipping nutation/
    // precession is fine for kilometer-level range — magnitude normalization
    // depends on log10(range), so even a few-arcsecond pointing error is
    // millimagnitude-level in the corrected vmag.
    let jd_ut1 = unix_to_jd(when.timestamp_millis() as f64 / 1000.0);
    let t = (jd_ut1 - 2_451_545.0) / 36_525.0;
    let gmst_sec = 67_310.548_41
        + (876_600.0 * 3600.0 + 8_640_184.812_866) * t
        + 0.093_104 * t * t
        - 6.2e-6 * t * t * t;
    let gmst_sec = gmst_sec.rem_euclid(86_400.0);
    gmst_sec * std::f64::consts::TAU / 86_400.0
}

fn unix_to_jd(unix_seconds: f64) -> f64 {
    2_440_587.5 + unix_seconds / 86_400.0
}

fn rotate_z(v: [f64; 3], theta: f64) -> [f64; 3] {
    let c = theta.cos();
    let s = theta.sin();
    [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]
}

async fn lookup_satellite_uuid(
    client: &lemonaid::Client,
    norad: i64,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let norad_str = norad.to_string();
    let resp = client
        .get_satellites_satellites_get(
            None,
            None,
            None,
            None,
            Some(&norad_str),
            None,
            Some(false),
            None,
        )
        .await?;
    let page = resp.into_inner();
    let sat = page
        .items
        .into_iter()
        .find(|s| s.norad_cat_id == Some(norad))
        .ok_or_else(|| format!("no satellite found with NORAD {}", norad))?;
    Ok(Uuid::parse_str(&sat.id)?)
}

fn save_raw_plot(filename: &str, target_label: &str, lc: &Lightcurve) {
    let t0 = lc
        .observations
        .iter()
        .map(|o| o.unix_seconds())
        .fold(f64::INFINITY, f64::min);
    let times: Vec<f64> = lc
        .observations
        .iter()
        .map(|o| (o.unix_seconds() - t0) / 3600.0)
        .collect();
    let mags: Vec<f64> = lc.observations.iter().map(|o| o.std_magnitude).collect();

    let title = format!("Raw lightcurve — {}", target_label);
    let fig = Figure::new()
        .size(PaperSize::A5Landscape)
        .title(&title)
        .xlabel("time [hours from first observation]")
        .ylabel("apparent magnitude")
        .scatter(&times, &mags, |s| {
            s.label("observations").marker(Marker::CircleFilled)
        });
    std::fs::write(filename, fig.to_svg()).expect("write svg");
}

fn save_folded_plot(filename: &str, target_label: &str, estimator: &str, period: f64, lc: &Lightcurve) {
    let mut pairs: Vec<(f64, f64)> = lc
        .observations
        .iter()
        .map(|o| {
            let t = o.unix_seconds();
            let phase = t.rem_euclid(period) / period;
            (phase, o.std_magnitude)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let phis: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
    let ms: Vec<f64> = pairs.iter().map(|(_, m)| *m).collect();

    let title = format!(
        "{} — {} folded at P = {:.4} s",
        target_label, estimator, period
    );
    let fig = Figure::new()
        .size(PaperSize::A5Landscape)
        .title(&title)
        .xlabel("phase")
        .ylabel("apparent magnitude")
        .scatter(&phis, &ms, |s| {
            s.label("folded").marker(Marker::CircleFilled)
        });
    std::fs::write(filename, fig.to_svg()).expect("write svg");
}
