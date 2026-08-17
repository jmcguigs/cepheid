//! Scale-dependent Auto detrend.
//!
//! Intra: this-pass mean + τ (or Λ). Inter/Full: global mean + global t +
//! shared τ, **no per-pass intercepts**.

use crate::entities::assessment::Pass;
use crate::entities::assessment::{DetrendColumn, DetrendMode, DetrendReport, SearchScale};
use crate::entities::series::{Modality, Series};
use nalgebra::{DMatrix, DVector};

const HUBER_C: f64 = 1.345;
const COND_DROP: f64 = 1.0e6;
const COND_FAIL: f64 = 1.0e8;

pub struct DetrendResult {
    pub y: Vec<f64>,
    pub report: DetrendReport,
}

pub fn auto_detrend(
    series: &Series,
    passes: &[Pass],
    scale: SearchScale,
    mode: DetrendMode,
    indices: Option<&[usize]>,
) -> DetrendResult {
    let t = series.t_s();
    let y = series.y();
    let w0 = series.weights();
    let idx: Vec<usize> = match indices {
        Some(i) => i.to_vec(),
        None => (0..t.len()).collect(),
    };
    if idx.is_empty() {
        return DetrendResult {
            y: Vec::new(),
            report: DetrendReport::default(),
        };
    }

    let pass_of = assign_passes(t, passes);
    let mut pass_mean = vec![0.0; passes.len().max(1)];
    if !passes.is_empty() {
        for (p, pass) in passes.iter().enumerate() {
            let mut sw = 0.0;
            let mut swy = 0.0;
            for &i in &idx {
                if t[i] >= pass.t_start_s && t[i] <= pass.t_end_s {
                    sw += w0[i];
                    swy += w0[i] * y[i];
                }
            }
            pass_mean[p] = if sw > 0.0 { swy / sw } else { 0.0 };
        }
    } else {
        let sw: f64 = idx.iter().map(|&i| w0[i]).sum();
        let swy: f64 = idx.iter().map(|&i| w0[i] * y[i]).sum();
        pass_mean[0] = if sw > 0.0 { swy / sw } else { 0.0 };
    }

    if mode == DetrendMode::None {
        return DetrendResult {
            y: idx.iter().map(|&i| y[i]).collect(),
            report: DetrendReport {
                mode,
                scale,
                pass_mean,
                ..DetrendReport::default()
            },
        };
    }

    let intra = matches!(scale, SearchScale::IntraPass);
    let phi = series.covariates().solar_phase_rad.as_deref();
    let elev = series.covariates().elevation_rad.as_deref();
    let want_phase = matches!(mode, DetrendMode::Auto | DetrendMode::PhaseFunction)
        && phi.is_some()
        && series.meta().modality != Modality::RfPower;
    let want_elev = matches!(mode, DetrendMode::Auto | DetrendMode::Elevation)
        && elev.is_some()
        && series.meta().modality == Modality::RfPower;

    let mut cols: Vec<DetrendColumn> = Vec::new();
    let mut raw: Vec<Vec<f64>> = Vec::new();

    // Intercept
    if intra {
        cols.push(DetrendColumn::PassMean {
            pass: pass_of.get(idx[0]).copied().unwrap_or(0),
        });
    } else {
        cols.push(DetrendColumn::GlobalMean);
    }
    raw.push(vec![1.0; idx.len()]);

    let add_col = |raw: &mut Vec<Vec<f64>>,
                   cols: &mut Vec<DetrendColumn>,
                   col: DetrendColumn,
                   values: Vec<f64>| {
        let values = standardize_col(&values);
        let trial = append_design(raw, &values, &idx_weights(&w0, &idx));
        if trial > COND_DROP {
            return;
        }
        raw.push(values);
        cols.push(col);
    };

    if intra {
        // Prefer covariate over τ if well-conditioned; never both.
        let mut used_cov = false;
        if want_phase {
            if let Some(ph) = phi {
                let lam: Vec<f64> = idx.iter().map(|&i| (1.0 + ph[i].cos()) / 2.0).collect();
                let before = cols.len();
                add_col(&mut raw, &mut cols, DetrendColumn::PhaseLambert, lam);
                used_cov = cols.len() > before;
            }
        } else if want_elev {
            if let Some(el) = elev {
                let ev: Vec<f64> = idx.iter().map(|&i| el[i]).collect();
                let before = cols.len();
                add_col(&mut raw, &mut cols, DetrendColumn::Elevation, ev);
                used_cov = cols.len() > before;
            }
        }
        if !used_cov && !matches!(mode, DetrendMode::LinearTime) {
            let tau: Vec<f64> = idx
                .iter()
                .map(|&i| {
                    let p = pass_of.get(i).and_then(|&p| passes.get(p));
                    t[i] - p.map(|pp| pp.t_start_s).unwrap_or(t[idx[0]])
                })
                .collect();
            add_col(&mut raw, &mut cols, DetrendColumn::Tau, tau);
        }
    } else {
        // Inter/Full: global t always (except explicit None already handled).
        if !matches!(mode, DetrendMode::PhaseFunction | DetrendMode::Elevation)
            || matches!(mode, DetrendMode::Auto | DetrendMode::LinearTime)
        {
            let gt: Vec<f64> = idx.iter().map(|&i| t[i]).collect();
            add_col(&mut raw, &mut cols, DetrendColumn::GlobalTime, gt);
        }
        let tau: Vec<f64> = idx
            .iter()
            .map(|&i| {
                let p = pass_of.get(i).and_then(|&p| passes.get(p));
                t[i] - p.map(|pp| pp.t_start_s).unwrap_or(t[idx[0]])
            })
            .collect();
        add_col(&mut raw, &mut cols, DetrendColumn::Tau, tau);
        if want_phase {
            if let Some(ph) = phi {
                let lam: Vec<f64> = idx.iter().map(|&i| (1.0 + ph[i].cos()) / 2.0).collect();
                add_col(&mut raw, &mut cols, DetrendColumn::PhaseLambert, lam);
            }
        } else if want_elev {
            if let Some(el) = elev {
                let ev: Vec<f64> = idx.iter().map(|&i| el[i]).collect();
                add_col(&mut raw, &mut cols, DetrendColumn::Elevation, ev);
            }
        }
    }

    let yw: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
    let ww: Vec<f64> = idx_weights(&w0, &idx);
    match irls_huber(&raw, &yw, &ww) {
        Some((coeffs, cond)) => {
            let fitted = apply_design(&raw, &coeffs);
            let resid: Vec<f64> = yw.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();
            DetrendResult {
                y: resid,
                report: DetrendReport {
                    mode,
                    scale,
                    n_beta: cols.len(),
                    columns: cols,
                    coeffs,
                    cond,
                    floating_mean_gls: false,
                    fallback: false,
                    pass_mean,
                },
            }
        }
        None => {
            // Fallback: Intra = this-pass mean; Inter = global mean. Never per-pass on Inter.
            let mean = if intra {
                pass_mean
                    .get(pass_of.get(idx[0]).copied().unwrap_or(0))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                let sw: f64 = ww.iter().sum();
                if sw > 0.0 {
                    yw.iter()
                        .zip(ww.iter())
                        .map(|(yi, wi)| yi * wi)
                        .sum::<f64>()
                        / sw
                } else {
                    0.0
                }
            };
            DetrendResult {
                y: yw.iter().map(|yi| yi - mean).collect(),
                report: DetrendReport {
                    mode,
                    scale,
                    columns: if intra {
                        vec![DetrendColumn::PassMean {
                            pass: pass_of.get(idx[0]).copied().unwrap_or(0),
                        }]
                    } else {
                        vec![DetrendColumn::GlobalMean]
                    },
                    coeffs: vec![mean],
                    cond: f64::INFINITY,
                    n_beta: 1,
                    floating_mean_gls: false,
                    fallback: true,
                    pass_mean,
                },
            }
        }
    }
}

pub fn assign_passes(t: &[f64], passes: &[Pass]) -> Vec<usize> {
    t.iter()
        .map(|&ti| {
            passes
                .iter()
                .position(|p| ti >= p.t_start_s && ti <= p.t_end_s)
                .unwrap_or(0)
        })
        .collect()
}

pub fn pass_index_lists(t: &[f64], passes: &[Pass]) -> Vec<Vec<usize>> {
    let of = assign_passes(t, passes);
    let mut lists = vec![Vec::new(); passes.len().max(1)];
    for (i, p) in of.into_iter().enumerate() {
        if p < lists.len() {
            lists[p].push(i);
        }
    }
    lists
}

fn idx_weights(w: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| w[i]).collect()
}

fn standardize_col(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let s = var.sqrt().max(1e-12);
    v.iter().map(|x| (x - mean) / s).collect()
}

fn append_design(existing: &[Vec<f64>], new_col: &[f64], w: &[f64]) -> f64 {
    let k = existing.len() + 1;
    let n = new_col.len();
    let mut x = DMatrix::<f64>::zeros(n, k);
    for j in 0..existing.len() {
        for i in 0..n {
            x[(i, j)] = existing[j][i];
        }
    }
    for i in 0..n {
        x[(i, k - 1)] = new_col[i];
    }
    cond_xtwx(&x, w)
}

fn cond_xtwx(x: &DMatrix<f64>, w: &[f64]) -> f64 {
    let k = x.ncols();
    let n = x.nrows();
    let mut a = DMatrix::<f64>::zeros(k, k);
    for i in 0..n {
        let wi = w[i];
        for p in 0..k {
            for q in 0..=p {
                let v = wi * x[(i, p)] * x[(i, q)];
                a[(p, q)] += v;
                if p != q {
                    a[(q, p)] += v;
                }
            }
        }
    }
    match a.symmetric_eigen() {
        ev => {
            let vals = ev.eigenvalues;
            let mut lo = f64::INFINITY;
            let mut hi = 0.0_f64;
            for v in vals.iter() {
                let av = v.abs();
                if av > hi {
                    hi = av;
                }
                if av < lo {
                    lo = av;
                }
            }
            if lo <= 0.0 {
                f64::INFINITY
            } else {
                hi / lo
            }
        }
    }
}

fn irls_huber(cols: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<(Vec<f64>, f64)> {
    let (beta, cond) = wls(cols, y, w)?;
    if cond > COND_FAIL {
        return None;
    }
    let fit = apply_design(cols, &beta);
    let resid: Vec<f64> = y.iter().zip(fit.iter()).map(|(a, b)| a - b).collect();
    let mut abs_dev: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
    abs_dev.sort_by(|a, b| a.total_cmp(b));
    let mad = abs_dev[abs_dev.len() / 2];
    let s = (1.4826 * mad).max(1e-12);
    let w2: Vec<f64> = w
        .iter()
        .zip(resid.iter())
        .map(|(wi, r)| {
            let u = (r / s).abs();
            let h = if u <= HUBER_C { 1.0 } else { HUBER_C / u };
            wi * h
        })
        .collect();
    let (beta2, cond2) = wls(cols, y, &w2)?;
    if cond2 > COND_FAIL {
        return None;
    }
    Some((beta2, cond2))
}

fn wls(cols: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<(Vec<f64>, f64)> {
    let k = cols.len();
    let n = y.len();
    if k == 0 || n == 0 {
        return None;
    }
    let mut x = DMatrix::<f64>::zeros(n, k);
    for j in 0..k {
        for i in 0..n {
            x[(i, j)] = cols[j][i];
        }
    }
    let cond = cond_xtwx(&x, w);
    let mut xtwx = DMatrix::<f64>::zeros(k, k);
    let mut xty = DVector::<f64>::zeros(k);
    for i in 0..n {
        let wi = w[i];
        for p in 0..k {
            xty[p] += wi * x[(i, p)] * y[i];
            for q in 0..=p {
                let v = wi * x[(i, p)] * x[(i, q)];
                xtwx[(p, q)] += v;
                if p != q {
                    xtwx[(q, p)] += v;
                }
            }
        }
    }
    let beta = xtwx.lu().solve(&xty)?;
    Some((beta.iter().copied().collect(), cond))
}

fn apply_design(cols: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut out = vec![0.0; n];
    for (j, col) in cols.iter().enumerate() {
        let b = beta.get(j).copied().unwrap_or(0.0);
        for i in 0..n {
            out[i] += b * col[i];
        }
    }
    out
}
