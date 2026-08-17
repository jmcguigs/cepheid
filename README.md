# Cepheid

`cepheid` is a Rust library for analyzing lightcurve data (optical photometry
and, after ingest, RF power). The product period API is
`assess_periodicity` (generalized Lomb–Scargle + PDM, with a stable-satellite
null). Range/phase normalization for visual magnitudes is also included.

Default knobs (frozen; do not retune a failing fixture): Baluev α = 1e-3,
B = 200 local permutations, window_ratio = 1.3, occupancy 0.4, pass gap
factor 8 / 60 s. See `tests/periodicity_ssa.rs`.