// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// RustQuant: A Rust library for quantitative finance tools.
// Copyright (C) 2023 https://github.com/avhz
// Dual licensed under Apache 2.0 and MIT.
// See:
//      - LICENSE-APACHE.md
//      - LICENSE-MIT.md
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//! VaR backtests: Kupiec's proportion-of-failures test, Christoffersen's
//! independence test, and the Basel Committee traffic light, plus the
//! Maximum Undetectable Lie diagnostic: the largest uniform understatement
//! of a reported VaR series that still passes a chosen backtest panel.
//!
//! Sources for the three standard tests:
//!
//! - Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
//!   measurement models. Journal of Derivatives, 3(2), 73-84.
//! - Christoffersen, P. F. (1998). Evaluating interval forecasts.
//!   International Economic Review, 39(4), 841-862.
//! - Basel Committee on Banking Supervision (1996). Supervisory framework
//!   for the use of "backtesting" in conjunction with the internal models
//!   approach to market risk capital requirements. Table 2.
//!
//! The Maximum Undetectable Lie diagnostic is from Gupta, S., "The Maximum
//! Undetectable Lie: Constructing the Worst Value-at-Risk Series That a
//! Backtest Panel Passes and the Limitations of E-Backtesting", paper and
//! reference implementation at
//! <https://github.com/sarthakguptaquant/gaming-oracle>.

use crate::distributions::{Binomial, ChiSquared, Distribution, Gaussian};
use RustQuant_error::RustQuantError;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// STRUCTS AND ENUMS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Result of [`kupiec_pof`] or [`christoffersen_independence`]: a
/// likelihood ratio statistic, distributed chi-squared with one degree of
/// freedom under the null.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LikelihoodRatioTest {
    /// The likelihood ratio statistic.
    pub statistic: f64,
    /// The p-value against the chi-squared(1) null distribution.
    pub p_value: f64,
}

/// Basel Committee traffic light zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrafficLightZone {
    /// Cumulative exception probability below 95%.
    Green,
    /// Cumulative exception probability between 95% and 99.99%.
    Yellow,
    /// Cumulative exception probability at or above 99.99%.
    Red,
}

/// Result of [`basel_traffic_light`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrafficLightResult {
    /// The zone of the exception count.
    pub zone: TrafficLightZone,
    /// Number of exceptions in the window.
    pub exceptions: usize,
    /// Number of observations in the window.
    pub observations: usize,
    /// Probability of observing at most `exceptions` exceptions if the VaR
    /// is correctly calibrated.
    pub cumulative_probability: f64,
}

/// A backtest that [`max_undetectable_understatement`] can be asked to
/// pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarBacktest {
    /// [`kupiec_pof`].
    Kupiec,
    /// [`christoffersen_independence`].
    Independence,
    /// [`basel_traffic_light`].
    TrafficLight,
}

/// Result of [`max_undetectable_understatement`].
#[derive(Clone, Debug, PartialEq)]
pub struct MaxUndetectableUnderstatement {
    /// The largest fraction `s` such that scaling the reported VaR by
    /// `1.0 - s` still passes every selected backtest. `NaN` if the
    /// reported VaR does not pass to begin with.
    pub shrink: f64,
    /// Average reduction in VaR at the limit, `shrink * mean(var)`.
    pub mean_understatement: f64,
    /// Exceptions produced by the reported VaR.
    pub exceptions_reported: usize,
    /// Exceptions produced by the VaR scaled by `1.0 - shrink`.
    pub exceptions_at_limit: usize,
    /// Whether the reported VaR passes every selected backtest.
    pub baseline_passes: bool,
    /// Backtests that reject the reported VaR, empty if it passes.
    pub failed_at_baseline: Vec<VarBacktest>,
    /// Backtests that reject the next, slightly smaller understatement
    /// beyond the limit. Empty if the VaR can be scaled all the way to
    /// zero.
    pub binding: Vec<VarBacktest>,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// FUNCTIONS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fn check_prob(prob: f64) -> Result<(), RustQuantError> {
    if !(prob > 0.0 && prob < 1.0) {
        return Err(RustQuantError::InvalidArgument(
            "prob must be strictly between 0 and 1".to_string(),
        ));
    }
    Ok(())
}

fn check_hits(hits: &[bool]) -> Result<(), RustQuantError> {
    if hits.is_empty() {
        return Err(RustQuantError::InvalidArgument(
            "hits must contain at least one observation".to_string(),
        ));
    }
    Ok(())
}

// Chi-squared(1) survival function, robust at x = 0. RustQuant's `ChiSquared`
// requires x > 0 for one degree of freedom because its *density* is improper
// there, but the CDF (and therefore the survival function) is well defined
// at 0 for every degree of freedom: P(X <= 0) = 0, so P(X > 0) = 1.
fn chi2_sf_1(x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else {
        1.0 - ChiSquared::new(1).cdf(x)
    }
}

// Chi-squared(1) critical value at size `size`, i.e. the value `c` such
// that P(X > c) = size. A chi-squared(1) variable is the square of a
// standard normal, so this is exact and closed-form via the normal
// quantile function, rather than `ChiSquared::inv_cdf`: that search starts
// from, and stays within a few steps of, `0.5 * df`, so for one degree of
// freedom it cannot reach the right tail this test panel needs (the 95th
// percentile of chi-squared(1) is about 3.84, well outside its reachable
// range).
fn chi2_crit_1(size: f64) -> f64 {
    let z = Gaussian::new(0.0, 1.0).inv_cdf(1.0 - size / 2.0);
    z * z
}

fn bernoulli_loglik(count: f64, n: f64, p: f64) -> f64 {
    // 0 * ln(0) is taken to be 0, so the two degenerate terms are only
    // added when they are not multiplied by zero.
    let mut loglik = 0.0;
    if count > 0.0 {
        loglik += count * p.ln();
    }
    if n - count > 0.0 {
        loglik += (n - count) * (1.0 - p).ln();
    }
    loglik
}

fn kupiec_lr(hits: &[bool], prob: f64) -> f64 {
    let n = hits.len() as f64;
    let count = hits.iter().filter(|&&h| h).count() as f64;
    let phat = count / n;
    let lr = -2.0 * (bernoulli_loglik(count, n, prob) - bernoulli_loglik(count, n, phat));
    lr.max(0.0)
}

/// Indicator of days on which the loss exceeded the reported VaR.
///
/// # Errors
///
/// Returns `RustQuantError::InvalidArgument` if `returns` and `var` have
/// different lengths.
///
/// # Examples
///
/// ```
/// # use RustQuant::math::var_exceptions;
///
/// let hits = var_exceptions(&[-0.03, -0.01, 0.02, -0.02], &[0.02, 0.02, 0.02, 0.02]).unwrap();
///
/// // a loss equal to the VaR is not an exception
/// assert_eq!(hits, vec![true, false, false, false]);
/// ```
pub fn var_exceptions(returns: &[f64], var: &[f64]) -> Result<Vec<bool>, RustQuantError> {
    if returns.len() != var.len() {
        return Err(RustQuantError::InvalidArgument(
            "returns and var must have the same length".to_string(),
        ));
    }

    Ok(returns.iter().zip(var).map(|(r, v)| -r > *v).collect())
}

/// Kupiec's proportion-of-failures (unconditional coverage) test.
///
/// Tests whether the exception rate in `hits` is consistent with the
/// nominal VaR probability `prob`. The likelihood ratio is distributed
/// chi-squared with one degree of freedom under the null. The test is
/// two-sided, so it also rejects a VaR series with too few exceptions, not
/// only too many.
///
/// Source: Kupiec, P. H. (1995). Techniques for verifying the accuracy of
/// risk measurement models. Journal of Derivatives, 3(2), 73-84.
///
/// # Errors
///
/// Returns `RustQuantError::InvalidArgument` if `prob` is not strictly
/// between 0 and 1, or if `hits` is empty.
///
/// # Examples
///
/// ```
/// # use RustQuant::math::kupiec_pof;
///
/// let hits = vec![false; 250];
/// let test = kupiec_pof(&hits, 0.01).unwrap();
///
/// assert!(test.p_value > 0.05);
/// ```
pub fn kupiec_pof(hits: &[bool], prob: f64) -> Result<LikelihoodRatioTest, RustQuantError> {
    check_prob(prob)?;
    check_hits(hits)?;

    let statistic = kupiec_lr(hits, prob);

    Ok(LikelihoodRatioTest {
        statistic,
        p_value: chi2_sf_1(statistic),
    })
}

fn independence_lr(hits: &[bool]) -> f64 {
    let (mut n00, mut n01, mut n10, mut n11) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for pair in hits.windows(2) {
        match (pair[0], pair[1]) {
            (false, false) => n00 += 1.0,
            (false, true) => n01 += 1.0,
            (true, false) => n10 += 1.0,
            (true, true) => n11 += 1.0,
        }
    }

    let total = n00 + n01 + n10 + n11;
    if total == 0.0 {
        return 0.0;
    }

    let pi = (n01 + n11) / total;
    let ll_null = bernoulli_loglik(n01 + n11, total, pi);

    let mut ll_alt = 0.0;
    if n00 + n01 > 0.0 {
        ll_alt += bernoulli_loglik(n01, n00 + n01, n01 / (n00 + n01));
    }
    if n10 + n11 > 0.0 {
        ll_alt += bernoulli_loglik(n11, n10 + n11, n11 / (n10 + n11));
    }

    (-2.0 * (ll_null - ll_alt)).max(0.0)
}

/// Christoffersen's test that VaR exceptions are serially independent.
///
/// The alternative is a first-order Markov chain for the exception
/// indicator, so a VaR model that clusters its breaches (several bad days
/// in a row) can still pass [`kupiec_pof`] if the total count is right.
/// The likelihood ratio is distributed chi-squared with one degree of
/// freedom under the null.
///
/// Source: Christoffersen, P. F. (1998). Evaluating interval forecasts.
/// International Economic Review, 39(4), 841-862.
///
/// # Errors
///
/// Returns `RustQuantError::InvalidArgument` if `hits` is empty.
///
/// # Examples
///
/// ```
/// # use RustQuant::math::christoffersen_independence;
///
/// let hits = vec![false; 250];
/// let test = christoffersen_independence(&hits).unwrap();
///
/// assert_eq!(test.statistic, 0.0);
/// ```
pub fn christoffersen_independence(hits: &[bool]) -> Result<LikelihoodRatioTest, RustQuantError> {
    check_hits(hits)?;

    let statistic = independence_lr(hits);

    Ok(LikelihoodRatioTest {
        statistic,
        p_value: chi2_sf_1(statistic),
    })
}

/// Basel Committee traffic light zone for the most recent observations.
///
/// The yellow zone begins where the cumulative binomial probability of the
/// exception count reaches 95% and the red zone where it reaches 99.99%.
/// For 250 observations and `prob = 0.01` this gives green for 0 to 4
/// exceptions, yellow for 5 to 9 and red for 10 or more.
///
/// `window` selects the number of most recent observations of `hits` to
/// use; `None` uses all of them.
///
/// Source: Basel Committee on Banking Supervision (1996). Supervisory
/// framework for the use of "backtesting" in conjunction with the internal
/// models approach to market risk capital requirements. Table 2.
///
/// # Errors
///
/// Returns `RustQuantError::InvalidArgument` if `prob` is not strictly
/// between 0 and 1, if `hits` is empty, or if `window` is `Some(0)`.
///
/// # Examples
///
/// ```
/// # use RustQuant::math::{basel_traffic_light, TrafficLightZone};
///
/// let hits = vec![false; 250];
/// let result = basel_traffic_light(&hits, 0.01, Some(250)).unwrap();
///
/// assert_eq!(result.zone, TrafficLightZone::Green);
/// ```
pub fn basel_traffic_light(
    hits: &[bool],
    prob: f64,
    window: Option<usize>,
) -> Result<TrafficLightResult, RustQuantError> {
    check_prob(prob)?;
    check_hits(hits)?;
    if window == Some(0) {
        return Err(RustQuantError::InvalidArgument(
            "window must be positive or None".to_string(),
        ));
    }

    let windowed = match window {
        Some(w) if w < hits.len() => &hits[hits.len() - w..],
        _ => hits,
    };
    let observations = windowed.len();
    let exceptions = windowed.iter().filter(|&&h| h).count();
    let cumulative_probability = Binomial::new(observations, prob).cdf(exceptions as f64);

    let zone = if cumulative_probability >= 0.9999 {
        TrafficLightZone::Red
    } else if cumulative_probability >= 0.95 {
        TrafficLightZone::Yellow
    } else {
        TrafficLightZone::Green
    };

    Ok(TrafficLightResult {
        zone,
        exceptions,
        observations,
        cumulative_probability,
    })
}

#[allow(clippy::too_many_arguments)]
fn gate_failures(
    hits: &[bool],
    prob: f64,
    crit: f64,
    tests: &[VarBacktest],
    ignore_too_few: bool,
    window: Option<usize>,
) -> Vec<VarBacktest> {
    let mut failed = Vec::new();

    if tests.contains(&VarBacktest::Kupiec) {
        let lr = kupiec_lr(hits, prob);
        let too_few = (hits.iter().filter(|&&h| h).count() as f64) / (hits.len() as f64) < prob;
        if lr > crit && !(ignore_too_few && too_few) {
            failed.push(VarBacktest::Kupiec);
        }
    }
    if tests.contains(&VarBacktest::Independence) && independence_lr(hits) > crit {
        failed.push(VarBacktest::Independence);
    }
    if tests.contains(&VarBacktest::TrafficLight) {
        let zone = basel_traffic_light(hits, prob, window)
            .expect("prob and hits are already validated by the caller")
            .zone;
        if zone != TrafficLightZone::Green {
            failed.push(VarBacktest::TrafficLight);
        }
    }

    failed
}

/// Largest uniform understatement of a VaR series that still passes a
/// chosen backtest panel: the Maximum Undetectable Lie diagnostic.
///
/// A passing backtest only says that the reported VaR was not rejected. It
/// does not say how close it was. This function reports how much lower the
/// whole VaR series could have been scaled, with the same `returns`, and
/// still not have been rejected by any test in `tests`.
///
/// Because every test in `tests` reads `var` only through its exception
/// indicator, the exception set for a proportional scaling
/// `(1.0 - s) * var` changes only at the breakpoints `s = 1 - loss / var`.
/// The search evaluates the panel once per breakpoint, so it is exact for
/// this family of scalings; it is a lower bound on what a non-proportional
/// (day-by-day) rescaling could hide.
///
/// `ignore_too_few`, when true, means a Kupiec rejection caused by *too
/// few* exceptions does not count as a failure: fewer exceptions than
/// expected is not evidence of an understated VaR, so it should not block
/// the search for one.
///
/// Source: Gupta, S., "The Maximum Undetectable Lie: Constructing the
/// Worst Value-at-Risk Series That a Backtest Panel Passes and the
/// Limitations of E-Backtesting", paper and reference implementation at
/// <https://github.com/sarthakguptaquant/gaming-oracle>.
///
/// # Errors
///
/// Returns `RustQuantError::InvalidArgument` if `returns` and `var` have
/// different lengths or are empty, if any `var` value is not finite and
/// strictly positive, if `prob` or `size` is not strictly between 0 and 1,
/// or if `tests` is empty.
///
/// # Examples
///
/// ```
/// # use RustQuant::math::{max_undetectable_understatement, VarBacktest};
///
/// let returns = vec![-0.001; 300];
/// let var = vec![1.0; 300];
/// let tests = [VarBacktest::Kupiec, VarBacktest::Independence, VarBacktest::TrafficLight];
///
/// let result =
///     max_undetectable_understatement(&returns, &var, 0.01, 0.05, &tests, true, Some(250))
///         .unwrap();
///
/// assert!(result.baseline_passes);
/// // a series that never breaches can be scaled almost to zero undetected
/// assert!(result.shrink > 0.9);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn max_undetectable_understatement(
    returns: &[f64],
    var: &[f64],
    prob: f64,
    size: f64,
    tests: &[VarBacktest],
    ignore_too_few: bool,
    window: Option<usize>,
) -> Result<MaxUndetectableUnderstatement, RustQuantError> {
    if returns.len() != var.len() || returns.is_empty() {
        return Err(RustQuantError::InvalidArgument(
            "returns and var must be non-empty and the same length".to_string(),
        ));
    }
    if var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(RustQuantError::InvalidArgument(
            "var must be finite and strictly positive".to_string(),
        ));
    }
    check_prob(prob)?;
    if !(size > 0.0 && size < 1.0) {
        return Err(RustQuantError::InvalidArgument(
            "size must be strictly between 0 and 1".to_string(),
        ));
    }
    if tests.is_empty() {
        return Err(RustQuantError::InvalidArgument(
            "tests must be non-empty".to_string(),
        ));
    }

    // Critical value shared by the Kupiec and independence gates; computed
    // once, not on every breakpoint.
    let crit = chi2_crit_1(size);

    let ratio: Vec<f64> = returns.iter().zip(var).map(|(r, v)| -r / v).collect();
    let reported: Vec<bool> = ratio.iter().map(|&r| r > 1.0).collect();
    let n_reported = reported.iter().filter(|&&h| h).count();

    let failed0 = gate_failures(&reported, prob, crit, tests, ignore_too_few, window);
    if !failed0.is_empty() {
        return Ok(MaxUndetectableUnderstatement {
            shrink: f64::NAN,
            mean_understatement: f64::NAN,
            exceptions_reported: n_reported,
            exceptions_at_limit: n_reported,
            baseline_passes: false,
            failed_at_baseline: failed0.clone(),
            binding: failed0,
        });
    }

    // The exception set is constant between breakpoints, so scanning them
    // in decreasing order and stopping at the first pass finds the exact
    // limit with one evaluation of the panel per breakpoint.
    let mut thresholds: Vec<f64> = ratio
        .iter()
        .copied()
        .filter(|&r| r > 0.0 && r < 1.0)
        .collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).expect("ratios are finite"));
    thresholds.dedup();

    let mut candidates = vec![0.0];
    candidates.extend(thresholds.iter().copied());
    candidates.push(1.0);

    let mut limit_idx = candidates.len() - 1;
    for (i, &thr) in candidates[..candidates.len() - 1].iter().enumerate() {
        let hits: Vec<bool> = ratio.iter().map(|&r| r > thr).collect();
        if gate_failures(&hits, prob, crit, tests, ignore_too_few, window).is_empty() {
            limit_idx = i;
            break;
        }
    }

    let limit = candidates[limit_idx];
    let binding = if limit_idx > 0 {
        let hits: Vec<bool> = ratio.iter().map(|&r| r > candidates[limit_idx - 1]).collect();
        gate_failures(&hits, prob, crit, tests, ignore_too_few, window)
    } else {
        Vec::new()
    };

    let shrink = 1.0 - limit;
    let var_mean = var.iter().sum::<f64>() / var.len() as f64;
    let exceptions_at_limit = ratio.iter().filter(|&&r| r > limit).count();

    Ok(MaxUndetectableUnderstatement {
        shrink,
        mean_understatement: shrink * var_mean,
        exceptions_reported: n_reported,
        exceptions_at_limit,
        baseline_passes: true,
        failed_at_baseline: Vec::new(),
        binding,
    })
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UNIT TESTS
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod tests_var_backtest {
    use super::*;
    use RustQuant_utils::assert_approx_equal;

    const CRIT_1: f64 = 3.841_458_820_694_124; // qchisq(0.95, df = 1)

    fn hits_at(nobs: usize, positions: &[usize]) -> Vec<bool> {
        let mut h = vec![false; nobs];
        for &p in positions {
            h[p] = true;
        }
        h
    }

    fn spaced_hits(nobs: usize, count: usize) -> Vec<bool> {
        if count == 0 {
            return vec![false; nobs];
        }
        let positions: Vec<usize> = (0..count)
            .map(|i| ((i * (nobs - 1)) as f64 / (count - 1).max(1) as f64).round() as usize)
            .collect();
        hits_at(nobs, &positions)
    }

    #[test]
    fn test_var_exceptions() {
        let ret = [-0.03, -0.01, 0.02, -0.02];
        let var = [0.02, 0.02, 0.02, 0.02];
        assert_eq!(
            var_exceptions(&ret, &var).unwrap(),
            vec![true, false, false, false]
        );
        assert!(var_exceptions(&ret, &var[..2]).is_err());
    }

    #[test]
    fn test_kupiec_nonrejection_region_255() {
        // Kupiec (1995, J. Derivatives 3(2), Table 1), 95% nonrejection
        // region for a 99% VaR at T = 255: N in [1, 6].
        let accepted: Vec<usize> = (0..=255)
            .filter(|&x| kupiec_pof(&spaced_hits(255, x), 0.01).unwrap().statistic <= CRIT_1)
            .collect();
        assert_eq!(*accepted.iter().min().unwrap(), 1);
        assert_eq!(*accepted.iter().max().unwrap(), 6);
    }

    #[test]
    fn test_kupiec_zero_exceptions_closed_form() {
        let test = kupiec_pof(&vec![false; 250], 0.01).unwrap();
        assert_approx_equal!(test.statistic, -2.0 * 250.0 * 0.99_f64.ln(), 1e-10);
    }

    #[test]
    fn test_kupiec_all_exceptions() {
        let test = kupiec_pof(&vec![true; 10], 0.1).unwrap();
        assert_approx_equal!(test.statistic, -2.0 * 10.0 * 0.1_f64.ln(), 1e-10);
    }

    #[test]
    fn test_kupiec_exact_calibration_is_zero() {
        let test = kupiec_pof(&spaced_hits(200, 2), 0.01).unwrap();
        assert_approx_equal!(test.statistic, 0.0, 1e-10);
        assert_approx_equal!(test.p_value, 1.0, 1e-10);
    }

    #[test]
    fn test_kupiec_input_validation() {
        assert!(kupiec_pof(&vec![false; 10], 0.0).is_err());
        assert!(kupiec_pof(&[], 0.01).is_err());
    }

    #[test]
    fn test_christoffersen_hand_computed() {
        // 20 observations, exceptions at positions 3, 4, 10.
        let hits = hits_at(20, &[3, 4, 10]);
        // Transitions over the 19 consecutive pairs:
        // 0->0: 14, 0->1: 2 (into 3 and into 10), 1->0: 2 (out of 4, out of 10), 1->1: 1 (3->4).
        let (n00, n01, n10, n11) = (14.0_f64, 2.0_f64, 2.0_f64, 1.0_f64);
        let pi01 = n01 / (n00 + n01);
        let pi11 = n11 / (n10 + n11);
        let pi = (n01 + n11) / 19.0;
        let ll_alt = n00 * (1.0 - pi01).ln()
            + n01 * pi01.ln()
            + n10 * (1.0 - pi11).ln()
            + n11 * pi11.ln();
        let ll_null = (n00 + n10) * (1.0 - pi).ln() + (n01 + n11) * pi.ln();
        let expected = -2.0 * (ll_null - ll_alt);

        let test = christoffersen_independence(&hits).unwrap();
        assert_approx_equal!(test.statistic, expected, 1e-10);
    }

    #[test]
    fn test_christoffersen_clustering_is_rejected_spacing_is_not() {
        let nobs = 500;
        let spaced = spaced_hits(nobs, 5);
        let clustered = hits_at(nobs, &[100, 101, 102, 103, 104]);
        assert!(christoffersen_independence(&spaced).unwrap().statistic < CRIT_1);
        assert!(christoffersen_independence(&clustered).unwrap().statistic > CRIT_1);
        // same exception count, so Kupiec cannot tell them apart
        assert_approx_equal!(
            kupiec_pof(&spaced, 0.01).unwrap().statistic,
            kupiec_pof(&clustered, 0.01).unwrap().statistic,
            1e-10
        );
    }

    #[test]
    fn test_christoffersen_no_exceptions_is_zero() {
        assert_eq!(
            christoffersen_independence(&vec![false; 100])
                .unwrap()
                .statistic,
            0.0
        );
    }

    #[test]
    fn test_christoffersen_single_observation() {
        assert_eq!(
            christoffersen_independence(&[true]).unwrap().statistic,
            0.0
        );
    }

    #[test]
    fn test_traffic_light_table_2() {
        // Basel Committee (1996), Table 2: cumulative probability (%) of at
        // most x exceptions in 250 observations at 99% coverage.
        let basel: [(usize, f64); 11] = [
            (0, 8.11),
            (1, 28.58),
            (2, 54.32),
            (3, 75.81),
            (4, 89.22),
            (5, 95.88),
            (6, 98.63),
            (7, 99.60),
            (8, 99.89),
            (9, 99.97),
            (10, 99.99),
        ];
        for (count, expected_pct) in basel {
            let hits = spaced_hits(250, count);
            let result = basel_traffic_light(&hits, 0.01, None).unwrap();
            assert_approx_equal!(100.0 * result.cumulative_probability, expected_pct, 0.01);
            let expected_zone = if count <= 4 {
                TrafficLightZone::Green
            } else if count <= 9 {
                TrafficLightZone::Yellow
            } else {
                TrafficLightZone::Red
            };
            assert_eq!(result.zone, expected_zone);
            assert_eq!(result.exceptions, count);
            assert_eq!(result.observations, 250);
        }
    }

    #[test]
    fn test_traffic_light_only_last_window_used() {
        let mut hits = vec![true; 30];
        hits.extend(vec![false; 250]);
        assert_eq!(
            basel_traffic_light(&hits, 0.01, Some(250)).unwrap().zone,
            TrafficLightZone::Green
        );
        assert_eq!(
            basel_traffic_light(&hits, 0.01, None).unwrap().zone,
            TrafficLightZone::Red
        );
    }

    #[test]
    fn test_traffic_light_invalid_window() {
        assert!(basel_traffic_light(&vec![false; 10], 0.01, Some(0)).is_err());
    }

    fn mul_sim() -> (Vec<f64>, Vec<f64>) {
        // A deterministic, mildly heteroskedastic series: large enough and
        // varied enough to exercise the breakpoint search without pulling
        // in an RNG dependency for the test.
        let nobs = 400;
        let mut returns = Vec::with_capacity(nobs);
        let mut var = Vec::with_capacity(nobs);
        for i in 0..nobs {
            let t = i as f64;
            let sigma = 0.01 * (1.0 + 0.3 * (0.05 * t).sin().abs());
            // A pseudo-random-looking but fully deterministic residual.
            let u = ((t * 12.9898).sin() * 43758.5453).fract();
            let z = (u - 0.5) * 3.4; // roughly in [-1.7, 1.7]
            returns.push(sigma * z);
            var.push(2.5 * sigma); // reported VaR comfortably above typical moves
        }
        (returns, var)
    }

    const ALL_TESTS: [VarBacktest; 3] = [
        VarBacktest::Kupiec,
        VarBacktest::Independence,
        VarBacktest::TrafficLight,
    ];

    #[test]
    fn test_mul_passes_and_is_consistent() {
        let (ret, var) = mul_sim();
        let result =
            max_undetectable_understatement(&ret, &var, 0.01, 0.05, &ALL_TESTS, true, Some(250))
                .unwrap();
        assert!(result.baseline_passes);
        assert!((0.0..=1.0).contains(&result.shrink));
        assert!(result.exceptions_at_limit >= result.exceptions_reported);
        assert_approx_equal!(
            result.mean_understatement,
            result.shrink * var.iter().sum::<f64>() / var.len() as f64,
            1e-10
        );
    }

    #[test]
    fn test_mul_brute_force_grid() {
        let (ret, var) = mul_sim();
        let result =
            max_undetectable_understatement(&ret, &var, 0.01, 0.05, &ALL_TESTS, true, Some(250))
                .unwrap();

        let passes = |s: f64| -> bool {
            let scaled: Vec<f64> = var.iter().map(|v| (1.0 - s) * v).collect();
            let hits = var_exceptions(&ret, &scaled).unwrap();
            let ind_ok = christoffersen_independence(&hits).unwrap().statistic <= CRIT_1;
            let tl_ok = basel_traffic_light(&hits, 0.01, Some(250)).unwrap().zone
                == TrafficLightZone::Green;
            let mean_hits = hits.iter().filter(|&&h| h).count() as f64 / hits.len() as f64;
            let kup_ok = kupiec_pof(&hits, 0.01).unwrap().statistic <= CRIT_1 || mean_hits < 0.01;
            ind_ok && tl_ok && kup_ok
        };

        let steps = 2000;
        let mut best = 0.0_f64;
        for i in 0..=steps {
            let s = i as f64 / steps as f64;
            if passes(s) {
                best = s;
            }
        }
        // exact limit is at or above the best grid point and within one cell
        assert!(result.shrink >= best - 1e-9);
        assert!(result.shrink - best <= 1.0 / steps as f64 + 1e-9);
        assert!(passes(result.shrink * (1.0 - 1e-9)));
    }

    #[test]
    fn test_mul_more_tests_never_increase_limit() {
        let (ret, var) = mul_sim();
        let only_kupiec = max_undetectable_understatement(
            &ret,
            &var,
            0.01,
            0.05,
            &[VarBacktest::Kupiec],
            true,
            Some(250),
        )
        .unwrap();
        let all_tests =
            max_undetectable_understatement(&ret, &var, 0.01, 0.05, &ALL_TESTS, true, Some(250))
                .unwrap();
        assert!(all_tests.shrink <= only_kupiec.shrink + 1e-9);
    }

    #[test]
    fn test_mul_never_losing_series_can_be_understated_fully() {
        let ret = vec![0.001; 300];
        let var = vec![1.0; 300];
        let result =
            max_undetectable_understatement(&ret, &var, 0.01, 0.05, &ALL_TESTS, true, None)
                .unwrap();
        assert_approx_equal!(result.shrink, 1.0, 1e-9);
        assert_eq!(result.exceptions_at_limit, 0);
        assert!(result.binding.is_empty());
    }

    #[test]
    fn test_mul_two_sided_baseline_failure() {
        let ret = vec![0.001; 250];
        let var = vec![1.0; 250];
        let result =
            max_undetectable_understatement(&ret, &var, 0.01, 0.05, &ALL_TESTS, false, None)
                .unwrap();
        assert!(!result.baseline_passes);
        assert!(result.failed_at_baseline.contains(&VarBacktest::Kupiec));
        assert!(result.shrink.is_nan());
    }

    #[test]
    fn test_mul_overstated_reported_var_fails_baseline() {
        let (ret, var) = mul_sim();
        let overstated: Vec<f64> = var.iter().map(|v| 0.2 * v).collect();
        let result = max_undetectable_understatement(
            &ret,
            &overstated,
            0.01,
            0.05,
            &ALL_TESTS,
            true,
            Some(250),
        )
        .unwrap();
        assert!(!result.baseline_passes);
        assert!(result.shrink.is_nan());
    }

    #[test]
    fn test_mul_input_validation() {
        let (ret, var) = mul_sim();
        let zero_var = vec![0.0; var.len()];
        assert!(max_undetectable_understatement(
            &ret, &zero_var, 0.01, 0.05, &ALL_TESTS, true, None
        )
        .is_err());
        assert!(max_undetectable_understatement(&ret, &var, 0.01, 0.05, &[], true, None).is_err());
        assert!(max_undetectable_understatement(
            &ret,
            &var[..var.len() - 1],
            0.01,
            0.05,
            &ALL_TESTS,
            true,
            None
        )
        .is_err());
        assert!(max_undetectable_understatement(
            &ret, &var, 0.01, 1.5, &ALL_TESTS, true, None
        )
        .is_err());
    }
}
