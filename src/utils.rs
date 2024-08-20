//! Convenience functions and utilities for rao.

/// Convenience function for converting a given `coupling` and `pitch` to
/// `sigma` in the creation of Gaussian influence functions.
/// 
/// In the classical exponential form, the influence function has a value
/// of `coupling` at a distance of `pitch` from the centre of the actuator.
/// If this coupling decays exponentially with the square of the distance,
/// then the influence function can be modelled as:
/// ```
/// fn influ_exp(coupling: f64, pitch: f64, x: f64) -> f64 {
///     coupling.powf((x/pitch).powf(2.0))
/// }
/// ```
/// which is equivalent to:
/// ```
/// fn influ_gauss(sigma: f64, x: f64) -> f64{
///     (-0.5*(x/sigma).powf(2.0)).exp()
/// }
/// ```
/// when
/// ```
/// let pitch: f64 = 22.0;  // centimetres, for example
/// let coupling: f64 = 0.4;  // coupling coefficient
/// // compute sigma:
/// let sigma = pitch/(1.0/coupling).ln().powf(0.5)/(2.0_f64).powf(0.5);
/// assert_eq!(rao::coupling_to_sigma(coupling, pitch), sigma);
/// ```
/// This function performs that error-prone computation.
#[must_use] pub fn coupling_to_sigma(coupling: f64, pitch: f64) -> f64
{
    pitch/(1.0/coupling).ln().powf(0.5)/(2.0_f64).powf(0.5)
}

/// Evaluate the [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function)
/// in it's base form `exp(-0.5*x^2))`.
#[must_use] pub fn gaussian(x: f64) -> f64 {
    (-0.5*(x).powf(2.0)).exp()
}
    
#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq};
    #[test]
    fn coupling_conversion() {
        assert_abs_diff_eq!(coupling_to_sigma((-1.0_f64).exp(), 1.0).powf(2.0), 0.5);
    }
}

mod vkcov_approx {
    use std::f64::consts::TAU; // 2 * PI

    const FRONT_TERM: f64 = 0.085_830_681_062_285_46;

    #[must_use] pub fn vk_cov(x: f64, r0: f64, l0: f64) -> f64 {
        // computes the rightmost two terms of Eqn (2.24)
        // let x_safe = (x + 1e-9) * (2.0 * PI / l0);
        // don't need to be "safe" any more because the approximation is
        // extremely stable for non-negative input.
        let y = btiotb_approx(x * (TAU / l0));
        (l0 / r0).powf(5.0/3.0) * FRONT_TERM * y
    }

    /// see [this gist](https://gist.github.com/jcranney/cfb9f1347c31be3c94c9c4be94f0c1af)
    /// Better Than It Ought To Be approximation
    pub fn btiotb_approx(x: f64) -> f64 {
        1.005_634_9 * (-0.935_996_8 * x).exp() * (1.740_443_6 * x + 1.0) / ( x + 1.0 )
    }
    
}

/// von Karman covariance function
///
/// calculated the covariance between two points separated by a distance `r`
/// for a von Karman layer of turbulence with a specified r0 and L0.
pub use vkcov_approx::vk_cov;