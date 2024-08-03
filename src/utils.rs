use crate::geometry::Vec2D;

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
pub fn coupling_to_sigma(coupling: f64, pitch: f64) -> f64
{
    pitch/(1.0/coupling).ln().powf(0.5)/(2.0_f64).powf(0.5)
}

/// Evaluate the [Gaussian function](https://en.wikipedia.org/wiki/Gaussian_function)
/// in it's base form `exp(-0.5*x^2))`.
pub fn gaussian(x: f64) -> f64 {
    (-0.5*(x).powf(2.0)).exp()
}
/// Evaluate the centrally symmetric 2D Gaussian function.
pub fn gaussian2d(v: Vec2D) -> f64 {
    (-0.5*(v.norm2())).exp()
}
    
#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq};
    #[test]
    fn coupling_conversion() {
        assert_abs_diff_eq!(coupling_to_sigma((-1.0_f64).exp(), 1.0).powf(2.0), 0.5)
    }

    #[test]
    fn gaussian_1d_eq_2d() {
        let x = 1.234;
        let y = 0.0;
        assert!(gaussian(x) > 0.0);
        assert_abs_diff_eq!(gaussian(x),gaussian2d(Vec2D::new(x,y)));
    }
}
