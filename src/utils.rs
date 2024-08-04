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

mod vkcov {
    use std::f64::consts::{     // Using std lib constants
        PI,                     // Pi
        FRAC_PI_2               // Pi / 2
    };

    /// # Precision limit for Bessel computation
    const PRECISION_CONVERGENCE: f64 = 1.0e-8;
    const MAX_ITER_BESSEL: i32 = 500;
    const NU: f64 = 5.0/6.0;
    const GAMMA_P11_6: f64 = 0.9406531400494903; //basic::gamma(5.0/6.0+1.0);
    const GAMMA_N11_6: f64 = 5.5663153388283035; //basic::gamma(5.0/6.0-1.0);
    const FRONT_TERM: f64 = 0.08583068106228546;


    fn i_nu_real_n(z: f64) -> f64 {
        const NU: f64 = -5.0/6.0;
        let z2: f64 = z / 2.0;                                // Halving z
        let mut k: f64 = 0.0;                                       // Order counter
        let mut d1: f64 = 1.0;                                      // First div
        let mut d2: f64 = GAMMA_N11_6; // basic::gamma(nu + 1.0);                   // Second div
        let mut term: f64 = z2.powf(NU) / d2;                 // The term at each step
        let mut sum: f64 = 0.0;              // The result of the operation
        let mut counter: i32 = 0;                                   // Iteration counter
        
        // If the first term is already too small we exit directly
        if term.abs() < PRECISION_CONVERGENCE {
            return sum;
        }

        // Computing the terms of the infinite series
        'convergence: while counter < MAX_ITER_BESSEL {
            
            counter += 1;
            sum += term;

            // If the changed compared to the final value is small we break
            if (term / sum).abs() < PRECISION_CONVERGENCE {
                break 'convergence;
            }

            k += 1.0;                                               // Incrementing value
            d1 *= k;                                                // Next value in the n! term
            d2 *= NU + k;                                           // Next value in the gamma(n+k+1) term
            term = z2.powf(k.mul_add(2.0, NU)) / (d1 * d2);
        }

        sum
    }

    fn i_nu_real_p(z: f64) -> f64 {
        const NU: f64 = 5.0/6.0;
        let z2: f64 = z / 2.0;                                // Halving z
        let mut k: f64 = 0.0;                                       // Order counter
        let mut d1: f64 = 1.0;                                      // First div
        let mut d2: f64 = GAMMA_P11_6; // basic::gamma(nu + 1.0);                   // Second div
        let mut term: f64 = z2.powf(NU) / d2;                 // The term at each step
        let mut sum: f64 = 0.0;              // The result of the operation
        let mut counter: i32 = 0;                                   // Iteration counter
        
        // If the first term is already too small we exit directly
        if term.abs() < PRECISION_CONVERGENCE {
            return sum;
        }

        // Computing the terms of the infinite series
        'convergence: while counter < MAX_ITER_BESSEL {
            
            counter += 1;
            sum += term;

            // If the changed compared to the final value is small we break
            if (term / sum).abs() < PRECISION_CONVERGENCE {
                break 'convergence;
            }

            k += 1.0;                                               // Incrementing value
            d1 *= k;                                                // Next value in the n! term
            d2 *= NU + k;                                           // Next value in the gamma(n+k+1) term
            term = z2.powf(k.mul_add(2.0, NU)) / (d1 * d2);
        }

        sum
    }

    fn k_real(z: f64) -> f64 {
        (FRAC_PI_2 / (NU * PI).sin()) * (i_nu_real_n(z) - i_nu_real_p(z))
    }

    pub const FIVESIXTHS: f64 = 5.0/6.0;
    
    pub fn vk_cov(x: f64, r0: f64, l0: f64) -> f64 {
        // computes the rightmost two terms of Eqn (2.24)
        let x_safe = (x + 1e-9) * (2.0 * PI / l0);
        let y = k_real(x_safe);
        (l0 / r0).powf(5.0/3.0) * FRONT_TERM * y * x_safe.powf(FIVESIXTHS)
    }
}

pub use vkcov::vk_cov;