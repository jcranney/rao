//! # rao
//! 
//! `rao` - Adaptive Optics tools in Rust - is a set of fast and robust adaptive
//! optics utilities. The current scope of `rao` is for the calculation of
//! large matrices in AO, used in the ocnfiguration of real-time adaptive optics,
//! control. Specifically, we aim to provide fast, scalable, and reliable APIs for
//! generating:
//!  - `rao::IMat` - the interaction matrix between measurements and actuators,
//!  - `rao::CovMat` - the covariance matrix between random variables (e.g., measurements)
//!
//! These two matrices are the largest computational burden in the configuration
//! of real-time control for AO, and having fast and scalable methods for their
//! computation enables the optimisation of the AO loop [citation needed].
//!
//! # Examples
//! Building an interaction matrix for a square-grid DM and a square-grid SH-WFS:
//! ```
//! use crate::rao::Matrix;
//! const N_SUBX: i32 = 8;  // 8 x 8 subapertures
//! const PITCH: f64 = 0.2;  // 0.2 metres gap between actuators
//! const COUPLING: f64 = 0.5;  // actuator cross-coupling
//! 
//! // build list of measurements
//! let mut measurements = vec![];
//! for i in 0..N_SUBX {
//!     for j in 0..N_SUBX {
//!         let x0 = ((j-N_SUBX/2) as f64 + 0.5)*PITCH;
//!         let y0 = ((i-N_SUBX/2) as f64 + 0.5)*PITCH;
//!         let xz = 0.0;  // angular x-component (radians)
//!         let yz = 0.0;  // angular y-compenent (radians)
//!         // define the optical axis of subaperture
//!         let line = rao::Line::new(x0,xz,y0,yz);
//!         // slope-style measurement
//!         // x-slope
//!         measurements.push(rao::Measurement::Slope{
//!             line: line.clone(),
//!              method: rao::SlopeMethod::TwoEdge{
//!                 edge_separation: PITCH,
//!                 edge_length: PITCH,
//!                 npoints: 5,
//!                 gradient_axis: rao::Vec2D::x_unit(),
//!             }
//!         });
//!         // y-slope
//!         measurements.push(rao::Measurement::Slope{
//!             line: line.clone(),
//!             method: rao::SlopeMethod::TwoEdge{
//!                 edge_separation: PITCH,
//!                 edge_length: PITCH,
//!                 npoints: 5,
//!                 gradient_axis: rao::Vec2D::y_unit(),
//!             }
//!         });
//!     }
//! }
//! 
//! // build list of actuators
//! let mut actuators = vec![];
//! for i in 0..(N_SUBX+1) {
//!     for j in 0..(N_SUBX+1) {
//!         let x = ((j-N_SUBX/2) as f64)*PITCH;
//!         let y = ((i-N_SUBX/2) as f64)*PITCH;
//!         actuators.push(
//!             // Gaussian influence functions
//!             rao::Actuator::Gaussian{
//!                 // std defined by coupling and pitch
//!                 sigma: rao::coupling_to_sigma(COUPLING, PITCH),
//!                 // position of actuator in 3D (z=altitude)
//!                 position: rao::Vec3D::new(x, y, 0.0),
//!             }
//!         );
//!     }
//! }
//! 
//! // instanciate imat from (actu,meas)
//! let imat = rao::IMat::new(&measurements, &actuators);
//! // serialise imat for saving
//! let data: Vec<f64> = imat.flattened_array();
//! ```

use std::fmt;
#[macro_use] extern crate impl_ops;

mod utils;
pub use crate::utils::*;
mod geometry;
pub use crate::geometry::{
    Vec2D,
    Vec3D,
    Line,
};
mod linalg;
pub use crate::linalg::Matrix;

pub trait Sampler {
    fn get_bundle(&self, line: Line) -> Vec<(Line,f64)>;
}


pub trait Measurable {
    fn phase(&self, p: &Line) -> f64;

    fn distance(&self, p: &Line, q: &Line) -> f64;

    fn slope(&self, pos: &Line, neg: &Line) -> f64 {
        let num = self.phase(&pos) - self.phase(&neg);
        let den = self.distance(&pos, &neg);
        num / den
    }
    fn measure(&self, measurement: &Measurement) -> f64 {
        match measurement {
            Measurement::Zero => 0.0,
            Measurement::Phase{line} => self.phase(line),
            Measurement::Slope{line,method} => {
                method.bundle(line)
                .into_iter()
                .map(|(pos,neg,coeff)|
                    self.slope(&pos,&neg)*coeff
                ).sum()
            }
        }
    }
}

pub trait CoSampleable {
    fn eval(&self, p: &Line, q: &Line) -> f64;
    fn distance(&self, p: &Line, q: &Line) -> f64;

    fn cosample(&self, meas_left: &Measurement, meas_right: &Measurement) -> f64 {
        match (meas_left, meas_right) {
            (Measurement::Zero, _) => 0.0,
            (_, Measurement::Zero) => 0.0,
            (
                Measurement::Phase{line: line_left},
                Measurement::Phase{line: line_right},
            ) => self.eval(line_left, line_right),
            (
                Measurement::Phase{line: line_left},
                Measurement::Slope{line: line_right, method: method_right},
            ) => {
                let bundle_right = method_right.bundle(line_right);
                bundle_right
                .into_iter()
                .map(|(pos_right,neg_right,coeff_right)| {
                    let num = self.eval(&pos_right,&line_left) - self.eval(&neg_right,&line_left);
                    let den = self.distance(&pos_right, &neg_right);
                    (num / den) * coeff_right
                }).sum()
            },
            (
                Measurement::Slope{line: line_left, method: method_left},
                Measurement::Phase{line: line_right},
            ) => {
                let bundle_left = method_left.bundle(line_left);
                bundle_left
                .into_iter()
                .map(|(pos_left,neg_left,coeff_left)| {
                    let num = self.eval(&pos_left,&line_right) - self.eval(&neg_left,&line_right);
                    let den = self.distance(&pos_left, &neg_left);
                    (num / den) * coeff_left
                }).sum()
            },
            (
                Measurement::Slope{line: line_left, method: method_left},
                Measurement::Slope{line: line_right, method: method_right},
            ) => {
                let bundle_left = method_left.bundle(line_left);
                let bundle_right = method_right.bundle(line_right);
                bundle_left
                .iter()
                .map(|(pos_left,neg_left,coeff_left)| {
                    bundle_right
                    .iter()
                    .map(|(pos_right,neg_right,coeff_right)| {
                        let mut num = self.eval(&pos_right,&pos_left);
                        num += self.eval(&neg_right,&neg_left);
                        num -= self.eval(&pos_right,&neg_left);
                        num -= self.eval(&neg_right,&pos_left);
                        let den = self.distance(&pos_right, &neg_right)*self.distance(&pos_left, &neg_left);
                        (num / den) * coeff_right * coeff_left
                    }).sum::<f64>()
                }).sum()
            },
        }
    }
}


/// The atomic measurement unit.
/// 
/// A [Measurement] provides a scalar-valued sample of the system. Similar to an
/// [Actuator], a single measurement device (e.g., a Shack Hartmann WFS) might be
/// comprised of many [Measurement]s, e.g., `&[Measurement; N]`.
#[derive(Debug)]
pub enum Measurement{
    /// The null measurement, always returning 0.0 regardless of the measured object.
    Zero,
    /// Phase measurement along a given [Line].
    Phase {
        /// [Line] to trace measurement through.
        line: Line,
    },
    /// Slope measurement along a given [Line], using a particular [SlopeMethod].
    Slope {
        /// [Line] to trace measurement through, though depending on the slope method,
        /// the slope may be calculated using regions "near" the Line.
        line: Line,
        /// [SlopeMethod] used to measure slope.
        method: SlopeMethod,
    },
}

/// Variants allowing different methods of slope computation.
#[derive(Debug)]
pub enum SlopeMethod {
    /// Measure the average slope over the interval connecting two points.
    TwoPoint {
        /// *Negative* end of the interval
        neg: Vec2D,
        /// *Positive* end of the interval
        pos: Vec2D,
    },
    /// Approximate the average slope over a region defined by two edges of a rectangle.
    /// The edges are defined by their length, their separation, and the axis along which
    /// they are separated (the *gradient axis*).
    TwoEdge {
        /// Length of edges (e.g., subaperture height for x-slope measurement)
        edge_length: f64,
        /// Separation of edges (e.g., subaperture width for x-slope measurement)
        edge_separation: f64,
        /// Axis of gradient vector to sample, e.g., x-slope -> `Vec2D::new(1.0, 0.0)`
        gradient_axis: Vec2D,
        /// Number of points to sample along each edge (more points is more accurate but more demanding).
        npoints: u32,
    },
}

impl SlopeMethod {
    fn bundle(&self, line: &Line) -> Vec<(Line,Line,f64)> {
        match self {
            SlopeMethod::TwoPoint{pos, neg} => vec![((line+pos), (line+neg), 1.0)],
            SlopeMethod::TwoEdge{edge_length, edge_separation, gradient_axis, npoints} => {
                let point_a =  edge_length * 0.5 * gradient_axis.ortho();
                let point_b = -point_a.clone();
                let points = Vec2D::linspace(&point_a, &point_b, *npoints);
                points
                .iter()
                .map(|p|
                    (
                        (line.clone() + (p.clone() + gradient_axis * edge_separation * 0.5)),
                        (line.clone() + (p.clone() - gradient_axis * edge_separation * 0.5)),
                        (1.0 / *npoints as f64),
                    )
                ).collect()
            },
        }
    }
}

/// The atomic actuation unit.
/// 
/// An [Actuator]'s state is defined by a scalar value, so a device with `N`
/// actuatable degrees of freedom is considered as `N` different [Actuator]s,
/// e.g., `&[Actuator; N]`.
#[derive(Debug)]
pub enum Actuator{
    /// A null actuator, making zero impact on any `Measurement`
    Zero,
    /// A circularly symmetric Gaussian actuator, centred at `position` with
    /// a specified scalar `sigma`. See [gaussian2d] for more info.
    Gaussian {
        /// sigma of gaussian function in metres. 
        sigma: f64,
        /// position of actuator in 3d space, z=altitude.
        position: Vec3D,
    },
    TipTilt {
        /// position along slope of TT actuator surface where the amplitude
        /// of the phase is equal to +1.0 units. This vector is colinear with
        /// the acuation axis. E.g., if this vector is `(1.0,0.0)`, then the 
        /// response of the actuator will be a 
        ///Deliberately not in arcseconds
        /// so that you have to be deliberate and careful about your units.
        unit_response: Vec2D,
    },
}

impl Measurable for Actuator {
    fn phase(&self, line: &Line) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::Gaussian{sigma, position} => {
                let distance = position.distance_at_altitude(line);
                gaussian(distance / sigma)
            },
            Self::TipTilt{unit_response} => {
                line.position_at_altitude(0.0).dot(unit_response)
            }
        }
    }

    fn distance(&self, p: &Line, q: &Line) -> f64 {
        match self {
            Self::Zero => 1.0,
            Self::Gaussian{position, ..} => {
                let dp = position.displacement_at_altitude(p);
                let dq = position.displacement_at_altitude(q);
                (dp-dq).norm()
            },
            Self::TipTilt{..} => {
                let dp = p.position_at_altitude(0.0);
                let dq = q.position_at_altitude(0.0);
                (dp-dq).norm()
            }
        }
    }
}

/// Interaction Matrix between measurements and actuators.
/// 
/// The interaction matrix ([IMat]) is the interface between measurements and
/// actuators. Specifically, the [IMat] has elements which are equal to the
/// response of each [Measurement] to a unit input from each [Actuator].
/// # Examples
/// Let's assume we have:
///  - Two [Actuator]s, with Gaussian influence functions, located at `(x, y)`:
///    - `(+1.0, 0.0)` metres, and
///    - `(-1.0, 0.0)` metres, 
///
///    on a deformable mirror conjugated to 10 km in altitude, and with a coupling
///    of 0.4 at pitch of 2.0 metres.
///  - Three [Measurement]s, measuring the y-slope on-axis, at projected pupil
///    positions of `(x, y)`:
///    - `(-1.0, -1.0)`
///    - `( 0.0,  0.0)`
///    - `(+1.0, +1.0)`
/// We first construct those measurements and actuators, then build an imat from
/// them, and print the elements of that imat:
/// ```
/// const PITCH: f64 = 2.0;  // metres
/// const ALTITUDE: f64 = 10_000.0;  // metres
/// const ACTU_POS: [[f64;2];2] = [
///     [-1.0, 0.0],  // [x1,y1]
///     [ 1.0, 0.0],  // [x2,y2]
/// ];
/// const MEAS_POS: [[f64;2];3] = [  // metres
///     [-1.0, -1.0],  // [x1,y1]
///     [ 0.0,  0.0],  // [x2,y2]
///     [ 1.0,  1.0],  // [x3,y3]
/// ];
///
/// const COUPLING: f64 = 0.4; // dimensionless
/// let sigma = rao::coupling_to_sigma(COUPLING, PITCH);
/// let actuators = ACTU_POS.map(|[x,y]|
///     rao::Actuator::Gaussian {
///         sigma: sigma,
///         position: rao::Vec3D::new(x, y, ALTITUDE),
///     }
/// );
/// let measurements = MEAS_POS.map(|[x,y]|
///     rao::Measurement::Slope {
///         line: rao::Line::new_on_axis(x, y),
///         method: rao::SlopeMethod::TwoPoint {
///             neg: rao::Vec2D::new(-PITCH/2.0, 0.0),
///             pos: rao::Vec2D::new( PITCH/2.0, 0.0),
///         }
///     }
/// );
/// 
/// let imat = rao::IMat::new(&measurements, &actuators);
/// println!("{}", imat);
/// ```
/// which will print something similar to:
/// ```txt
/// [[  0.36  0.15 ]
///  [ -0.00 -0.00 ]
///  [ -0.15 -0.36 ]]
/// ```
/// # Notes on units
/// In Adaptive Optics, there is little/no standardisation of units for measurements
/// and actuators, but some units are seen more often than others. For example, we
/// often see:
///  - microns, metres, radians, or waves for phase measurement units,
///  - arcsec, radians, or dimensionless units for slope measurement units,
///  - microns, metres, waves, arcseconds (e.g., for tip-tilt mirrors), or volts
///    for actuator units.
///
/// This is a common point of confusion, particularly for AO newcomers. In fact,
/// if we are operating under the paraxial regime, and we only consider linear interaction
/// functions (i.e., those that are well captured by an *Interaction Matrix*), then
/// we can safely refuse to define any particular units during the construction of an
/// interaction matrix. To demonstrate, consider the above example. The units of the
/// slopes are in *influence function units* per *distance units*. The influence
/// function in this case is Gaussian, and is parameterised only by its standard
/// deviation, `sigma`, which is in the same lineal distance units as the various
/// coordinates we defined. In the example above, we denotes those units as metres,
/// but if one assumed different distance units, the resulting interaction matrix
/// would be identical. Let's say that I use the units of
/// [furlongs](https://en.wikipedia.org/wiki/Furlong) (1 furlong ==  201.1680 metres).
/// Then I would have measured my AO system to have the geometry:
/// ```
/// const PITCH: f64 = 9.941e-3;  // furlongs
/// const ALTITUDE: f64 = 49.71;  // furlongs
/// const ACTU_POS: [[f64;2];2] = [
///     [-4.971e-3, 0.0],  // [x1,y1]
///     [ 4.971e-3, 0.0],  // [x2,y2]
/// ];
/// const MEAS_POS: [[f64;2];3] = [  // furlongs
///     [-4.971e-3, -4.971e-3],
///     [ 0.0,  0.0],
///     [ 4.971e-3,  4.971e-3],
/// ];
/// ```
/// Replacing the variables in the example above with these ones, I would get my
/// interaction matrix and would be confident that the units of the interaction
/// matrix are "influence function units per furlong per actuator unit"
/// ```txt
/// [[ 73.31 29.32 ]
///  [ -0.00 -0.00 ]
///  [ -29.32 -73.31 ]]
/// ```
/// Note that this is exactly the same interaction matrix as before, but scaled by
/// a factor of 201.1680 (metres per furlong).
/// The way to read this is, for example, the gradient of the first influence 
/// function when traced along the first measurement axis is 73.31 units per furlong,
/// or indeed, per YOUR_UNITS where you assumed those units in the definition of
/// the system. The point is, if you are consistent with your inputs, then you can 
/// use any units and the output will comply.
/// 
/// At present, the only influence function available is the Gaussian one, but this
/// "unit agnosticism" is so attractive that it might as well set the convention for
/// this crate: *where possible, avoid assuming/defining/requiring units*.
#[derive(Debug)]
pub struct IMat<'a> {
    /// slice of actuators defining this interaction matrix
    actuators: &'a [Actuator],
    /// slice of measurements defining this interaction matrix
    measurements: &'a [Measurement],
}

impl<'a> IMat<'a>{
    /// Define a new [IMat] with measurements and actuators. Note that this function
    /// is as *lazy* as possible, and the actual computation of the interaction matrix
    /// only happens when the elements of that matrix are requested.
    pub fn new(measurements: &'a [Measurement], actuators: &'a [Actuator]) -> Self {
        IMat {
            measurements,
            actuators,
        }
    }
}

impl Matrix for IMat<'_> {
    fn eval(&self, meas_idx: usize, actu_idx: usize) -> f64 {
        let actuator: &Actuator = &self.actuators[actu_idx];
        let measurement: &Measurement = &self.measurements[meas_idx];
        match (actuator, measurement) {
            (Actuator::Zero, _) => 0.0,
            (_, Measurement::Zero) => 0.0,
            (_, meas) => actuator.measure(meas),
        }
    }
    fn nrows(&self) -> usize {
        self.measurements.len()
    }
    fn ncols(&self) -> usize {
        self.actuators.len()
    }
}

impl fmt::Display for IMat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.format(f)
    }
}

impl<T: CoSampleable> fmt::Display for CovMat<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.format(f)
    }
}

pub struct VonKarmanLayer {
    r0: f64,
    l0: f64,
    alt: f64,
}

impl VonKarmanLayer {
    pub fn new(r0: f64, l0: f64, alt: f64) -> VonKarmanLayer {
        VonKarmanLayer {
            r0, l0, alt
        }
    }
    pub fn new_test_layer(r0: f64) -> VonKarmanLayer {
        VonKarmanLayer {
            r0: r0,
            l0: 25.0,
            alt: 0.0,
        }
    }
}

impl CoSampleable for VonKarmanLayer {
    fn eval(&self, linea: &Line, lineb:&Line) -> f64 {
        let p1 = linea.position_at_altitude(self.alt);
        let p2 = lineb.position_at_altitude(self.alt);
        vk_cov((p1-p2).norm(), self.r0, self.l0)
    }
    fn distance(&self, linea: &Line, lineb: &Line) -> f64 {
        let p1 = linea.position_at_altitude(self.alt);
        let p2 = lineb.position_at_altitude(self.alt);
        (p1-p2).norm()
    }
}

#[derive(Debug)]
pub struct CovMat<'a, T: CoSampleable>
{
    measurements_left: &'a [Measurement],
    measurements_right: &'a [Measurement],
    cov_model: &'a T,
}

impl<'a, T: CoSampleable> CovMat<'a, T> {
    pub fn new(
        measurements_left: &'a [Measurement],
        measurements_right: &'a [Measurement],
        cov_model: &'a T
    ) -> CovMat<'a, T> {
        CovMat {
            measurements_left,
            measurements_right,
            cov_model,
        }
    }
}


impl<T: CoSampleable> Matrix for CovMat<'_, T> {
    fn nrows(&self) -> usize {
        self.measurements_left.len()
    }
    fn ncols(&self) -> usize {
        self.measurements_right.len()
    }
    fn eval(&self, row_index: usize, col_index: usize) -> f64 {
        let meas_left: &Measurement = &self.measurements_left[row_index];
        let meas_right: &Measurement = &self.measurements_right[col_index];
        self.cov_model.cosample(&meas_left, &meas_right)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq};
    
    #[test]
    fn gaussian_on_axis_phase() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::new(0.0, 0.0, 0.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::new_on_axis(0.0, 0.0)
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0), 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn gaussian_off_axis_phase() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::new(0.0, 0.0, 1000.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::new(0.0, 1.0/1000.0, 0.0, 0.0)
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0), 0.5, epsilon = f64::EPSILON);
    }

    #[test]
    fn gaussian_off_axis_phase_twopoint() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::new(0.0, 0.0, 1000.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::new_from_two_points(
                    &Vec3D::new(1.0,1.0,0.0),
                    &Vec3D::new(1.0,-1.0,2000.0),
                )
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0), 0.5);
    }

    #[test]
    fn simple_symmetric() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::new(0.0, 0.0, 1000.0),
            },
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::new(1.0, 0.0, 1000.0),
            },
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::new_on_axis(0.0, 0.0),
            },
            Measurement::Phase{
                line: Line::new(1.0, 0.0, 0.0, 0.0),
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert!(imat.eval(0,0)>0.0);
        assert_abs_diff_eq!(imat.eval(0,0),imat.eval(1,1));
        assert_abs_diff_eq!(imat.eval(1,0),imat.eval(0,1));
    }

    #[test]
    fn slope_twopoint() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::origin(),
            }
        ];
        let measurements = [
            Measurement::Slope{
                line: Line::new(1.0, 0.0, 0.0, 0.0),
                method: SlopeMethod::TwoPoint{
                    neg: Vec2D::new(-1e-6, 0.0),
                    pos: Vec2D::new(1e-6, 0.0)
                }
            },
            Measurement::Slope{
                line: Line::new(1.0, 0.0, 0.0, 0.0),
                method: SlopeMethod::TwoPoint{
                    neg: Vec2D::new(-1e-7, 0.0),
                    pos: Vec2D::new(1e-7, 0.0)
                }
            },
            Measurement::Phase{
                line: Line::new(1.0+1e-7, 0.0, 0.0, 0.0),
            },
            Measurement::Phase{
                line: Line::new(1.0-1e-7, 0.0, 0.0, 0.0),
            },
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0),imat.eval(1,0),epsilon=1e-8);
        assert_abs_diff_eq!(
            (imat.eval(2,0)-imat.eval(3,0))/2e-7,
            imat.eval(1,0),
            epsilon=1e-8
        );
    }


    #[test]
    fn slope_twoedge() {
        let actuators = [
            Actuator::Gaussian{
                sigma: coupling_to_sigma(0.5,1.0),
                position: Vec3D::origin(),
            }
        ];

        let line = Line::new(1.0, 0.0, 0.0, 0.0);
        let measurements = [
            Measurement::Slope{
                line: line.clone(),
                method: SlopeMethod::TwoPoint{
                    neg: Vec2D::new(-1e-2, 0.0),
                    pos: Vec2D::new(1e-2, 0.0),
                }
            },
            Measurement::Slope{
                line: line,
                method: SlopeMethod::TwoEdge{
                    edge_length: 0.0,
                    edge_separation: 2e-2,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 10
                }
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0),imat.eval(1,0));
    }

    #[test]
    fn slope_tt_twopoint() {
        let actuators = [
            Actuator::TipTilt{
                unit_response: Vec2D::x_unit()
            }
        ];
        let measurements = [
            Measurement::Slope{
                line: Line::new(1.0, 0.0, 0.0, 0.0),
                method: SlopeMethod::TwoPoint{
                    neg: Vec2D::new(-1e-6, 0.0),
                    pos: Vec2D::new(1e-6, 0.0)
                }
            },
            Measurement::Slope{
                line: Line::new(1.0, 0.0, 0.0, 0.0),
                method: SlopeMethod::TwoPoint{
                    neg: Vec2D::new(-1e-7, 0.0),
                    pos: Vec2D::new(1e-7, 0.0)
                }
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0),imat.eval(1,0),epsilon=1e-8);
    }

    #[test]
    fn tt_on_axis_phase() {
        let actuators = [
            Actuator::TipTilt{
                unit_response: Vec2D::x_unit()
            }
        ];
        let measurements = [
            Measurement::Phase {
                line: Line::new_on_axis(0.0, 0.0)
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0), 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn tt_off_axis_phase() {
        let actuators = [
            Actuator::TipTilt{
                unit_response: Vec2D::x_unit()
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::new(2.0, 0.0, 0.0, 0.0)
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0), 2.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_vkcov() {
        let vk = VonKarmanLayer {
            r0: 0.1,
            l0: 25.0,
            alt: 1000.0,
        };
        let line = Line::new_on_axis(0.0,0.0);
        let a = vk.eval(&line, &line);
        assert_abs_diff_eq!(a,vk_cov(0.0, vk.r0, vk.l0));
        assert_abs_diff_eq!(a,856.3466131373517,epsilon=1e-3);
    }

    #[test]
    fn test_vkcovmat() {
        let vk = VonKarmanLayer {
            r0: 0.1,
            l0: 25.0,
            alt: 1000.0,
        };
        let measurements: Vec<Measurement> = (0..10)
        .map(|i| i as f64 * 0.8)
        .map(|x|
            Measurement::Phase{
                line: Line::new_on_axis(x, 0.0)
            }
        ).collect();
        let covmat = CovMat{
            measurements_left: &measurements,
            measurements_right: &measurements,
            cov_model: &vk
        };
        println!("{}", covmat);
    }
    
    #[test]
    fn test_mixedvkcovmat() {
        let vk = VonKarmanLayer {
            r0: 0.1,
            l0: 25.0,
            alt: 1000.0,
        };
        let measurements = [
            Measurement::Phase {
                line: Line::new_on_axis(0.0, 0.0)
            },
            Measurement::Slope {
                line: Line::new_on_axis(0.0, 0.0),
                method: SlopeMethod::TwoPoint {
                    pos: Vec2D::new(0.1,0.0),
                    neg: Vec2D::new(-0.1,0.0),
                },
            },
            Measurement::Slope {
                line: Line::new_on_axis(0.0, 0.0),
                method: SlopeMethod::TwoEdge {
                    edge_separation: 0.2,
                    edge_length: 0.2,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 10,
                }
            }
        ];
        let covmat = CovMat{
            measurements_left: &measurements,
            measurements_right: &measurements,
            cov_model: &vk
        };
        println!("{}", covmat);
    }
}
