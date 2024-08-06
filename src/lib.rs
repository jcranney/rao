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
//!         measurements.push(rao::Measurement::SlopeTwoEdge{
//!             central_line: line.clone(),
//!             edge_separation: PITCH,
//!             edge_length: PITCH,
//!             npoints: 5,
//!             gradient_axis: rao::Vec2D::x_unit(),
//!         });
//!         // y-slope
//!         measurements.push(rao::Measurement::SlopeTwoEdge{
//!             central_line: line.clone(),
//!             edge_separation: PITCH,
//!             edge_length: PITCH,
//!             npoints: 5,
//!             gradient_axis: rao::Vec2D::y_unit(),
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
mod core;
pub use crate::core::{
    Sampler,
    Sampleable,
    CoSampleable,
    IMat,
    CovMat,
};


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
    /// Measure the average slope over the interval connecting two points.
    SlopeTwoLine {
        /// tbd
        line_pos: Line,
        line_neg: Line,
    },
    /// Approximate the average slope over a region defined by two edges of a rectangle.
    /// The edges are defined by their length, their separation, and the axis along which
    /// they are separated (the *gradient axis*).
    SlopeTwoEdge {
        central_line: Line,
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

impl Sampler for Measurement {
    fn get_bundle(&self) -> Vec<(Line,f64)> {
        match self {
            Measurement::Zero => vec![],
            Measurement::Phase{line} => vec![(line.clone(),1.0)],
            Measurement::SlopeTwoLine{line_pos, line_neg} => {
                let coeff = 1.0/line_pos.distance_at_ground(line_neg);
                vec![
                    (line_pos.clone(),  coeff),
                    (line_neg.clone(), -coeff),
                ]
            },
            Measurement::SlopeTwoEdge{central_line, edge_length, edge_separation, gradient_axis, npoints} => {
                let coeff = (1.0 / *npoints as f64) / edge_separation;
                let offset_vec = gradient_axis * edge_separation * 0.5;
                let point_a =  edge_length * 0.5 * gradient_axis.ortho();
                let point_b = -point_a.clone();
                Vec2D::linspace(&point_a, &point_b, *npoints)
                .iter()
                .map(|p|
                    vec![
                        (central_line + (p + &offset_vec),  coeff),
                        (central_line + (p - &offset_vec), -coeff),
                    ]
                ).flatten()
                .collect()
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

impl Sampleable for Actuator {
    fn sample(&self, line: &Line) -> f64 {
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
    fn cosample(&self, linea: &Line, lineb:&Line) -> f64 {
        let p1 = linea.position_at_altitude(self.alt);
        let p2 = lineb.position_at_altitude(self.alt);
        vk_cov((p1-p2).norm(), self.r0, self.l0)
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
        let line = Line::new(1.0, 0.0, 0.0, 0.0);
        let measurements = [
            Measurement::SlopeTwoLine{
                line_neg: &line+Vec2D::new(-1e-5, 0.0),
                line_pos: &line+Vec2D::new(1e-5, 0.0)
            },
            Measurement::SlopeTwoLine{
                line_neg: &line+Vec2D::new(-1e-6, 0.0),
                line_pos: &line+Vec2D::new(1e-6, 0.0)
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
            Measurement::SlopeTwoLine{
                line_neg: &line+Vec2D::new(-1e-2, 0.0),
                line_pos: &line+Vec2D::new(1e-2, 0.0),
            },
            Measurement::SlopeTwoEdge{
                central_line: line,
                edge_length: 0.0,
                edge_separation: 2e-2,
                gradient_axis: Vec2D::x_unit(),
                npoints: 100
            }
        ];
        let imat = IMat::new(&measurements, &actuators);
        assert_abs_diff_eq!(imat.eval(0,0),imat.eval(1,0),epsilon=1e-10);
    }

    #[test]
    fn slope_tt_twopoint() {
        let actuators = [
            Actuator::TipTilt{
                unit_response: Vec2D::x_unit()
            }
        ];
        let line = Line::new(1.0, 0.0, 0.0, 0.0);
        let measurements = [
            Measurement::SlopeTwoLine{
                line_neg: &line+Vec2D::new(-1e-6, 0.0),
                line_pos: &line+Vec2D::new(1e-6, 0.0)
            },
            Measurement::SlopeTwoLine{
                line_neg: &line+Vec2D::new(-1e-7, 0.0),
                line_pos: &line+Vec2D::new(1e-7, 0.0)
            },
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
        let a = vk.cosample(&line, &line);
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
        let covmat = CovMat::new(&measurements,&measurements,&vk);
        println!("{}", covmat);
    }
    
    #[test]
    fn test_mixedvkcovmat() {
        let vk = VonKarmanLayer {
            r0: 0.1,
            l0: 25.0,
            alt: 1000.0,
        };
        let line = Line::new_on_axis(0.0, 0.0);
        let measurements = [
            Measurement::Phase {
                line: line.clone(),
            },
            Measurement::SlopeTwoLine {
                line_pos: &line+Vec2D::new(0.1,0.0),
                line_neg: &line+Vec2D::new(-0.1,0.0),
            },
            Measurement::SlopeTwoEdge {
                central_line: line.clone(),
                edge_separation: 0.2,
                edge_length: 0.2,
                gradient_axis: Vec2D::x_unit(),
                npoints: 10,
            },
        ];
        let covmat = CovMat::new(&measurements, &measurements, &vk);
        println!("{}", covmat);
    }
}
