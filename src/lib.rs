//! # rao
//! 
//! `rao` - Adaptive Optics tools in Rust - is a set of fast and robust adaptive
//! optics utilities. The current scope of `rao` is for the calculation of
//! large matrices in AO, used in the configuration of real-time adaptive optics,
//! control. Specifically, we aim to provide fast, scalable, and reliable APIs for
//! generating:
//!  - `rao::IMat` - the interaction matrix between measurements and actuators,
//!  - `rao::CovMat` - the covariance matrix between measurements.
//!
//! These two matrices are typically the largest computational burden in the
//! configuration of real-time control (RTC) for AO, and also the most 
//! performance-sensitive parts of the RTC.
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

pub mod utils;
pub use utils::coupling_to_sigma;
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


/// Common [Sampler]s in Adaptive Optics
/// 
/// A [Measurement] provides a scalar-valued sample of an AO system. A single
/// measurement device (e.g., a Shack Hartmann WFS) is typically comprised of 
/// many [Measurement]s, e.g., `&[Measurement; N]`.
#[derive(Debug)]
pub enum Measurement{
    /// The null measurement, always returning 0.0 regardless of the measured object.
    Zero,
    /// Phase measurement along a given [Line].
    Phase {
        /// [Line] in 3D space to trace through [Sampleable] objects.
        line: Line,
    },
    /// Slope measurement, defined by two [Line]s in 3D space.
    ///
    /// The slope measured is equal to the *sampled function at the point where
    /// `line_pos` intersects it* minus the *sampled function at the point
    /// where `line_neg` intersects it*, divided by the *distance between the
    /// two lines at zero-altitude*.
    SlopeTwoLine {
        /// positive end of slope (by convention)
        line_pos: Line,
        /// negative end of slope (by convention)
        line_neg: Line,
    },
    /// Slope measurement, defined by a [Line] in 3D space and  the sampled edges
    /// a rectangle in 2D space.
    ///
    /// Approximate the average slope over a region defined by two edges of a rectangle.
    /// The edges are defined by their length, their separation, and the axis along which
    /// they are separated (the *gradient axis*). The edges of the rectangle are sampled
    /// so that the spacing between points is uniform and the furthest distance to an
    /// unsampled point of the edge is minimal, i.e.:
    ///  - `npoints=1` => `[----x----]`,
    ///  - `npoints=2` => `[--x---x--]`,
    ///  - `npoints=3` => `[-x--x--x-]`,
    ///  - *etc*
    ///
    /// This is not the typical "linspace", which is ill-defined for 1 point, though one
    /// could implement that as a [Sampler] wrapper around [`Measurement::SlopeTwoLine`].
    SlopeTwoEdge {
        /// Principle axis of the WFS
        central_line: Line,
        /// Length of edges (e.g., subaperture height for x-slope measurement)
        edge_length: f64,
        /// Separation of edges (e.g., subaperture width for x-slope measurement)
        edge_separation: f64,
        /// Axis of gradient vector to sample, e.g., x-slope -> `Vec2D::new(1.0, 0.0)`
        gradient_axis: Vec2D,
        /// Number of points to sample along each edge (more points can be more accurate).
        npoints: u32,
    },
}

/// [Measurement] is the prototypical [Sampler] type.
impl Sampler for Measurement {
    /// The [`Sampler::get_bundle`] method implementation for the [Measurement] variants
    /// should serve as an example for implementing other [Sampler] types. Inspect the
    /// source code for the reference implementation. In short:
    ///  - a [`Measurement::Zero`] variant returns an empty vector,
    ///  - a [`Measurement::Phase`] variant returns a single line with a coefficient of `1.0`,
    ///  - a [`Measurement::SlopeTwoLine`] variant returns two lines, with a positive
    ///    and negative coefficient (for the *start* and *end* of the slope), scaled by
    ///    the inverse of the ground separation of the lines, so that the resulting units
    ///    are in *sampled function units per distance unit*.
    ///  - a [`Measurement::SlopeTwoEdge`] variant returns `2 * npoints` lines, consisting of
    ///    `npoints` positive coefficients, and `npoints` negative coefficients, each scaled
    ///    by the inverse of the ground-layer separation and a factor of `1.0 / npoints as f64`,
    ///    in order to get a result which is in units of *sampled function units per distance unit*. 
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
                let coeff = (1.0 / f64::from(*npoints)) / edge_separation;
                let offset_vec = gradient_axis * edge_separation * 0.5;
                let point_a =  edge_length * 0.5 * gradient_axis.ortho();
                let point_b = -point_a.clone();
                Vec2D::linspace(&point_a, &point_b, *npoints)
                .iter()
                .flat_map(|p|
                    vec![
                        (central_line + (p + &offset_vec),  coeff),
                        (central_line + (p - &offset_vec), -coeff),
                    ])
                .collect()
            },
        }
    }
}



/// Common [Sampleable]s in Adaptive Optics.
/// 
/// An [Actuator]'s state is defined by a scalar value, so a device with `N`
/// actuatable degrees of freedom is considered as `N` different [Actuator]s,
/// e.g., `&[Actuator; N]`.
#[derive(Debug)]
pub enum Actuator{
    /// A null actuator, making zero impact on any `Measurement`
    Zero,
    /// A circularly symmetric Gaussian actuator, centred at `position` with
    /// a specified scalar `sigma`. See [`utils::gaussian`] for more info.
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
                utils::gaussian(distance / sigma)
            },
            Self::TipTilt{unit_response} => {
                line.position_at_altitude(0.0).dot(unit_response)
            }
        }
    }
}

/// Simple covariance model, this might be refactored into an enum of models.
pub struct VonKarmanLayer {
    pub r0: f64,
    pub l0: f64,
    pub alt: f64,
}

impl VonKarmanLayer {
    /// Construct a new von Karman turbulence layer from its parameters
    #[must_use]
    pub fn new(r0: f64, l0: f64, alt: f64) -> VonKarmanLayer {
        VonKarmanLayer {
            r0, l0, alt
        }
    }
}

/// [`VonKarmanLayer`] is (for now) the prototypical [`CoSampleable`] object.
///
/// Perhaps confusingly, this implementation allows the cosampling of the
/// von Karman turbulence statistical model, returning the covariance between
/// two [`Line`]s intercepting that layer.
impl CoSampleable for VonKarmanLayer {
    fn cosample(&self, line_a: &Line, line_b:&Line) -> f64 {
        let p1 = line_a.position_at_altitude(self.alt);
        let p2 = line_b.position_at_altitude(self.alt);
        utils::vk_cov((p1-p2).norm(), self.r0, self.l0)
    }
}


pub struct Pupil {
    pub rad_outer: f64,
    pub rad_inner: f64,
    pub spider_thickness: f64,
    pub spiders: Vec<(Vec2D,Vec2D)>
}

impl Sampleable for Pupil {
    fn sample(&self, ell: &Line) -> f64 {
        let p = ell.position_at_altitude(0.0);
        let mut out: f64 = 1.0;
        let r = p.norm();
        if r > self.rad_outer {
            out *= 0.0;
        }
        if r < self.rad_inner {
            out *= 0.0;
        }
        for spider in &self.spiders {
            if signed_distance_to_capsule(
                &p, &spider.0, &spider.1, self.spider_thickness/2.0
            ) < 0.0 {
                out *= 0.0;
            }
        }
        out
    }
}

fn signed_distance_to_capsule(
    position: &Vec2D,
    capsule_start: &Vec2D,
    capsule_end: &Vec2D,
    radius: f64
) -> f64 {
    let pa: Vec2D = position - capsule_start;
    let ba: Vec2D = capsule_end - capsule_start;
    let mut h: f64 = pa.dot(&ba)/ba.dot(&ba);
    h = h.clamp(0.0, 1.0);
    (pa - ba*h).norm() - radius
}



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
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
        assert_abs_diff_eq!(a,utils::vk_cov(0.0, vk.r0, vk.l0));
        assert_abs_diff_eq!(a,856.346_613_137_351_7,epsilon=1e-3);
    }

    #[test]
    fn test_vkcovmat() {
        let vk = VonKarmanLayer {
            r0: 0.1,
            l0: 25.0,
            alt: 1000.0,
        };
        let measurements: Vec<Measurement> = (0..10)
        .map(|i| f64::from(i) * 0.8)
        .map(|x|
            Measurement::Phase{
                line: Line::new_on_axis(x, 0.0)
            }
        ).collect();
        let covmat = CovMat::new(&measurements,&measurements,&vk);
        println!("{covmat}");
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
        println!("{covmat}");
    }

    #[test]
    fn make_pupil() {
        let pup = [Pupil{
            rad_outer: 4.0,
            rad_inner: 0.5,
            spider_thickness: 0.1,
            spiders: vec![
                (Vec2D::new(-4.0,-4.0),Vec2D::new(3.0,0.0)),
                (Vec2D::new(-4.0,-4.0),Vec2D::new(3.0,0.0)),
                (Vec2D::new(-4.0,-4.0),Vec2D::new(3.0,0.0)),
                (Vec2D::new(-4.0,-4.0),Vec2D::new(3.0,0.0)),
                ],
        }];
        const NPOINTS: u32 = 1000;
        let x = Vec2D::linspace(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new(4.0, 0.0),
            NPOINTS,
        );
        let y = Vec2D::linspace(
            &Vec2D::new(0.0, -4.0),
            &Vec2D::new(0.0, 4.0),
            NPOINTS,
        );
        println!("{y:?}");
        let p: Vec<Measurement> = y.iter().flat_map(|y|
            x.iter().map(|x| Measurement::Phase{
                line:Line::new(x.x, 0.0, y.y, 0.0)
            })).collect();
        let pup_vec = IMat::new(&p, &pup);
        let shape = [NPOINTS as usize, NPOINTS as usize];
        let data: Vec<f64> = pup_vec.flattened_array();
        let primary_hdu = fitrs::Hdu::new(&shape, data);
        fitrs::Fits::create("/tmp/pup.fits", primary_hdu)
        .expect("Failed to create");
    }
}
