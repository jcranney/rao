use crate::geometry::Line;
use crate::linalg::Matrix;
use std::fmt;


// Core traits:
// - Sampleable (e.g., actuator influence functions, phase covariance functions)
// - Sampler (e.g., phase sample, slope measurements)

pub trait Sampler {
    /// a function which takes a principle line and returns a vector of lines
    /// and coefficients, each of which specify the weight that the samples are
    /// linearly combined with to form a single sample. 
    ///
    /// todo: provide example, because this is very hard to parse as is.
    fn get_bundle(&self) -> Vec<(Line,f64)>;
    
    fn sample(&self, object: &impl Sampleable) -> f64 {
        self.get_bundle()
        .into_iter()
        .map(|(l,a)|
            object.sample(&l)*a
        ).sum()
    }
    
    fn cosample(&self, other: &impl Sampler, object: &impl CoSampleable) -> f64 {
        let bundle_left = self.get_bundle();
        let bundle_right = other.get_bundle();
        bundle_left
        .iter()
        .map(|(line_left,coeff_left)| {
            bundle_right
            .iter()
            .map(|(line_right,coeff_right)| {
                object.cosample(&line_left,&line_right)*coeff_left*coeff_right
            })
            .sum::<f64>()
        }).sum()
    }
}


pub trait Sampleable {
    fn sample(&self, p: &Line) -> f64;
}

pub trait CoSampleable {
    fn cosample(&self, p: &Line, q: &Line) -> f64;
}


/// Interaction Matrix between [Sampler] and [Sampleable].
/// 
/// DOCS NEED UPDATING, OVER SPECIFIED CURRENTLY
/// The interaction matrix ([IMat]) is the interface between any object that
/// implements [Sampler] (e.g., a measurement) and another object that implements
/// [Sampleable] (e.g., an actuator).
/// 
/// Specifically, the [IMat] has elements which are equal to the
/// sampled value of each [Sampleable] object when sampled by a [Sampler].
/// # Examples
/// Let's assume we have:
///  - Two [crate::Actuator]s, with Gaussian influence functions, located at `(x, y)`:
///    - `(+1.0, 0.0)` metres, and
///    - `(-1.0, 0.0)` metres, 
///
///    on a deformable mirror conjugated to 10 km in altitude, and with a coupling
///    of 0.4 at pitch of 2.0 metres.
///  - Three [crate::Measurement]s, measuring the y-slope on-axis, at projected pupil
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
///     rao::Measurement::SlopeTwoLine {
///         line_neg: rao::Line::new_on_axis(x-PITCH/2.0, y),
///         line_pos: rao::Line::new_on_axis(x+PITCH/2.0, 0.0),
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
/// In Adaptive Optics, there is very little standardisation of units for measurements
/// and actuators, but some units are seen more often than others. For example, we
/// often see:
///  - microns, metres, radians, or waves for phase measurement units,
///  - arcsec, radians, or other dimensionless units for slope measurement units,
///  - microns, metres, waves, arcseconds (e.g., for tip-tilt mirrors), or volts
///    for actuator units.
///
/// This is a common point of confusion, particularly for AO newcomers. As it happens,
/// if we are operating under the paraxial regime, and we only consider linear interaction
/// functions (i.e., those that are well captured by an *Interaction Matrix*), then
/// we can safely refuse to define any particular units at compile time for an
/// interaction matrix. To demonstrate, consider the above example. The units of the
/// slopes are in *influence function units* per *distance units*. The influence
/// function in this case is Gaussian, and is parameterised only by its standard
/// deviation, `sigma`, which is in the same lineal distance units as the various
/// coordinates we defined. In the example above, we denotes those units as metres,
/// but if one assumed different distance units, the resulting interaction matrix
/// would change only in the appropriate way. Let's say that I use the units of
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
pub struct IMat<'a, T: Sampler, U: Sampleable> {
    /// slice of measurements defining this interaction matrix
    samplers: &'a [T],
    /// slice of actuators defining this interaction matrix
    sampleables: &'a [U],
}

impl<'a, T: Sampler, U: Sampleable> IMat<'a, T, U>{
    /// Define a new [IMat] with measurements and actuators. Note that this function
    /// is as *lazy* as possible, and the actual computation of the interaction matrix
    /// only happens when the elements of that matrix are requested.
    pub fn new(samplers: &'a [T], sampleables: &'a [U]) -> Self {
        IMat {
            samplers,
            sampleables,
        }
    }
}

impl<T: Sampler, U: Sampleable> Matrix for IMat<'_, T, U> {
    fn eval(&self, sampler_idx: usize, sampleable_idx: usize) -> f64 {
        let sampler = &self.samplers[sampler_idx];
        let sampleable = &self.sampleables[sampleable_idx];
        sampler.sample(sampleable)
    }
    fn nrows(&self) -> usize {
        self.samplers.len()
    }
    fn ncols(&self) -> usize {
        self.sampleables.len()
    }
}

impl<T: Sampler, U: Sampleable> fmt::Display for IMat<'_, T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.format(f)
    }
}


#[derive(Debug)]
pub struct CovMat<'a, T: CoSampleable, L: Sampler, R: Sampler>
{
    samplers_left: &'a [L],
    samplers_right: &'a [R],
    cov_model: &'a T,
}

impl<'a, T: CoSampleable, L: Sampler, R: Sampler> CovMat<'a, T, L, R> {
    pub fn new(
        samplers_left: &'a [L],
        samplers_right: &'a [R],
        cov_model: &'a T
    ) -> CovMat<'a, T, L, R> {
        CovMat {
            samplers_left,
            samplers_right,
            cov_model,
        }
    }
}

impl<T: CoSampleable, L: Sampler, R: Sampler> Matrix for CovMat<'_, T, L, R> {
    fn nrows(&self) -> usize {
        self.samplers_left.len()
    }
    fn ncols(&self) -> usize {
        self.samplers_right.len()
    }
    fn eval(&self, row_index: usize, col_index: usize) -> f64 {
        let sampler_left: &L = &self.samplers_left[row_index];
        let sampler_right: &R = &self.samplers_right[col_index];
        sampler_left.cosample(sampler_right, self.cov_model)
    }
}

impl<T: CoSampleable, L: Sampler, R: Sampler> fmt::Display for CovMat<'_, T, L, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.format(f)
    }
}