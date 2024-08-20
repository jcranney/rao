use crate::geometry::Line;
use crate::linalg::Matrix;
use std::fmt;

// Core traits:
// - Sampleable (e.g., actuator influence functions, phase covariance functions)
// - Sampler (e.g., phase sample, slope measurements)

/// Any type that implements [Sampler] can be used to *sample* a [Sampleable] object
/// or to *cosample* a [CoSampleable] object.
///
/// A [crate::Measurement] is a typical [Sampler] object, but anything which can provide
/// a *bundle* of [Line]s via the [Sampler::get_bundle] method is able to implement
/// the [Sampler] trait. Other (maybe not useful) implementions could include:
///  - range-finding probes, that allow one to find the length of the ray between
///    the ground and some element in 3D space.
///  - flux-detector, for sampling regions of a [Sampleable] flux field.
///  - LGS elongation estimator, sampling the length of the elongated sodium guide
///    star along a particular axis, as seen by a partiucular point in the pupil.
///    Actually, this functionality comes for free in the [crate::Measurement]
///    variants, but one would need to implement the appropriate [Sampleable] 
///    trait for a new `struct SodiumProfile` type.
pub trait Sampler {
    /// A method which takes a principle line and returns a vector of lines
    /// and coefficients, each of which specify the weight that the samples are
    /// linearly combined with to form a single sample. See, for example, 
    /// [crate::Measurement::get_bundle].
    fn get_bundle(&self) -> Vec<(Line,f64)>;
    
    /// A method to sample a [Sampleable] object with the bundle of lines returned
    /// by [Sampler::get_bundle].
    ///
    /// Iterates over Vec<(Line,f64)>, samples the [Sampleable] function the each
    /// lines, and then linearly combines those samples with the float coefficient.
    /// Until I figure out math formatting in rust docs, this python pseduo-code
    /// will have to do:
    /// ```python
    /// y = 0.0
    /// for (line, coeff) in bundle_self:
    ///   y += sampleable(line) * coeff
    /// return y
    /// ```
    fn sample(&self, object: &dyn Sampleable) -> f64 {
        self.get_bundle()
        .into_iter()
        .map(|(l,a)|
            object.sample(&l)*a
        ).sum()
    }
    
    /// Similar to [Sampler::sample] but *co-samples* a [CoSampleable] function 
    /// with a pair of [Sampler]s.
    /// 
    /// Note crucially that the two [Sampler]s do not themselves need to be the
    /// same type, they only need to both implement the [Sampler] trait. E.g., 
    /// in building a covariance matrix (a covariance function makes sense as a 
    /// a [CoSampleable] type), one can meaningfully have cross-terms in the
    /// covariance matrix which correspond to the covariance between 
    /// slope-measurements and phase measurements.
    ///
    /// This method does nested iterations of the bundles returned by each 
    /// [Sampler::get_bundle] method, and co-samples a [CoSampleable] function 
    /// with each line-pair. Then, *quadratically* combined according to the 
    /// product of the co-sampled function and the two float coefficients.
    ///
    /// Until I figure out math formatting in rust docs, this python pseduo-code
    /// will have to do:
    /// ```python
    /// y = 0.0
    /// for (line_a, coeff_a) in bundle_self:
    ///   for (line_b, coeff_b) in bundle_other:
    ///     y += cosampleable(line_a, line_b) * coeff_a * coeff_b
    /// return y
    /// ```
    fn cosample(&self, other: &dyn Sampler, object: &dyn CoSampleable) -> f64 {
        let bundle_left = self.get_bundle();
        let bundle_right = other.get_bundle();
        bundle_left
        .iter()
        .map(|(line_left,coeff_left)| {
            bundle_right
            .iter()
            .map(|(line_right,coeff_right)| {
                object.cosample(line_left,line_right)*coeff_left*coeff_right
            })
            .sum::<f64>()
        }).sum()
    }
}

/// Trait to enable a type to be sampled by a [Sampler].
///
/// The only requirement for this trait is that the type implements the
/// [Sampleable::sample] method.
pub trait Sampleable {
    /// takes the object itself and a [crate::Line], and returns a scalar float.
    fn sample(&self, p: &Line) -> f64;
}

/// Trait to enable a type to be cosampled by a pair of [Sampler]s
///
/// The only requirement for this trait is that the type implements the
/// [CoSampleable::cosample] method.
pub trait CoSampleable {
    /// takes the object itself and two [crate::Line]s, and returns a scalar float.
    fn cosample(&self, p: &Line, q: &Line) -> f64;
}

/// Generalised interaction matrix between [Sampler] and [Sampleable].
/// 
/// The interaction matrix ([IMat]) is the interface between any object that
/// implements [Sampler] (e.g., a measurement) and another object that implements
/// [Sampleable] (e.g., an actuator).
/// Specifically, the [IMat] has elements which are equal to the
/// sampled value of each [Sampleable] object when sampled by a [Sampler].
/// # Examples
/// Let's assume we have:
///  - Two [crate::Actuator]s, with Gaussian influence functions, located at `(x, y)`:
///    - `(+1.0, 0.0)` metres,
///    - `(-1.0, 0.0)` metres, 
///
///    on a deformable mirror conjugated to 10 km in altitude, and with a coupling
///    of 0.4 at a pitch of 2.0 metres.
///  - Three [crate::Measurement]s, measuring the y-slope on-axis, at projected pupil
///    positions of `(x, y)`:
///    - `(-1.0, -1.0)`
///    - `( 0.0,  0.0)`
///    - `(+1.0, +1.0)`
/// We construct those measurements and actuators, then we can build an imat from
/// them (since [crate::Measurement] implements [Sampler] and [crate::Actuator] 
/// implements [Sampleable]). Finally, we can print the elements of that imat:
/// ```
/// const PITCH: f64 = 2.0;  // metres
/// const ALTITUDE: f64 = 10_000.0;  // metres
/// const ACTU_POS: [[f64;2];2] = [
///     [-1.0, 0.0],  // [x1,y1] metres
///     [ 1.0, 0.0],  // [x2,y2] metres
/// ];
/// const MEAS_POS: [[f64;2];3] = [  
///     [-1.0, -1.0],  // [x1,y1] metres
///     [ 0.0,  0.0],  // [x2,y2] metres
///     [ 1.0,  1.0],  // [x3,y3] metres
/// ];
///
/// const COUPLING: f64 = 0.4; // coupling per pitch
/// let sigma = rao::coupling_to_sigma(COUPLING, PITCH); // metres
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
/// 
/// These values tell the *slope per actuator* response in units of "influence 
/// function units per distance units" per "actuation units". Perhaps it is implied
/// by classical assumptions that the specific units here might be:
/// *arcseconds per volt*, or something similar, but as explained below, there is
/// no need for the `rao` library to be this definitive on units.
///
// # Notes on units
// In Adaptive Optics, there is very little standardisation of units for measurements
// and actuators, but some units are seen more often than others. For example, we
// often see:
//  - microns, metres, radians, or waves for phase measurement units,
//  - arcsec, radians, or other dimensionless units for slope measurement units,
//  - microns, metres, waves, arcseconds (e.g., for tip-tilt mirrors), or volts
//    for actuator units.
//
// This is a common point of confusion, particularly for AO newcomers. As it happens,
// if we are operating under the paraxial regime, and we only consider linear interaction
// functions (i.e., those that are well captured by an *Interaction Matrix*), then
// we can safely refuse to define any particular units in this library. 
//
// To demonstrate, consider the above example. The units of the
// slope measurements are in *influence function units* per *distance units*.
//
// The influence
// function in this case is Gaussian, and is parameterised only by its standard
// deviation, `sigma`, which is in the same lineal distance units as the various
// coordinates we defined (in that case, metres). The Gaussian function has a value of 1.0 at its centre,
// so the assumption is that the command units are such that an input of 1.0 would
// produce a phase of 1.0 in the desired phase units. Let's assume that the desired
// phase units are microns, then the commands should be scaled such that a command of
// 1.0 would produce a surface aberration of the DM equal to 1.0 microns at the 
// actuator position.
// Then the output of the interaction matrix (the slopes) would be in units of 
// microns per metres == micro-radians (based on our assumptions here). If one desired "arcseconds"
// units for slopes, then one can convert (the dimensionless) micro-radians to 
// arcseconds by the usual `180/PI*3600/1e6`. Building these assumptions into this
// library assumes the user's intentions, and leaving them out puts the burden on
// the user to treat their units with care and precision. I'm still not sure if
// it's a brave choice or a cowardly one, but we decide to assume the user's 
// *attention* rather than their *intention*.
//
// In the example above, we denoted the distance units as metres,
// but if one assumed different distance units, the resulting interaction matrix
// would change only in the appropriate way. Let's say that I prefer using the units of
// [furlongs](https://en.wikipedia.org/wiki/Furlong) (1 furlong ==  201.1680 metres).
// Then I would have measured my AO system to have the geometry:
// ```
// const PITCH: f64 = 9.941e-3;  // furlongs
// const ALTITUDE: f64 = 49.71;  // furlongs
// const ACTU_POS: [[f64;2];2] = [
//     [-4.971e-3, 0.0],  // [x1,y1]
//     [ 4.971e-3, 0.0],  // [x2,y2]
// ];
// const MEAS_POS: [[f64;2];3] = [  // furlongs
//     [-4.971e-3, -4.971e-3],
//     [ 0.0,  0.0],
//     [ 4.971e-3,  4.971e-3],
// ];
// ```
// Replacing the variables in the example above with these ones, I would get my
// interaction matrix and would be confident that the units of the interaction
// matrix are "influence function units per furlong per actuator unit"
// ```txt
// [[ 73.31 29.32 ]
//  [ -0.00 -0.00 ]
//  [ -29.32 -73.31 ]]
// ```
// Note that this is exactly the same interaction matrix as before, but scaled by
// a factor of 201.1680 (metres per furlong).
// The way to read this is, for example, the gradient of the first influence 
// function when traced along the first measurement axis is 73.31 units per furlong,
// or indeed, per YOUR_UNITS where you assumed those units in the definition of
// the system. The point is, if you are consistent with your inputs, then you can 
// use any units and the output will comply.
// 
// At present, the only influence functions available are the Gaussian one, and
// "tip-tilt" but this "unit agnosticism" is so attractive that it might as well
// set the convention for this crate:
// *where possible, avoid assuming/defining/requiring units*.
#[derive(Debug)]
pub struct IMat<'a, T: Sampler, U: Sampleable> {
    /// slice of [Sampler]s defining this interaction matrix
    pub samplers: &'a [T],
    /// slice of [Sampleable]s defining this interaction matrix
    pub sampleables: &'a [U],
}

impl<'a, T: Sampler, U: Sampleable> IMat<'a, T, U>{
    /// Define a new [IMat] with [Sampler]s and [Sampleable]s. This function
    /// is *lazy*, and the actual computation of the interaction matrix
    /// only happens when the elements of that matrix are requested (and happens
    /// every time they are requested).
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


/// Generalised covariance matrix between two [Sampler]s.
///
/// Given two slices of [Sampler]s (`samplers_left` and `samplers_right`) and a
/// [CoSampleable] object, the [CovMat] is the set of cosamples of that object 
/// by all pairs of elements between `samplers_left` and `samplers_right`.
///
/// The naming, (*CovMat*, *cov_model*, etc.) comes from the prototypical
/// example of this object: the covariance matrix between measurements of an 
/// AO system. Taking the example from [IMat], we additionally define a covariance
/// model:
/// ```
/// const PITCH: f64 = 2.0;  // metres
/// const MEAS_POS: [[f64;2];3] = [  
///     [-1.0, -1.0],  // [x1,y1] metres
///     [ 0.0,  0.0],  // [x2,y2] metres
///     [ 1.0,  1.0],  // [x3,y3] metres
/// ];
///
/// // -- snip --
///
/// let measurements = MEAS_POS.map(|[x,y]|
///     rao::Measurement::SlopeTwoLine {
///         line_neg: rao::Line::new_on_axis(x-PITCH/2.0, y),
///         line_pos: rao::Line::new_on_axis(x+PITCH/2.0, y),
///     }
/// );
/// 
/// // Define a von Karman layer of turbulence:
/// let vk_layer = rao::VonKarmanLayer::new(
///     0.1,  // r0 (Fried parameter), metres
///     25.0, // L0 (outer scale), metres
///     0.0, // altitude of layer, metres
/// );
///
/// let covmat = rao::CovMat::new(&measurements, &measurements, &vk_layer);
/// println!("{}", covmat);
/// ```
/// which will print something similar to:
/// ```txt
/// [[ 95.73 48.02 16.44 ]
///  [ 48.02 95.73 48.02 ]
///  [ 16.44 48.02 95.73 ]]
/// ```
/// which is the covariance matrix between those measurements. The geometry of
/// the system is reflected in the values of the matrix - the measurements which
/// are closer together have a higher covariance. 
// The units follow the same argument
// as discussed in the documentation of the [IMat] type, but for this specific
// example, we can infer from the comments that the elements of the covariance
// matrix have units of "covariance function units per distance units"^2, and
// assuming that the user expects the influence function to return units of 
// microns, then these elements are in units of microrad^2.
#[derive(Debug)]
pub struct CovMat<'a, T: CoSampleable, L: Sampler, R: Sampler>
{
    /// left-hand-side slice of [Sampler]s
    ///
    /// "left" here refers to the interpretation of a matrix which can be
    /// left or right multiplied.
    pub samplers_left: &'a [L],
    /// right-hand-side slice of [Sampler]s
    ///
    /// "right" here refers to the interpretation of a matrix which can be
    /// left or right multiplied.
    pub samplers_right: &'a [R],
    /// Covariance model to be cosampled,
    ///
    /// Must implement the [CoSampleable] trait.
    pub cov_model: &'a T,
}

impl<'a, T: CoSampleable, L: Sampler, R: Sampler> CovMat<'a, T, L, R> {
    /// Construct a new [CovMat] given the defining properties.
    ///
    /// This function is *lazy* and the evaluation of the [CovMat] elements
    /// only happens when they are requested (and happens *every* time they
    /// are requested).
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