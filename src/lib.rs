use std::fmt;
use rayon::prelude::*;
#[macro_use] extern crate impl_ops;
use std::ops;


#[derive(Debug,Clone)]
pub struct PlaneCoordinate {
    x: f64,
    y: f64,
}
impl PlaneCoordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Self {x,y}
    }
    pub fn x_unit() -> Self {
        Self {x:1.0, y:0.0}
    }
    pub fn y_unit() -> Self {
        Self {x:0.0, y:0.0}
    }
    pub fn length(&self) -> f64 {
        (self.x.powf(2.0)+self.y.powf(2.0)).powf(0.5)
    }
}

impl_op_ex!(+
    |a:&PlaneCoordinate,b:&PlaneCoordinate| -> PlaneCoordinate
    {
        PlaneCoordinate {
            x: a.x + b.x,
            y: a.y + b.y,
        }
    }
);
impl_op_ex!(-
    |a:&PlaneCoordinate,b:&PlaneCoordinate| -> PlaneCoordinate
    {
        PlaneCoordinate {
            x: a.x - b.x,
            y: a.y - b.y,
        }
    }
);



#[derive(Debug,Clone)]
pub struct ElementCoordinate {
    x: f64,  // position in metres relative to optical axis
    y: f64,  // position in metres relative to optical axis
    z: f64,  // conjugation altitude (0km == Pupil, +infty == Object)
}

impl ElementCoordinate {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {x,y,z}
    }
    pub fn origin() -> Self {
        Self {x:0.0, y:0.0, z: 0.0}
    }
    pub fn distance_at_altitude(&self, line: &Line) -> f64 {
        let line_intersection = line.position_at_altitude(self.z);
        let dist = (line_intersection.x - self.x).powf(2.0)
                  +(line_intersection.y - self.y).powf(2.0);
        dist.powf(0.5)
    }
    pub fn displacement_at_altitude(&self, line: &Line) -> PlaneCoordinate {
        let line_intersection = line.position_at_altitude(self.z);
        PlaneCoordinate{
            x: (line_intersection.x - self.x),
            y: (line_intersection.y - self.y),
        }
    }
}

impl_op_ex_commutative!(+
    |a:&ElementCoordinate,b:&PlaneCoordinate| -> ElementCoordinate
    {
        ElementCoordinate {x:a.x+b.x,y:a.y+b.y,z:a.z}
    }
);
impl_op_ex_commutative!(+
    |a:&Line,b:&PlaneCoordinate| -> Line
    {
        match a {
            Line::OpticalAxis => Line::Parametric{
                x0: b.x,
                xz: 0.0,
                y0: b.y,
                yz: 0.0,
            },
            Line::TwoPoint(coorda,coordb) => Line::TwoPoint(
                ElementCoordinate{
                    x: coorda.x + b.x,
                    y: coorda.y + b.y,
                    z: coorda.z,
                },
                ElementCoordinate{
                    x: coordb.x + b.x,
                    y: coordb.y + b.y,
                    z: coordb.z,
                }
            ),
            Line::Parametric{x0,xz,y0,yz} => Line::Parametric{
                x0: x0+b.x,
                xz: *xz,
                y0: y0+b.y,
                yz: *yz,
            },
        }
    }
);



#[derive(Debug)]
pub enum Line {
    OpticalAxis,
    TwoPoint(ElementCoordinate, ElementCoordinate),
    Parametric { // where z = z, x = x0+xz*z, y = y0+yz*z
        x0: f64,
        xz: f64,
        y0: f64,
        yz: f64,
    }
}

impl Line {
    fn position_at_altitude(&self, alt: f64) -> PlaneCoordinate {
        match self {
            Self::OpticalAxis => PlaneCoordinate{x:0.0, y:0.0},
            Self::TwoPoint(a,b) => {
                let t = (alt-a.z)/(b.z - a.z);
                let x = a.x + t*(b.x - a.x);
                let y = a.y + t*(b.y - a.y);
                PlaneCoordinate{x, y}
            },
            Self::Parametric{x0,xz,y0,yz} => {
                PlaneCoordinate {
                    x: alt*xz + x0,
                    y: alt*yz + y0,
                }
            },
        }
    }
}

#[derive(Debug)]
pub enum Actuator{
    Zero,
    Gaussian {
        coupling: f64,  // coupling per metre, not traditional coupling per pitch
        position: ElementCoordinate,
    }
}

impl Actuator {
    fn phase(&self, line: &Line) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::Gaussian {
                coupling,
                position,
            } => {
                let distance = position.distance_at_altitude(line);
                Self::gaussian(distance, (coupling).ln())
            }
        }
    }
    fn slope(&self, line: &Line, method: &SlopeMethod) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::Gaussian {
                coupling,
                position,
            } => {
                match method {
                    SlopeMethod::Axial{gradient_axis} => {
                        let displacement = position.displacement_at_altitude(line);
                        let c = coupling.ln();
                        let fxy = Self::gaussian2d(displacement.x, displacement.y, c);
                        2.0*fxy*c*(displacement.x*gradient_axis.x + displacement.y*gradient_axis.y)        
                    },
                    SlopeMethod::TwoPoint{neg, pos} => {
                        let c = coupling.ln();
                        let distance_neg = (position).distance_at_altitude(&(line+neg));
                        let f_neg = Self::gaussian(distance_neg, c);
                        let distance_pos = (position).distance_at_altitude(&(line+pos));
                        let f_pos = Self::gaussian(distance_pos, c);
                        (f_pos - f_neg)/(pos-neg).length()
                    },
                    _ => todo!(),
                }
            }
        }
    }
    fn gaussian(x: f64, a: f64) -> f64 {
        (a*(x).powf(2.0)).exp()
    }
    fn gaussian2d(x: f64, y: f64, a: f64) -> f64 {
        (a*(x.powf(2.0)+y.powf(2.0))).exp()
    }
}


#[derive(Debug)]
pub enum SlopeMethod {
    Axial {
        gradient_axis: PlaneCoordinate,
    },
    TwoPoint {
        neg: PlaneCoordinate,
        pos: PlaneCoordinate,
    },
    TwoEdges,
    Area,
}

#[derive(Debug)]
pub enum Measurement{
    Zero,
    Phase {
        line: Line,
    },
    Slope {
        line: Line,
        method: SlopeMethod,
    },
}

#[derive(Debug)]
pub struct Imat<'a> {
    actuators: &'a [Actuator],
    measurements: &'a [Measurement],
}

impl<'a> Imat<'a>{
    pub fn new(actuators: &'a [Actuator], measurements: &'a [Measurement]) -> Self {
        Imat {
            actuators,
            measurements,
        }
    }
    pub fn element(&self, actu_idx: usize, meas_idx: usize) -> f64 {
        let actuator: &Actuator = &self.actuators[actu_idx];
        let measurement: &Measurement = &self.measurements[meas_idx];
        match (actuator, measurement) {
            (Actuator::Zero, _) => 0.0,
            (_, Measurement::Zero) => 0.0,
            (_, Measurement::Phase{line}) => actuator.phase(line),
            (_,Measurement::Slope{line, method}) => {
                actuator.slope(line, method)
            }
        }
    }
    pub fn flatarray(&self) -> Vec<f64> {
        let results = (0..self.actuators.len())
        .into_par_iter()
        .map(|actu_idx|
            (0..self.measurements.len())
            .into_iter()
            .map(|meas_idx| 
                self.element(actu_idx, meas_idx)
            ).collect::<Vec<f64>>()
        )
        .flatten()
        .collect();
        results
    }
}

impl fmt::Display for Imat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for actu_idx in 0..self.actuators.len() {
            match actu_idx {
                0 => write!(f, "[[")?,
                _ => write!(f, "\n [")?,
            }    
            for meas_idx in 0..self.measurements.len() {
                write!(f, " {:5.2}", self.element(actu_idx, meas_idx))?;
            }
            write!(f, " ]")?;
        }
        write!(f, "]")?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn always_pass() {
        assert!(true);
    }

    #[test]
    fn gaussian_on_axis_phase() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(0.0, 0.0, 0.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::OpticalAxis
            }
        ];
        let imat = Imat::new(&actuators, &measurements);
        assert_eq!(imat.element(0,0), 1.0);
    }

    #[test]
    fn gaussian_off_axis_phase() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(0.0, 0.0, 1000.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::Parametric {
                    x0: 0.0,
                    xz: 1.0/1000.0,
                    y0: 0.0,
                    yz: 0.0,
                }
            }
        ];
        let imat = Imat::new(&actuators, &measurements);
        assert_eq!(imat.element(0,0), 0.5);
    }

    #[test]
    fn gaussian_off_axis_phase_twopoint() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(0.0, 0.0, 1000.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::TwoPoint(
                    ElementCoordinate::new(1.0,1.0,0.0),
                    ElementCoordinate::new(1.0,-1.0,2000.0),
                )
            }
        ];
        let imat = Imat::new(&actuators, &measurements);
        assert_eq!(imat.element(0,0), 0.5);
    }

    #[test]
    fn simple_symmetric() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(0.0, 0.0, 1000.0),
            },
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(1.0, 0.0, 1000.0),
            },
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::Parametric {
                    x0: 0.0,
                    y0: 0.0,
                    xz: 0.0,
                    yz: 0.0,
                }
            },
            Measurement::Phase{
                line: Line::Parametric {
                    x0: 1.0,
                    y0: 0.0,
                    xz: 0.0,
                    yz: 0.0,
                }
            }
        ];
        let imat = Imat::new(&actuators, &measurements);
        assert!(imat.element(0,0)>0.0);
        assert_eq!(imat.element(0,0),imat.element(1,1));
        assert_eq!(imat.element(1,0),imat.element(0,1));
    }

    #[test]
    fn slope_axial() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::origin(),
            }
        ];
        let measurements = [0.0,0.5,1.0,1.5,2.0].map(|x|
            Measurement::Slope{
                line: Line::Parametric{
                    x0: x,
                    xz: 0.0,
                    y0: 0.0,
                    yz: 0.0,
                },
                method: SlopeMethod::Axial{
                    gradient_axis: PlaneCoordinate::x_unit(),
                },
            }
        );
        let imat = Imat::new(&actuators, &measurements);
        println!("{}",imat);
        // if we assume the influence function is 0.5^(x^2), then the
        // gradient at x=1 should be ln(0.5) ~= -0.69314718...
        assert_eq!(imat.element(0,2),(0.5_f64).ln());
    }

    #[test]
    fn slope_twopoint() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::origin(),
            }
        ];
        let measurements = [0.0,0.5,1.0,1.5,2.0].map(|x|
            Measurement::Slope{
                line: Line::Parametric{
                    x0: x,
                    xz: 0.0,
                    y0: 0.0,
                    yz: 0.0,
                },
                method: SlopeMethod::TwoPoint{
                    neg: PlaneCoordinate{
                        x: -1e-4,
                        y: 0.0,
                    },
                    pos: PlaneCoordinate{
                        x: 1e-4,
                        y: 0.0,
                    }
                }
            }
        );
        let imat = Imat::new(&actuators, &measurements);
        println!("{}",imat);
        // if we assume the influence function is 0.5^(x^2), then the
        // gradient at x=1 should be ln(0.5) ~= -0.69314718...
        assert!((imat.element(0,2)-(0.5_f64).ln()).abs() < 1e-7);
    }
}
