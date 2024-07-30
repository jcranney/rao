use std::fmt;

struct PlaneCoordinate {
    x: f64,
    y: f64,
}

#[derive(Debug)]
pub struct ElementCoordinate {
    x: f64,  // position in metres relative to optical axis
    y: f64,  // position in metres relative to optical axis
    z: f64,  // conjugation altitude (0km == Pupil, +infty == Object)
}

impl ElementCoordinate {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {x,y,z}
    }
    pub fn distance_at_altitude(&self, line: &Line) -> f64 {
        let line_intersection = line.position_at_altitude(self.z);
        let dist = (line_intersection.x - self.x).powf(2.0)
                  +(line_intersection.y - self.y).powf(2.0);
        dist.powf(0.5)
    }
}


#[derive(Debug)]
pub enum Line {
    OpticalAxis,
    TwoPoints(ElementCoordinate, ElementCoordinate),
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
            Self::TwoPoints(a,b) => {
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
    fn gaussian(x: f64, a: f64) -> f64 {
        (a*(x).powf(2.0)).exp()
    }
}


#[derive(Debug)]
pub enum Measurement{
    Zero,
    Phase {
        line: Line,
    },
    Slope,
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
            (_, Measurement::Phase{line}) => {
                actuator.phase(line)
            },
            (_, Measurement::Slope) => todo!()
        }
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
    fn gaussian_off_axis_phase_twopoints() {
        let actuators = [
            Actuator::Gaussian{
                coupling: 0.5,
                position: ElementCoordinate::new(0.0, 0.0, 1000.0),
            }
        ];
        let measurements = [
            Measurement::Phase{
                line: Line::TwoPoints(
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
}
