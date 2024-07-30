use rao::*;

fn main() {
    let mut actuators = vec![];
    let mut measurements = vec![];
    for af in (0..10).map(|a| a as f64 * 0.2) {
        actuators.push(Actuator::Gaussian{
            coupling: 0.5*0.2,
            position: ElementCoordinate::new(af, 0.0, 0.0),
        });
    }
    for mf in (0..10).map(|m| m as f64 * 0.2) {
        measurements.push(Measurement::Phase{
            line: Line::Parametric{
                x0: mf,
                xz: 0.0,
                y0: 0.0,
                yz: 0.0,
            }
        });
    }
    let imat = Imat::new(&actuators, &measurements);
    println!("{}", imat);
}

