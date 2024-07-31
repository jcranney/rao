use rao::*;
use fitrs::{Fits, Hdu};

#[test]
fn main() {
    let actuators: Vec<rao::Actuator> = (0..1000)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Actuator::Gaussian{
            coupling: 0.5*0.2,
            position: ElementCoordinate::new(posx, 0.0, 0.0),
        }
    ).collect();
    
    let measurements: Vec<rao::Measurement> = (0..1000)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Measurement::Phase{
            line: Line::Parametric{
                x0: posx,
                xz: 0.0,
                y0: 0.0,
                yz: 0.0,
            },
        }
    ).collect();

    let imat = Imat::new(&actuators, &measurements);
    let shape = [measurements.len(), actuators.len()];
    let data: Vec<f64> = imat.flatarray();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("new_file.fits", primary_hdu).expect("Failed to create");
}

