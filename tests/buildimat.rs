use rao::*;
use fitrs::{Fits, Hdu};

#[test]
fn phane_gaussian() {
    let actuators: Vec<rao::Actuator> = (0..1000)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Actuator::Gaussian{
            sigma: coupling_to_sigma(0.5,1.0),
            position: Vec3D::new(posx, 0.0, 0.0),
        }
    ).collect();
    
    let measurements: Vec<rao::Measurement> = (0..1000)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Measurement::Phase{
            line: Line::new(posx, 0.0, 0.0, 0.0)
        }
    ).collect();

    let imat = IMat::new(&measurements, &actuators);
    let shape = [measurements.len(), actuators.len()];
    let data: Vec<f64> = imat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/dev/null", primary_hdu).expect("Failed to create");
}

#[test]
fn edgeslope_gaussian() {
    let actuators: Vec<rao::Actuator> = (0..100)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Actuator::Gaussian{
            sigma: coupling_to_sigma(0.5,1.0),
            position: Vec3D::new(posx, 0.0, 0.0),
        }
    ).collect();
    
    let measurements: Vec<rao::Measurement> = (0..100)
    .map(|idx| idx as f64 * 0.2)
    .map(|posx|
        Measurement::Slope{
            line: Line::new(posx, 0.0, 0.0, 0.0),
            method: SlopeMethod::TwoEdge{
                edge_separation: 0.2,
                edge_length: 0.2,
                npoints: 5,
                gradient_axis: Vec2D::x_unit(),
            }
        }
    ).collect();

    let imat = IMat::new(&measurements, &actuators);
    let shape = [measurements.len(), actuators.len()];
    let data: Vec<f64> = imat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/dev/null", primary_hdu).expect("Failed to create");
}
