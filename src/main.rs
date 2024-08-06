use rao::*;
use fitrs::{Fits, Hdu};
use std::time;

fn main() {
    let now = time::Instant::now();
    let mut actuators = vec![];
    let mut measurements = vec![];
    for af in (0..3000).map(|a| a as f64 * 0.2) {
        actuators.push(Actuator::Gaussian{
            sigma: 0.5,
            position: Vec3D::new(af, 0.0, 0.0),
        });
    }
    for mf in (0..8000).map(|m| m as f64 * 0.2) {
        measurements.push(Measurement::SlopeTwoLine {
            line_pos: Line::new_on_axis(0.1+mf, 0.0),
            line_neg: Line::new_on_axis(-0.1+mf, 0.0),
        });
    }
    let imat = IMat::new(&measurements, &actuators);
    println!("\nBuilding synthetic interaction matrix");
    println!("nactu: {:10}", actuators.len());
    println!("nmeas: {:10}", measurements.len());
    let shape = [measurements.len(), actuators.len()];
    println!("{:10.2e} sec for initialising", 1e-6*(now.elapsed().as_micros() as f64));
    let now = time::Instant::now();
    let data: Vec<f64> = imat.flattened_array();
    println!("{:10.2e} sec for building imat", 1e-6*(now.elapsed().as_micros() as f64));
    let now = time::Instant::now();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("new_file.fits", primary_hdu).expect("Failed to create");
    println!("{:10.2e} sec for saving fits", 1e-6*(now.elapsed().as_micros() as f64));
}

