use rao::*;
use fitrs::{Fits, Hdu};
use std::time;


#[test]
fn phase_gaussian() {
    let actuators: Vec<rao::Actuator> = (0..1000)
    .map(|idx| f64::from(idx) * 0.2)
    .map(|posx|
        Actuator::Gaussian{
            sigma: coupling_to_sigma(0.5,1.0),
            position: Vec3D::new(posx, 0.0, 0.0),
        }
    ).collect();
    
    let measurements: Vec<rao::Measurement> = (0..1000)
    .map(|idx| f64::from(idx) * 0.2)
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
    .map(|idx| f64::from(idx) * 0.2)
    .map(|posx|
        Actuator::Gaussian{
            sigma: coupling_to_sigma(0.5,1.0),
            position: Vec3D::new(posx, 0.0, 0.0),
        }
    ).collect();
    
    let measurements: Vec<rao::Measurement> = (0..100)
    .map(|idx| f64::from(idx) * 0.2)
    .map(|posx|
        Measurement::SlopeTwoEdge{
            central_line: Line::new(posx, 0.0, 0.0, 0.0),
            edge_separation: 0.2,
            edge_length: 0.2,
            npoints: 5,
            gradient_axis: Vec2D::x_unit(),
        }
    ).collect();

    let imat = IMat::new(&measurements, &actuators);
    let shape = [measurements.len(), actuators.len()];
    let data: Vec<f64> = imat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/dev/null", primary_hdu).expect("Failed to create");
}

#[test]
fn imat_and_cov() {
    let now = time::Instant::now();
    
    let actuators: Vec<Actuator> =  
    (0..100)
    .map(|x| Actuator::Gaussian {
        sigma: coupling_to_sigma(0.5, 0.2),
        position: Vec3D::new(f64::from(x) * 0.2 + 0.1, 0.0, 0.0),
    }).collect();
    
    let measurements: Vec<Measurement> = 
    (0..100)
    .map(|x| Measurement::SlopeTwoLine {
            line_pos: Line::new_on_axis(f64::from(x) * 0.2 + 0.1, 0.0),
            line_neg: Line::new_on_axis(f64::from(x) * 0.2 - 0.1, 0.0),
    }).collect();
    
    // Build the covmat:
    let cov_model = VonKarmanLayer::new(
        0.1, // r0
        25.0, // L0
        0.0, // altitude
    );
    
    // Build the imat:
    let imat = IMat::new(&measurements, &actuators);
    println!("\nBuilding synthetic interaction matrix");
    println!("nactu: {:10}", actuators.len());
    println!("nmeas: {:10}", measurements.len());
    
    let shape = [imat.nrows(), imat.ncols()];
    println!("{:10.2e} sec for initialising", 1e-6*(now.elapsed().as_micros() as f64));
    
    let now = time::Instant::now();
    let data: Vec<f64> = imat.flattened_array();
    println!("{:10.2e} sec for building imat", 1e-6*(now.elapsed().as_micros() as f64));
    
    let now = time::Instant::now();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/toy_imat.fits", primary_hdu).expect("Failed to create");
    println!("{:10.2e} sec for saving fits", 1e-6*(now.elapsed().as_micros() as f64));
    
    let covmat = CovMat::new(&measurements, &measurements, &cov_model);
    println!("\nBuilding covariance matrix");
    
    let now = time::Instant::now();
    let data: Vec<f64> = covmat.flattened_array();
    println!("{:10.2e} sec for building imat", 1e-6*(now.elapsed().as_micros() as f64));
    
    let now = time::Instant::now();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/toy_covmat.fits", primary_hdu).expect("Failed to create");
    println!("{:10.2e} sec for saving fits", 1e-6*(now.elapsed().as_micros() as f64));
}

