use rao::*;
use fitrs::{Fits, Hdu};

fn main() {
    let mut actuators: Vec<rao::Actuator> = vec![];
    for i in 0..64 {
        for j in 0..64 {
            let x = ((j as f64) - 32.0)*0.125;
            let y = ((i as f64) - 32.0)*0.125;
            actuators.push(
                Actuator::Gaussian{
                    sigma: coupling_to_sigma(0.5, 0.125),
                    position: Vec3D::new(x, y, 10000.0),
                }
            );
        }
    }
    const M: Measurement = Measurement::Zero;
    let measurements = &mut [M; 32*32*4*2];
    let thetax: [f64; 4] = [-10.0, -10.0, 10.0, 10.0];
    let thetay: [f64; 4] = [-10.0, 10.0, -10.0, 10.0];
    let mut idx = 0;
    for w in 0..4 {
        for i in 0..32 {
            for j in 0..32 {
                let x0 = ((j as f64) - 15.5) * 0.25;
                let y0 = ((i as f64) - 15.5) * 0.25;
                let xz = thetax[w]*4.848e-6;
                let yz = thetay[w]*4.848e-6;
                let line = Line::new(x0,xz,y0,yz);
                measurements[idx] = Measurement::Slope{
                    line: line.clone(),
                    method: SlopeMethod::TwoEdge{
                        edge_separation: 0.25,
                        edge_length: 0.25,
                        npoints: 4,
                        gradient_axis: Vec2D::x_unit(),
                    }
                };
                measurements[idx+32*32] = Measurement::Slope{
                    line: line.clone(),
                    method: SlopeMethod::TwoEdge{
                        edge_separation: 0.25,
                        edge_length: 0.25,
                        npoints: 4,
                        gradient_axis: Vec2D::y_unit(),
                    }
                };
                idx += 1;
            }
        }
        idx += 32*32;
    }
    let imat = IMat::new(measurements, &actuators);
    let shape = [measurements.len(), actuators.len()];
    let data: Vec<f64> = imat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/ultimate.fits", primary_hdu).expect("Failed to create");
}
