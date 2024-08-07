use rao::*;
use fitrs::{Fits, Hdu};

fn main() {
    const DMPITCH: f64 = 0.125;
    const WFSPITCH: f64 = 0.25;
    const NACTUX: usize = 64;
    const NSUBX: usize = 32;
    const NWFS: usize = 4;
    const AS2RAD: f64 = 4.848e-6;
    const THETAX: [f64; NWFS] = [-10.0, -10.0,  10.0,  10.0];
    const THETAY: [f64; NWFS] = [-10.0,  10.0, -10.0,  10.0];
    const COUPLING: f64 = 0.3;

    let mut actuators: Vec<rao::Actuator> = vec![];
    let mut phase_measurements: Vec<rao::Measurement> = vec![];
    for i in 0..NACTUX {
        for j in 0..NACTUX {
            let x = ((j as f64) - NACTUX as f64 / 2.0)*DMPITCH;
            let y = ((i as f64) - NACTUX as f64 / 2.0)*DMPITCH;
            actuators.push(
                Actuator::Gaussian{
                    sigma: coupling_to_sigma(COUPLING, DMPITCH),
                    position: Vec3D::new(x, y, 0.0),
                }
            );
            phase_measurements.push(
                Measurement::Phase {
                    line: Line::new_on_axis(x, y),
                },
            );
        }
    }

    const M: Measurement = Measurement::Zero;
    let mut slope_measurements = [M; NSUBX*NSUBX*NWFS*2];
    let mut idx = 0;
    for w in 0..NWFS {
        for i in 0..NSUBX {
            for j in 0..NSUBX {
                let x0 = ((j as f64) - NSUBX as f64/2.0 + 0.5) * WFSPITCH;
                let y0 = ((i as f64) - NSUBX as f64/2.0 + 0.5) * WFSPITCH;
                let xz = THETAX[w]*AS2RAD;
                let yz = THETAY[w]*AS2RAD;
                let line = Line::new(x0,xz,y0,yz);
                slope_measurements[idx] = Measurement::SlopeTwoEdge{
                    central_line: line.clone(),
                    edge_separation: WFSPITCH,
                    edge_length: WFSPITCH,
                    npoints: 1,
                    gradient_axis: Vec2D::x_unit(),
                };
                slope_measurements[idx+NSUBX*NSUBX] = Measurement::SlopeTwoEdge{
                    central_line: line.clone(),
                    edge_separation: WFSPITCH,
                    edge_length: WFSPITCH,
                    npoints: 1,
                    gradient_axis: Vec2D::y_unit(),
                };
                idx += 1;
            }
        }
        idx += NSUBX*NSUBX;
    }

    let cov = VonKarmanLayer::new(0.1, 25.0, 0.0);

    println!("creating imat (DM to slopes)");
    let mat = IMat::new(&slope_measurements, &actuators);
    let shape = [mat.ncols(), mat.nrows()];
    let data: Vec<f64> = mat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/ultimate_dmc.fits", primary_hdu)
    .expect("Failed to create");

    println!("creating imat ts (DM to phase)");
    let mat = IMat::new(&phase_measurements, &actuators);
    let shape = [mat.ncols(), mat.nrows()];
    let data: Vec<f64> = mat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/ultimate_dtc.fits", primary_hdu)
    .expect("Failed to create");

    println!("creating cmm (meas<->meas)");
    let mat = CovMat::new(&slope_measurements, &slope_measurements, &cov);
    let shape = [mat.ncols(), mat.nrows()];
    let data: Vec<f64> = mat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/ultimate_cmm.fits", primary_hdu)
    .expect("Failed to create");

    println!("creating ctm (phase<->meas)");
    let mat = CovMat::new(&phase_measurements, &slope_measurements, &cov);
    let shape = [mat.ncols(), mat.nrows()];
    let data: Vec<f64> = mat.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/ultimate_ctm.fits", primary_hdu)
    .expect("Failed to create");
}
