use rao::{Actuator, IMat, Line, Matrix, Measurement, Vec2D, Vec3D, coupling_to_sigma};
use fitrs::{Hdu,Fits};

fn main() {
    const N_SUBX: u32 = 8;  // 8 x 8 subapertures
    const PITCH: f64 = 0.2;  // 0.2 metres gap between actuators
    const COUPLING: f64 = 0.5;  // coupling of neightbouring actuators
    
    // build list of actuators
    let mut actuators: Vec<Actuator> = vec![];
    for i in 0..=N_SUBX {
        for j in 0..=N_SUBX {
            let x = (f64::from(j) - f64::from(N_SUBX/2))*PITCH;
            let y = (f64::from(i) - f64::from(N_SUBX/2))*PITCH;
            actuators.push(
                // Gaussian influence functions
                Actuator::Gaussian{
                    // standard deviation defined by coupling and pitch
                    sigma: coupling_to_sigma(COUPLING, PITCH),
                    // position of actuator in 3D
                    position: Vec3D::new(x, y, 0.0),  // z-dimension is altitude
                }
            );
        }
    }
    
    // build list of measurements
    let mut measurements: Vec<Measurement> = vec![];
    for i in 0..N_SUBX {
        for j in 0..N_SUBX {
            let x0 = (f64::from(j) - f64::from(N_SUBX/2) + 0.5)*PITCH;
            let y0 = (f64::from(i) - f64::from(N_SUBX/2) + 0.5)*PITCH;
            let xz = 0.0;  // angular x-component (radians)
            let yz = 0.0;  // angular y-compenent (radians)
            // define the line that the subaperture looks through:
            let line = Line::new(x0,xz,y0,yz);
            // slope-style measurement for x-axis slope:
            // using the two edges of the subaperture to
            // define the slope:
            measurements.push(Measurement::SlopeTwoEdge{
                central_line: line.clone(),
                edge_separation: PITCH,
                edge_length: PITCH,
                // npoints to sample along edge of subap:
                npoints: 5,
                // desired axis of slope measurement
                gradient_axis: Vec2D::x_unit(),
            });
            // same, but for y-axis slope:
            measurements.push(Measurement::SlopeTwoEdge{
                central_line: line.clone(),
                edge_separation: PITCH,
                edge_length: PITCH,
                // npoints to sample along edge of subap:
                npoints: 5,
                // desired axis of slope measurement
                gradient_axis: Vec2D::y_unit(),
            });
        }
    }
    // imat is defined by any set of measurements and actuators
    let imat = IMat::new(&measurements, &actuators);
    let data: Vec<f64> = imat.flattened_array(); // serialise the imat for saving
    let shape = [measurements.len(), actuators.len()];
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create("/tmp/miniimat.fits", primary_hdu).expect("Failed to create");
}
