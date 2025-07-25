![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)
![docs.rs](https://img.shields.io/docsrs/rao)


# rao

## rao

`rao` - Adaptive Optics tools in Rust - is a set of fast and robust adaptive
optics utilities. The current scope of `rao` is for the calculation of
large matrices in AO, used in the configuration of real-time adaptive optics,
control. Specifically, we aim to provide fast, scalable, and reliable APIs for
generating:
 - `rao::IMat` - the interaction matrix between measurements and actuators,
 - `rao::CovMat` - the covariance matrix between measurements.

These two matrices are typically the largest computational burden in the
configuration of real-time control (RTC) for AO, and also the most
performance-sensitive parts of the RTC.

## Examples
Building an interaction matrix for a square-grid DM and a square-grid SH-WFS:
```rust
use crate::rao::Matrix;
const N_SUBX: i32 = 8;  // 8 x 8 subapertures
const PITCH: f64 = 0.2;  // 0.2 metres gap between actuators
const COUPLING: f64 = 0.5;  // actuator cross-coupling

// build list of measurements
let mut measurements = vec![];
for i in 0..N_SUBX {
    for j in 0..N_SUBX {
        let x0 = ((j-N_SUBX/2) as f64 + 0.5)*PITCH;
        let y0 = ((i-N_SUBX/2) as f64 + 0.5)*PITCH;
        let xz = 0.0;  // angular x-component (radians)
        let yz = 0.0;  // angular y-compenent (radians)
        // define the optical axis of subaperture
        let line = rao::Line::new(x0,xz,y0,yz);
        // slope-style measurement
        // x-slope
        measurements.push(rao::Measurement::SlopeTwoEdge{
            central_line: line.clone(),
            edge_separation: PITCH,
            edge_length: PITCH,
            npoints: 5,
            gradient_axis: rao::Vec2D::x_unit(),
            altitude: f64::INFINITY,
        });
        // y-slope
        measurements.push(rao::Measurement::SlopeTwoEdge{
            central_line: line.clone(),
            edge_separation: PITCH,
            edge_length: PITCH,
            npoints: 5,
            gradient_axis: rao::Vec2D::y_unit(),
            altitude: f64::INFINITY,
        });
    }
}

// build list of actuators
let mut actuators = vec![];
for i in 0..(N_SUBX+1) {
    for j in 0..(N_SUBX+1) {
        let x = ((j-N_SUBX/2) as f64)*PITCH;
        let y = ((i-N_SUBX/2) as f64)*PITCH;
        actuators.push(
            // Gaussian influence functions
            rao::Actuator::Gaussian{
                // std defined by coupling and pitch
                sigma: rao::coupling_to_sigma(COUPLING, PITCH),
                // position of actuator in 3D (z=altitude)
                position: rao::Vec3D::new(x, y, 0.0),
            }
        );
    }
}

// instanciate imat from (actu,meas)
let imat = rao::IMat::new(&measurements, &actuators);
// serialise imat for saving
let data: Vec<f64> = imat.flattened_array();
```

License: MIT
