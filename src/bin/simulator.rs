use rao::*;
use fitrs::{Fits, Hdu};

fn main() {
    const AS2RAD: f64 = 4.848e-6;
    const NPHISAMPLES: u32 = 64;
    const NSUBX: u32 = 40;
    const NTSSAMPLES: u32 = 80;
    const NACTUX: u32 = 41;
    
    /////////////
    // define phi related coordinates:
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0, 0.0),
        &Vec2D::new( 4.0, 0.0),
        NPHISAMPLES,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0),
        &Vec2D::new( 0.0,  4.0),
        NPHISAMPLES,
    );
    let phi_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
    let phi: Vec<Measurement> = phi_coords
    .iter()
    .map(|p0|
        Measurement::Phase{
            line: Line::new_on_axis(p0.x,p0.y)
        }
    ).collect();
    
    let phip1: Vec<Measurement> = phi_coords
    .iter()
    .map(|p0|
        Measurement::Phase{
            line: Line::new_on_axis(p0.x+0.005,p0.y)
        }
    ).collect();
    
    /////////////
    // define measurement related coordinates:
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0, 0.0),
        &Vec2D::new( 4.0, 0.0),
        NSUBX,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0),
        &Vec2D::new( 0.0,  4.0),
        NSUBX,
    );
    let meas_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let wfs_dirs = vec![
        Vec2D::new(-10.0, -10.0),
        Vec2D::new(-10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
    ];
    let meas: Vec<Measurement> = wfs_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        meas_coords
        .iter().map(move |p|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                },
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                }
            ]
        ).flatten()
    ).flatten().collect();

    /////////////
    // define truth sensor related coordinates:
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0, 0.0),
        &Vec2D::new( 4.0, 0.0),
        NTSSAMPLES,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0),
        &Vec2D::new( 0.0,  4.0),
        NTSSAMPLES,
    );
    let ts_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let ts_dirs = vec![
        Vec2D::new(0.0, 0.0),
    ];
    let ts: Vec<Measurement> = ts_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        ts_coords
        .iter().map(move |p|
            Measurement::Phase{
                line: Line::new(p.x, dir.x, p.y, dir.y),
            }
        )
    ).flatten().collect();

    /////////////
    // define actuator related coordinates:
    let sf = NACTUX as f64 / (NACTUX-1) as f64;
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0*sf, 0.0),
        &Vec2D::new( 4.0*sf, 0.0),
        NACTUX,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0*sf),
        &Vec2D::new( 0.0,  4.0*sf),
        NACTUX,
    );
    let com_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let com: Vec<Actuator> = com_coords
    .iter()
    .map(move |p|
        Actuator::Gaussian{
            position: Vec3D::new(p.x, p.y, 0.0),
            sigma: coupling_to_sigma(0.3, 8.0/(NACTUX as f64 - 1.0)),
        }
    ).collect();

    let cov_model = VonKarmanLayer::new(0.15, 25.0, 0.0);
    
    let c_phi_phi = CovMat::new(&phi, &phi, &cov_model);
    save_mat("/tmp/c_phi_phi.fits", c_phi_phi);

    let c_phip1_phi = CovMat::new(&phip1, &phi, &cov_model);
    save_mat("/tmp/c_phip1_phi.fits", c_phip1_phi);
    
    let c_meas_phi = CovMat::new(&meas, &phi, &cov_model);
    save_mat("/tmp/c_meas_phi.fits", c_meas_phi);
    
    let c_meas_meas = CovMat::new(&meas, &meas, &cov_model);
    save_mat("/tmp/c_meas_meas.fits", c_meas_meas);
    
    let c_ts_meas = CovMat::new(&ts, &meas, &cov_model);
    save_mat("/tmp/c_ts_meas.fits", c_ts_meas);
    
    let d_meas_com = IMat::new(&meas, &com);
    save_mat("/tmp/d_meas_com.fits", d_meas_com);
    
    let d_ts_com = IMat::new(&ts, &com);
    save_mat("/tmp/d_ts_com.fits", d_ts_com);
    
    let d_phi_com = IMat::new(&phi, &com);
    save_mat("/tmp/d_phi_com.fits", d_phi_com);
}

fn save_mat(filename: &str, matrix: (impl Matrix + std::marker::Sync)) {
    println!("doing {}", filename);
    let shape = [matrix.ncols(), matrix.nrows()];
    let data: Vec<f64> = matrix.flattened_array();
    let primary_hdu = Hdu::new(&shape, data);
    Fits::create(filename, primary_hdu).expect("Failed to create");
}