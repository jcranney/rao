use std::ops;

/// 2D geometric vector, associated with transverse plane of optical
/// path.
///
/// This struct is used for defining displacements, coordinates, and directions
/// along the transverse plane (in a slice of the propogation axis).
/// # Examples
/// ```
/// let a = rao::Vec2D::new(3.0, 4.0);
/// assert_eq!(a.norm(),5.0);
/// let b = &a * 2.0;
/// assert_eq!(b.norm(),10.0);
/// let e_x = rao::Vec2D::x_unit();
/// let e_y = rao::Vec2D::y_unit();
/// assert_eq!(e_x.dot(&e_y), 0.0);
/// assert_eq!(e_x.dot(&e_x), 1.0);
/// assert_eq!(e_x.dot(&a), 3.0);
/// assert_eq!((e_x / 2.0).x, 0.5);
/// ```
#[derive(Debug,Clone,PartialEq)]
pub struct Vec2D {
    pub x: f64,
    pub y: f64,
}

impl Vec2D {
    /// Create a new [Vec2D] from the coordinates in the transverse plane.
    pub fn new(x: f64, y: f64) -> Self {
        Self {x,y}
    }
    /// create a unit vector along x-axis.
    pub fn x_unit() -> Self {
        Self {x:1.0, y:0.0}
    }
    /// create a unit vector along y-axis.
    pub fn y_unit() -> Self {
        Self {x:0.0, y:1.0}
    }
    /// return the (Euclidean) norm of the [Vec2D].
    pub fn norm(&self) -> f64 {
        self.norm2().powf(0.5)
    }
    /// return the (Euclidean) squared-norm of the [Vec2D].
    pub fn norm2(&self) -> f64 {
        self.x.powf(2.0)+self.y.powf(2.0)
    }
    /// return the dot (inner) product of the [Vec2D] with another (borrowed) [Vec2D].
    pub fn dot(&self, other: &Self) -> f64 {
        self.x*other.x+self.y*other.y
    }
    /// Calculate a uniformly spaced set of points between two [Vec2D]s.
    pub fn linspace(a: &Self, b: &Self, npoints: u32) -> Vec<Self> {
        (0..npoints)
        .map(|u| (u as f64 / npoints as f64) + 1.0/ (2.0 * npoints as f64))
        .map(|t| (1.0-t)*a + t*b)
        .collect()
    }
    /// Calculates a [Vec2D] that is rotated by +90 degrees, such that it is
    /// orthogonal to the input [Vec2D].
    pub fn ortho(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x
        }
    }
}

impl_op!(- |a:Vec2D| -> Vec2D { 
    Vec2D::new(-a.x, -a.y) 
});

impl_op_ex_commutative!(/ |a:&Vec2D,b:&f64| -> Vec2D {
    Vec2D {
        x: a.x / b,
        y: a.y / b,
    }
});
impl_op_ex_commutative!(* |a:&Vec2D,b:&f64| -> Vec2D {
    Vec2D {
        x: a.x * b,
        y: a.y * b,
    }
});
impl_op_ex!(+ |a:&Vec2D,b:&Vec2D| -> Vec2D {
    Vec2D {
        x: a.x + b.x,
        y: a.y + b.y,
    }
});
impl_op_ex!(- |a:&Vec2D,b:&Vec2D| -> Vec2D {
    Vec2D {
        x: a.x - b.x,
        y: a.y - b.y,
    }
});

impl_op_ex_commutative!(+
    |a:&Vec3D,b:&Vec2D| -> Vec3D
    {
        Vec3D {x:a.x+b.x,y:a.y+b.y,z:a.z}
    }
);
impl_op_ex_commutative!(+
    |a:&Line,b:&Vec2D| -> Line
    {
        Line::new(a.x0+b.x, a.xz, a.y0+b.y, a.yz)
    }
);


/// 3D geometric vector, associated with both transverse (x,y) and
/// propagation (z) dimensions.
///
/// This struct is used for defining coordinates of optical components that 
/// have a meaninful position along the optical propagation axis (e.g., an actuator
/// of a deformable mirror conjugated to some altitude). In particular this
/// struct is useful for finding the intersection of an optical ray (see [Line])
/// with the influence function of an actuator where that [Line] intersects the
/// altitude plane of the actuator.
/// # Examples
/// ```
/// // `point` at x=6.0 m, y=3.0 m, z=10.0 m
/// let point = rao::Vec3D::new(6.0, 3.0, 10.0);
/// // `line` with z-component of x = 1.0 metres/metre.
/// let line = rao::Line::new(0.0, 1.0, 0.0, 0.0);
/// // when `line` propagates to z=10.0 m, it will have
/// // x=10.0 m, y=0.0 m, which is +4.0 metres away from
/// // the `point` in x, and -3.0 metres away in y.
/// assert_eq!(
///     point.displacement_at_altitude(&line),
///     rao::Vec2D::new(4.0, -3.0)
/// );
/// assert_eq!(
///     point.distance_at_altitude(&line),
///     5.0
/// );
/// ```
#[derive(Debug,Clone,PartialEq)]
pub struct Vec3D {
    /// Position in x relative to the optical axis
    pub x: f64,
    /// Position in y relative to the optical axis
    pub y: f64,
    /// Position in z relative to the optical axis
    /// e.g., 0km => pupil-plane, +infinity => object-plane
    pub z: f64, 
}

impl Vec3D {
    /// Create a new [Vec3D] from a 3D coordinate.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {x,y,z}
    }
    /// Convenience function for creating a [Vec3D] at the origin.
    pub fn origin() -> Self {
        Self {x:0.0, y:0.0, z: 0.0}
    }
    /// Return the Euclidean norm of the displacement vector between self
    /// and the intersection of a [Line] at the altitude of self.
    pub fn distance_at_altitude(&self, line: &Line) -> f64 {
        self.displacement_at_altitude(line).norm()
    }
    /// Return the displacement vector between self
    /// and the intersection of a [Line] at the altitude of self.
    pub fn displacement_at_altitude(&self, line: &Line) -> Vec2D {
        let line_intersection = line.position_at_altitude(self.z);
        Vec2D{
            x: (line_intersection.x - self.x),
            y: (line_intersection.y - self.y),
        }
    }
}

/// Line that propagates along the optical axis.
///
/// The parametric form of the line as defined here is convenient for
/// finding the intersection of the line with a plane parallel with the
/// x-y plane, that is, points of constant altitude.
///
/// The `(x, y, z)` tuple is parameterised by `t`:
/// ```math
/// x = x0 + xz*t
/// y = y0 + yz*t
/// z = t
/// ```
#[derive(Debug, Clone)]
pub struct Line {
    /// The x-position at z=0.
    pub x0: f64,
    /// The component of the x-position which depends linearly on z.
    pub xz: f64,
    /// The y-position at z=0.
    pub y0: f64,
    /// The component of the y-position which depends linearly on z.
    pub yz: f64,
}

impl Line {
    /// Constructor for a new [Line].
    pub fn new(x0: f64, xz: f64, y0: f64, yz: f64) -> Line {
        Line {x0,xz,y0,yz}
    }
    /// Convenience function for [Line]s that do not depend on z
    pub fn new_on_axis(x0: f64, y0: f64) -> Line {
        Line::new(x0, 0.0, y0, 0.0)
    }
    /// Convenience function for defining a [Line] given two points
    /// in 3D. Results in singularities if a.z == b.z, otherwise is 
    /// gauaranteed to be stable.
    pub fn new_from_two_points(a: &Vec3D, b: &Vec3D) -> Line {
        let xz = (b.x - a.x)/(b.z - a.z);
        let yz = (b.y - a.y)/(b.z - a.z);
        let x0 = a.x - (a.z)/(b.z-a.z)*(b.x-a.x);
        let y0 = a.y - (a.z)/(b.z-a.z)*(b.y-a.y);
        Line::new(x0,xz,y0,yz)
    }
    /// Calculate the [Vec2D] coordinates in the transverse plane at
    /// a specified altitude.
    pub fn position_at_altitude(&self, alt: f64) -> Vec2D {
        Vec2D::new(alt*self.xz + self.x0, alt*self.yz + self.y0)
    }
    
    pub fn distance_at_ground(&self, other: &Line) -> f64 {
        Vec2D::new(self.x0 - other.x0, self.y0 - other.y0).norm()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq};

    #[test]
    fn linspace() {
        let a = Vec2D::new(1.0,2.0);
        let b = Vec2D::new(3.0,0.0);
        let ls = Vec2D::linspace(&a,&b,3);
        assert_abs_diff_eq!(ls[1].x, 2.0);
        assert_abs_diff_eq!(ls[1].y, 1.0);
    }

    #[test]
    fn adding() {
        let a = Vec2D::new(1.0,2.0);
        let b = Vec3D::new(10.0,20.0,30.0);
        let c = a + b;
        assert_abs_diff_eq!(c.x, 11.0);
        assert_abs_diff_eq!(c.y, 22.0);
        assert_abs_diff_eq!(c.z, 30.0);
    }
}
