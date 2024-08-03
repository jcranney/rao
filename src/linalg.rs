use rayon::prelude::*;
use std::fmt;

pub trait Matrix {
    fn nrows(&self)->usize;
    fn ncols(&self)->usize;
    fn eval(&self, row_index: usize, col_index: usize)->f64;
    
    /// Return the (C-format / row-major) flattened matrix, e.g., to
    /// be saved to disk.
    fn flattened_array(&self) -> Vec<f64> where Self:Sync {
        (0..self.nrows())
        .into_par_iter()
        .map(move |row_index|
            (0..self.ncols())
            .into_par_iter()
            .map(move |col_index| 
                self.eval(row_index, col_index)
            ).collect::<Vec<f64>>()
        )
        .flatten()
        .collect()
    }
    
    /// Return the (C-format / row-major) interaction matrix as a [Vec<Vec<64>>]
    fn matrix(&self) -> Vec<Vec<f64>> where Self:Sync {
        (0..self.nrows())
        .into_par_iter()
        .map(|row_index|
            (0..self.ncols())
            .into_par_iter()
            .map(|col_index| 
                self.eval(row_index, col_index)
            ).collect::<Vec<f64>>()
        ).collect()
    }

    /// format function, which can be used when implementing Display
    fn format(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for row_index in 0..self.nrows() {
            match row_index {
                0 => write!(f, "[[")?,
                _ => write!(f, "\n [")?,
            }    
            for col_index in 0..self.ncols() {
                write!(f, " {:5.2}", self.eval(row_index, col_index))?;
            }
            write!(f, " ]")?;
        }
        write!(f, "]")?;
        Ok(())
    }
}