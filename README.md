![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)
![docs.rs](https://img.shields.io/docsrs/rao)
![GitHub License](https://img.shields.io/github/license/jcranney/rao)
![Crates.io Version](https://img.shields.io/crates/v/rao)


# rao

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

## pyrao
There is a Python-wrapper for this package. This is less customisable, but allows for quick construction of interaction matrices and covariance matrices directly from Python. Check it out here: [github.com/jcranney/pyrao](https://github.com/jcranney/pyrao), or install it with pip:
```bash
pip install rao
```


## Examples
For the latest and most up-to-date examples, see [docs.rs/rao](https://docs.rs/rao/latest/rao/).

## Contributing
RAO is being actively developed, primarily as a tool for testing design features of [MAVIS](https://mavis-ao.org/), but all design choices are being left as general as possible. If you find a missing feature that would be useful for your purposes, please [create an issue on github](https://github.com/jcranney/rao/issues). If you have implemented extra functionality or fixed any bugs yourself, please [create a pull requiest](https://github.com/jcranney/rao/pulls) and I'd be happy to review/pull it.

## Known Limitations
 - Pupil boundary conditions. The functionality exists to sample a geometrically defined pupil at any resolution, but the measurements don't interact with this edge at all. This affects the high order sensing near the edge of the pupil, and the low order sensing everywhere.


License: MIT
