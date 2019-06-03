use std::f64::consts::PI;

pub type FidelityLevel = f64; // 0..10000

pub trait ResolutionError {
    /// Maximum error.
    fn a(&self, x: f64, phi: FidelityLevel) -> f64;

    /// Determines the number of local optima.
    fn w(&self, phi: FidelityLevel) -> f64;

    /// Influences the offset of the global optimum.
    fn b(&self, phi: FidelityLevel) -> f64;

    fn error(&self, xs: &[f64], phi: FidelityLevel) -> f64 {
        xs.iter()
            .map(|&x| self.a(x, phi) * (self.w(phi) * x + self.b(phi) + PI).cos())
            .sum()
    }
}

#[derive(Debug)]
pub struct ResolutionError1;
impl ResolutionError1 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        1.0 - 0.0001 * phi
    }
}
impl ResolutionError for ResolutionError1 {
    fn a(&self, _x: f64, phi: FidelityLevel) -> f64 {
        self.theta(phi)
    }

    fn w(&self, phi: FidelityLevel) -> f64 {
        10.0 * PI * self.theta(phi)
    }

    fn b(&self, phi: FidelityLevel) -> f64 {
        0.5 * PI * self.theta(phi)
    }
}

#[derive(Debug)]
pub struct ResolutionError2;
impl ResolutionError2 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        (-0.00025 * phi).exp()
    }
}
impl ResolutionError for ResolutionError2 {
    fn a(&self, _x: f64, phi: FidelityLevel) -> f64 {
        self.theta(phi)
    }

    fn w(&self, phi: FidelityLevel) -> f64 {
        10.0 * PI * self.theta(phi)
    }

    fn b(&self, phi: FidelityLevel) -> f64 {
        0.5 * PI * self.theta(phi)
    }
}
