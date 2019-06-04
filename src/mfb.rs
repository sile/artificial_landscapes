use crate::{Interval, Objective};
use rand::distributions::{Distribution, Normal};
use rand::{self, Rng as _};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

pub type FidelityLevel = f64; // 0..10000

#[derive(Debug)]
pub struct Mfb<F, E, C> {
    f: F,
    e: E,
    c: C,
    levels: Vec<FidelityLevel>,
}
impl<F, E, C> Mfb<F, E, C>
where
    F: Objective,
    E: ResolutionError,
    C: Cost,
{
    pub fn new(f: F, e: E, c: C, levels: Vec<FidelityLevel>) -> Self {
        Self { f, e, c, levels }
    }
}
// impl<F, E, C> Objective for Mfb<F, E, C>
// where
//     F: Objective,
//     E: ResolutionError,
//     C: Cost,
// {
//     type Output = Outputs;

//     fn input_domain(&self) -> &[Interval] {
//         self.f.input_domain()
//     }

//     fn evaluate(&self, xs: &[f64]) -> Self::Output {
//         self.f.evaluate(xs) + self.e.error(xs, phi)
//     }
// }

#[derive(Debug)]
pub struct ModifiedRastrigin {
    input_domain: Vec<Interval>,
}
impl ModifiedRastrigin {
    pub fn new(dimensions: NonZeroUsize) -> Self {
        let input_domain = (0..dimensions.get())
            .map(|_| unsafe { Interval::new_unchecked(-1.0, 1.0) })
            .collect();
        Self { input_domain }
    }
}
impl Objective for ModifiedRastrigin {
    type Output = f64;

    fn input_domain(&self) -> &[Interval] {
        &self.input_domain
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        xs.iter()
            .map(|&x| x.powi(2) + 1.0 - (10.0 * PI * x).cos())
            .sum()
    }
}

pub trait Cost {
    fn cost(&self, phi: FidelityLevel) -> u64;
}

#[derive(Debug)]
pub struct LinearCost;
impl Cost for LinearCost {
    fn cost(&self, phi: FidelityLevel) -> u64 {
        phi as u64
    }
}

#[derive(Debug)]
pub struct NonLinearCost;
impl Cost for NonLinearCost {
    fn cost(&self, phi: FidelityLevel) -> u64 {
        (0.001 * phi).powi(4) as u64
    }
}

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

#[derive(Debug)]
pub struct ResolutionError3;
impl ResolutionError3 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        assert!(0.0 <= phi);
        assert!(phi < 10_000.0);

        if phi < 1_000.0 {
            1.0 - 0.0002 * phi
        } else if phi < 2_000.0 {
            0.8
        } else if phi < 3_000.0 {
            1.2 - 0.0002 * phi
        } else if phi < 4_000.0 {
            0.6
        } else if phi < 5_000.0 {
            1.4 - 0.0002 * phi
        } else if phi < 6_000.0 {
            0.4
        } else if phi < 7_000.0 {
            1.6 - 0.0002 * phi
        } else if phi < 8_000.0 {
            0.2
        } else if phi < 9_000.0 {
            1.8 - 0.0002 * phi
        } else {
            0.0
        }
    }
}
impl ResolutionError for ResolutionError3 {
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
pub struct ResolutionError4 {
    global_minimum: f64, // TODO: Vec<f64>
}
impl ResolutionError4 {
    pub const fn new(global_minimum: f64) -> Self {
        Self { global_minimum }
    }

    fn theta(&self, phi: FidelityLevel) -> f64 {
        1.0 - 0.0001 * phi
    }

    fn psi(&self, x: f64) -> f64 {
        // TODO: consider input domain
        1.0 - (x - self.global_minimum).abs()
    }
}
impl ResolutionError for ResolutionError4 {
    fn a(&self, x: f64, phi: FidelityLevel) -> f64 {
        self.theta(phi) * self.psi(x)
    }

    fn w(&self, phi: FidelityLevel) -> f64 {
        10.0 * PI * self.theta(phi)
    }

    fn b(&self, phi: FidelityLevel) -> f64 {
        0.5 * PI * self.theta(phi)
    }
}

pub trait StochasticError {
    fn mu(&self, xs: &[f64], phi: FidelityLevel) -> f64;

    fn sigma(&self, phi: FidelityLevel) -> f64;

    fn error(&self, xs: &[f64], phi: FidelityLevel) -> f64 {
        let mut rng = rand::thread_rng(); // TODO:
        let distribution = Normal::new(self.mu(xs, phi), self.sigma(phi));
        distribution.sample(&mut rng)
    }
}

#[derive(Debug)]
pub struct StochasticError1;
impl StochasticError1 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        1.0 - 0.0001 * phi
    }
}
impl StochasticError for StochasticError1 {
    fn mu(&self, _xs: &[f64], _phi: FidelityLevel) -> f64 {
        0.0 // TODO: generalize
    }

    fn sigma(&self, phi: FidelityLevel) -> f64 {
        0.1 * self.theta(phi)
    }
}

#[derive(Debug)]
pub struct StochasticError2;
impl StochasticError2 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        (-0.0005 * phi).exp()
    }
}
impl StochasticError for StochasticError2 {
    fn mu(&self, _xs: &[f64], _phi: FidelityLevel) -> f64 {
        0.0 // TODO: generalize
    }

    fn sigma(&self, phi: FidelityLevel) -> f64 {
        0.1 * self.theta(phi)
    }
}

#[derive(Debug)]
pub struct StochasticError3 {
    global_optimum: Vec<f64>,
}
impl StochasticError3 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        1.0 - 0.0001 * phi
    }

    fn gamma(&self, xs: &[f64]) -> f64 {
        xs.iter()
            .zip(self.global_optimum.iter())
            .map(|(&x, &xo)| 1.0 - (x - xo).abs())
            .sum()
    }
}
impl StochasticError for StochasticError3 {
    fn mu(&self, xs: &[f64], phi: FidelityLevel) -> f64 {
        (0.1 * self.theta(phi) / xs.len() as f64) * self.gamma(xs)
    }

    fn sigma(&self, phi: FidelityLevel) -> f64 {
        0.1 * self.theta(phi)
    }
}

#[derive(Debug)]
pub struct StochasticError4 {
    global_optimum: Vec<f64>,
}
impl StochasticError4 {
    fn theta(&self, phi: FidelityLevel) -> f64 {
        (-0.0005 * phi).exp()
    }

    fn gamma(&self, xs: &[f64]) -> f64 {
        xs.iter()
            .zip(self.global_optimum.iter())
            .map(|(&x, &xo)| 1.0 - (x - xo).abs())
            .sum()
    }
}
impl StochasticError for StochasticError4 {
    fn mu(&self, xs: &[f64], phi: FidelityLevel) -> f64 {
        (0.1 * self.theta(phi) / xs.len() as f64) * self.gamma(xs)
    }

    fn sigma(&self, phi: FidelityLevel) -> f64 {
        0.1 * self.theta(phi)
    }
}
pub trait InstabilityError {
    fn p(&self, phi: FidelityLevel) -> f64;

    fn l(&self, xs: &[f64]) -> f64;

    fn error(&self, xs: &[f64], phi: FidelityLevel) -> f64 {
        let mut rng = rand::thread_rng(); // TODO
        let r = rng.gen_range(0.0, 1.0);
        if r <= self.p(phi) {
            self.l(xs)
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct InstabilityError1;
impl InstabilityError for InstabilityError1 {
    fn p(&self, phi: FidelityLevel) -> f64 {
        0.1 * (1.0 - 0.0001 * phi)
    }

    fn l(&self, xs: &[f64]) -> f64 {
        (10 * xs.len()) as f64 // TODO: generalize
    }
}

#[derive(Debug)]
pub struct InstabilityError2;
impl InstabilityError for InstabilityError2 {
    fn p(&self, phi: FidelityLevel) -> f64 {
        (-0.001 * phi - 0.1).exp()
    }

    fn l(&self, xs: &[f64]) -> f64 {
        (10 * xs.len()) as f64
    }
}
