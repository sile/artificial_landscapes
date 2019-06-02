//! **M**ulti-**F**idelity **S**ingle **O**bjective functions.
//!
//! # References
//!
//! - [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
use crate::{Interval, Objective};
use std::f64::consts::PI;
use std::fmt;
use std::iter;

const ZERO_TO_ONE: Interval = unsafe { Interval::new_unchecked(0.0, 1.0) };

pub trait MultiFidelitySingleObjective: Objective<Output = Outputs> {
    fn max_cost(&self) -> Cost {
        let xs = self
            .input_domain()
            .iter()
            .map(|i| i.min())
            .collect::<Vec<_>>();
        self.evaluate(&xs).last().unwrap_or_else(|| panic!()).0
    }
}

pub type Cost = u64;

pub struct Outputs(Box<dyn Iterator<Item = (Cost, f64)>>);
impl Outputs {
    pub fn new<I>(inner: I) -> Self
    where
        I: 'static + Iterator<Item = (Cost, f64)>,
    {
        Self(Box::new(inner))
    }
}
impl Iterator for Outputs {
    type Item = (Cost, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
impl fmt::Debug for Outputs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Outputs(_)")
    }
}

/// Currin exponential function (2 fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug)]
pub struct CurrinExponential;
impl CurrinExponential {
    fn f2(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];

        let a = 1.0 - (-1.0 / 2.0 * x2).exp();
        let b = 2300.0 * x1.powi(3) + 1900.0 * x1.powi(2) + 2092.0 * x1 + 60.0;
        let c = 100.0 * x1.powi(3) + 500.0 * x1.powi(2) + 4.0 * x1 + 20.0;
        a * (b / c)
    }

    fn f1(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];

        let a = self.f2(&[x1 + 0.05, x2 + 0.05]) / 4.0;
        let b = self.f2(&[x1 + 0.05, 0f64.max(x2 - 0.05)]) / 4.0;
        let c = self.f2(&[x1 - 0.05, x2 + 0.05]) / 4.0;
        let d = self.f2(&[x1 - 0.05, 0f64.max(x2 - 0.05)]) / 4.0;
        a + b + c + d
    }
}
impl Objective for CurrinExponential {
    type Output = Outputs;

    fn input_domain(&self) -> &[Interval] {
        const DOMAIN: [Interval; 2] = [ZERO_TO_ONE, ZERO_TO_ONE];
        &DOMAIN
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        Outputs::new(iter::once((1, self.f1(xs))).chain(iter::once((10, self.f2(xs)))))
    }
}
impl MultiFidelitySingleObjective for CurrinExponential {}

/// Park function (2 fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug)]
pub struct Park;
impl Park {
    fn f2(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];
        let x3 = xs[2];
        let x4 = xs[3];

        let a = x1 / 2.0;
        let b = (1.0 + (x2 + x3.powi(2)) * (x4 / x1.powi(2))).sqrt() - 1.0;
        let c = (x1 + 3.0 * x4) * (1.0 + x3.sin()).exp();
        a * b + c
    }

    fn f1(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];
        let x3 = xs[2];

        let a = (1.0 + x1.sin() / 10.0) * self.f2(xs);
        let b = 2.0 * x1.powi(2) + x2.powi(2) + x3.powi(2) + 0.5;
        a - b
    }
}
impl Objective for Park {
    type Output = Outputs;

    fn input_domain(&self) -> &[Interval] {
        const DOMAIN: [Interval; 4] = [ZERO_TO_ONE, ZERO_TO_ONE, ZERO_TO_ONE, ZERO_TO_ONE];
        &DOMAIN
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        Outputs::new(iter::once((1, self.f1(xs))).chain(iter::once((10, self.f2(xs)))))
    }
}
impl MultiFidelitySingleObjective for Park {}

/// Borehole function (2 fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug)]
pub struct Borehole;
impl Borehole {
    fn f2(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];
        let x3 = xs[2];
        let x4 = xs[3];
        let x5 = xs[4];
        let x6 = xs[5];
        let x7 = xs[6];
        let x8 = xs[7];

        let a = 2.0 * PI * x3 * (x4 - x6);
        let b = (x2 / x1).ln();
        let c = 1.0 + (2.0 * x7 * x3) / (b * x1.powi(2) * x8) + x3 / x5;
        a / (b * c)
    }

    fn f1(&self, xs: &[f64]) -> f64 {
        let x1 = xs[0];
        let x2 = xs[1];
        let x3 = xs[2];
        let x4 = xs[3];
        let x5 = xs[4];
        let x6 = xs[5];
        let x7 = xs[6];
        let x8 = xs[7];

        let a = 5.0 * x3 * (x4 - x6);
        let b = (x2 / x1).ln();
        let c = 1.5 + (2.0 * x7 * x3) / (b * x1.powi(2) * x8) + x3 / x5;
        a / (b * c)
    }
}
impl Objective for Borehole {
    type Output = Outputs;

    fn input_domain(&self) -> &[Interval] {
        const DOMAIN: [Interval; 8] = unsafe {
            [
                Interval::new_unchecked(0.05, 0.15),
                Interval::new_unchecked(100.0, 50_000.0),
                Interval::new_unchecked(63_070.0, 115_600.0),
                Interval::new_unchecked(990.0, 1_110.0),
                Interval::new_unchecked(63.1, 116.0),
                Interval::new_unchecked(700.0, 820.0),
                Interval::new_unchecked(1_120.0, 1_680.0),
                Interval::new_unchecked(9_855.0, 12_045.0),
            ]
        };
        &DOMAIN
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        Outputs::new(iter::once((1, self.f1(xs))).chain(iter::once((10, self.f2(xs)))))
    }
}
impl MultiFidelitySingleObjective for Borehole {}
