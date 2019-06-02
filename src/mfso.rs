//! **M**ulti-**F**idelity **S**ingle **O**bjective functions.
//!
//! # References
//!
//! - [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
use crate::Interval;
use std::fmt;
use std::iter;
use std::num::NonZeroUsize;

const ZERO_TO_ONE: Interval = unsafe { Interval::new_unchecked(0.0, 1.0) };

pub trait Objective {
    type Output;

    fn input_domain(&self) -> &[Interval];
    fn evaluate(&self, xs: &[f64]) -> Self::Output;

    fn dimension(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.input_domain().len()).unwrap_or_else(|| panic!())
    }
}

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

/// Currin exponential function.
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
