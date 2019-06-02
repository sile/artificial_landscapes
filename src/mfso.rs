//! **M**ulti-**F**idelity **S**ingle **O**bjective functions.
//!
//! # References
//!
//! - [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
use crate::{Interval, Objective};
use std::f64::consts::PI;
use std::fmt;
use std::iter;
use std::num::{NonZeroU64, NonZeroU8};

const ZERO_TO_ONE: Interval = unsafe { Interval::new_unchecked(0.0, 1.0) };

const ONE: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(1) };
const TEN: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(10) };

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

pub type Cost = NonZeroU64;

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
pub struct CurrinExponential {
    cost_factor: NonZeroU64,
}
impl Default for CurrinExponential {
    fn default() -> Self {
        Self::new(TEN)
    }
}
impl CurrinExponential {
    pub const fn new(cost_factor: NonZeroU64) -> Self {
        Self { cost_factor }
    }

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
        Outputs::new(
            iter::once((ONE, self.f1(xs))).chain(iter::once((self.cost_factor, self.f2(xs)))),
        )
    }
}
impl MultiFidelitySingleObjective for CurrinExponential {
    fn max_cost(&self) -> Cost {
        self.cost_factor
    }
}

/// Park function (2 fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug)]
pub struct Park {
    cost_factor: NonZeroU64,
}
impl Default for Park {
    fn default() -> Self {
        Self::new(TEN)
    }
}
impl Park {
    pub const fn new(cost_factor: NonZeroU64) -> Self {
        Self { cost_factor }
    }

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
        Outputs::new(
            iter::once((TEN, self.f1(xs))).chain(iter::once((self.cost_factor, self.f2(xs)))),
        )
    }
}
impl MultiFidelitySingleObjective for Park {
    fn max_cost(&self) -> Cost {
        self.cost_factor
    }
}

/// Borehole function (2 fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug)]
pub struct Borehole {
    cost_factor: NonZeroU64,
}
impl Default for Borehole {
    fn default() -> Self {
        Self::new(TEN)
    }
}
impl Borehole {
    pub const fn new(cost_factor: NonZeroU64) -> Self {
        Self { cost_factor }
    }

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
        Outputs::new(
            iter::once((ONE, self.f1(xs))).chain(iter::once((self.cost_factor, self.f2(xs)))),
        )
    }
}
impl MultiFidelitySingleObjective for Borehole {
    fn max_cost(&self) -> Cost {
        self.cost_factor
    }
}

/// Hartmann-3D function (multi fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug, Clone)]
pub struct Hartmann3d {
    max_fidelity: NonZeroU8,
    cost_factor: NonZeroU64,
}
impl Default for Hartmann3d {
    fn default() -> Self {
        Self::new(unsafe { NonZeroU8::new_unchecked(3) }, TEN)
    }
}
impl Hartmann3d {
    pub const fn new(max_fidelity: NonZeroU8, cost_factor: NonZeroU64) -> Self {
        Self {
            max_fidelity,
            cost_factor,
        }
    }

    fn alpha(&self, m: u8, i: usize) -> f64 {
        const ALPHA: [f64; 4] = [1.0, 1.2, 3.0, 3.2];
        const DELTA: [f64; 4] = [0.01, -0.01, -0.1, 0.1];

        ALPHA[i] + (self.max_fidelity.get() - m) as f64 * DELTA[i]
    }

    fn f(&self, m: u8, xs: &[f64]) -> f64 {
        const A: [[f64; 3]; 4] = [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ];
        const P: [[f64; 3]; 4] = [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0381, 0.5743, 0.8828],
        ];

        (0..4)
            .map(|i| {
                let a = self.alpha(m, i);
                let b = (0..3)
                    .map(|j| A[i][j] * (xs[j] - P[i][j]).powi(2))
                    .sum::<f64>();
                a * (-b).exp()
            })
            .sum::<f64>()
    }
}
impl Objective for Hartmann3d {
    type Output = Outputs;

    fn input_domain(&self) -> &[Interval] {
        const DOMAIN: [Interval; 3] = [ZERO_TO_ONE, ZERO_TO_ONE, ZERO_TO_ONE];
        &DOMAIN
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        let xs = Vec::from(xs);
        let this = self.clone();
        Outputs::new((0..self.max_fidelity.get()).map(move |m| {
            let v = this.f(m + 1, &xs);
            let c = unsafe { NonZeroU64::new_unchecked(this.cost_factor.get().pow(u32::from(m))) };
            (c, v)
        }))
    }
}
impl MultiFidelitySingleObjective for Hartmann3d {
    fn max_cost(&self) -> Cost {
        unsafe {
            NonZeroU64::new_unchecked(
                self.cost_factor
                    .get()
                    .pow(u32::from(self.max_fidelity.get())),
            )
        }
    }
}

/// Hartmann-6D function (multi fidelity).
///
/// See: [Multi-fidelity Gaussian Process Bandit Optimisation](https://arxiv.org/abs/1603.06288)
#[derive(Debug, Clone)]
pub struct Hartmann6d {
    max_fidelity: NonZeroU8,
    cost_factor: NonZeroU64,
}
impl Default for Hartmann6d {
    fn default() -> Self {
        Self::new(unsafe { NonZeroU8::new_unchecked(4) }, TEN)
    }
}
impl Hartmann6d {
    pub const fn new(max_fidelity: NonZeroU8, cost_factor: NonZeroU64) -> Self {
        Self {
            max_fidelity,
            cost_factor,
        }
    }

    fn alpha(&self, m: u8, i: usize) -> f64 {
        const ALPHA: [f64; 4] = [1.0, 1.2, 3.0, 3.2];
        const DELTA: [f64; 4] = [0.01, -0.01, -0.1, 0.1];

        ALPHA[i] + (self.max_fidelity.get() - m) as f64 * DELTA[i]
    }

    fn f(&self, m: u8, xs: &[f64]) -> f64 {
        const A: [[f64; 6]; 4] = [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ];
        const P: [[f64; 6]; 4] = [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ];

        (0..4)
            .map(|i| {
                let a = self.alpha(m, i);
                let b = (0..6)
                    .map(|j| A[i][j] * (xs[j] - P[i][j]).powi(2))
                    .sum::<f64>();
                a * (-b).exp()
            })
            .sum::<f64>()
    }
}
impl Objective for Hartmann6d {
    type Output = Outputs;

    fn input_domain(&self) -> &[Interval] {
        const DOMAIN: [Interval; 6] = [
            ZERO_TO_ONE,
            ZERO_TO_ONE,
            ZERO_TO_ONE,
            ZERO_TO_ONE,
            ZERO_TO_ONE,
            ZERO_TO_ONE,
        ];
        &DOMAIN
    }

    fn evaluate(&self, xs: &[f64]) -> Self::Output {
        let xs = Vec::from(xs);
        let this = self.clone();
        Outputs::new((0..self.max_fidelity.get()).map(move |m| {
            let v = this.f(m + 1, &xs);
            let c = unsafe { NonZeroU64::new_unchecked(this.cost_factor.get().pow(u32::from(m))) };
            (c, v)
        }))
    }
}
impl MultiFidelitySingleObjective for Hartmann6d {
    fn max_cost(&self) -> Cost {
        unsafe {
            NonZeroU64::new_unchecked(
                self.cost_factor
                    .get()
                    .pow(u32::from(self.max_fidelity.get())),
            )
        }
    }
}
