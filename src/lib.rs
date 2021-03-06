//! Test functions
//!
//! # References
//!
//! - [A Literature Survey of Benchmark Functions For Global Optimization Problems](https://arxiv.org/abs/1308.4008)
//! - [BenchmarkFcns](http://http://benchmarkfcns.xyz/fcns)
pub use self::a::{Ackley, AckleyN2, AckleyN3, AckleyN4};
use std::num::NonZeroUsize;

pub mod mfb;
pub mod mfso;

mod a;

pub trait Objective {
    type Output;

    fn input_domain(&self) -> &[Interval];
    fn evaluate(&self, xs: &[f64]) -> Self::Output;

    fn dimension(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.input_domain().len()).unwrap_or_else(|| panic!())
    }
}

pub trait GlobalOptimumInput {
    fn global_optimum_input(&self) -> &[f64];
}

#[derive(Debug, Clone, Copy)]
pub struct Interval {
    min: f64,
    max: f64,
}
impl Interval {
    pub fn new(min: f64, max: f64) -> Option<Self> {
        if min <= max {
            Some(Self { min, max })
        } else {
            None
        }
    }

    pub const unsafe fn new_unchecked(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    pub const fn min(&self) -> f64 {
        self.min
    }

    pub const fn max(&self) -> f64 {
        self.max
    }
}

pub trait SingleObjective {
    fn input_domain(&self) -> &[Interval];
    fn evaluate(&self, xs: &[f64]) -> f64;

    fn dimension(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.input_domain().len()).unwrap_or_else(|| panic!())
    }
}
