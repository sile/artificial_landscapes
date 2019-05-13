use crate::Property::{self, *};
use std::num::NonZeroUsize;

/// Ackley.
///
/// # References
///
/// - [BenchmarkFcns: Ackley Function](benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html)
#[derive(Debug, Clone)]
pub struct Ackley {
    dimension: NonZeroUsize,
}
impl Ackley {
    pub const PROPERTIES: &'static [Property] = &[Continous, Multimodal, Differentiable];

    /// Makes a new `Ackley` instance.
    pub const fn new(dimension: NonZeroUsize) -> Self {
        Self { dimension }
    }

    /// Evaluates the given values.
    pub fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(self.dimension, xs.len());
    }
}
