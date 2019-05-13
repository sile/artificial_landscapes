/// Property of test function.
///
/// # References
///
/// - [BenchmarkFcns](http://benchmarkfcns.xyz/fcns)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Property {
    Continous,
    Multimodal,
    Convex,
    Differentiable,
    Separable,
}
