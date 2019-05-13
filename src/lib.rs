pub use self::ackley::Ackley;
pub use self::property::Property;

mod ackley;
mod property;

pub trait GlobalMinimum {
    fn global_minimum(&self) -> f64;
}

pub trait Multimodal {}
pub trait Convex {}
pub trait Differentiable {}
pub trait Separable {}
pub trait Asymmetric {}
pub trait Constrained {}

pub struct Interval {
    min: f64,
    max: f64,
}
impl Interval {
    pub fn new(min: f64, max: f64) -> Option<Self> {
        Some(Self { min, max })
    }
}

pub trait InputDomain {
    fn input_domain(&self) -> &[Interval];
}

pub trait SingleObjective: InputDomain {
    fn global_minimum(&self) -> f64;
    fn evaluate(&self, xs: &[f64]) -> f64;
}

pub trait MultiObjective: InputDomain {
    fn evaluate(&self, xs: &[f64]) -> &[f64];
}
