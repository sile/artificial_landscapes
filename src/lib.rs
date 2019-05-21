pub use self::ackley::Ackley;
use std::num::NonZeroUsize;

mod ackley;

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
    fn global_minimum(&self) -> f64;
    fn evaluate(&self, xs: &[f64]) -> f64;

    fn dimension(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.input_domain().len()).unwrap_or_else(|| panic!())
    }
}
