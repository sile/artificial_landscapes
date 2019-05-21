use crate::{Interval, SingleObjectiveProblem};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

/// Ackley.
///
/// # References
///
/// - [BenchmarkFcns: Ackley Function](http://benchmarkfcns.xyz/benchmarkfcns/ackleyfcn.html)
#[derive(Debug, Clone)]
pub struct Ackley {
    input_domain: Vec<Interval>,
}
impl Ackley {
    /// Makes a new `Ackley` instance.
    pub fn new(dimension: NonZeroUsize) -> Self {
        let input_domain = (0..dimension.get())
            .map(|_| unsafe { Interval::new_unchecked(-32.0, 32.0) })
            .collect();
        Self { input_domain }
    }
}
impl SingleObjectiveProblem for Ackley {
    fn input_domain(&self) -> &[Interval] {
        &self.input_domain
    }

    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), self.dimension().get());

        const A: f64 = 20.0;
        const B: f64 = 0.2;
        const C: f64 = 2.0 * PI;

        let n = xs.len() as f64;

        let temp0 = -B * (xs.iter().map(|&x| x * x).sum::<f64>() / n).sqrt();
        let temp1 = xs.iter().map(|&x| (C * x).cos()).sum::<f64>() / n;
        -A * temp0.exp() - temp1.exp() + A + 1f64.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ackley_1dim_works() {
        let data = [
            ([22.65758825365655], 21.92521364455731),
            ([3.9694607169472533], 11.007798457945992),
            ([-30.469959101822425], 22.2986824594293),
            ([-1.8101696468489337], 7.3466524181661015),
            ([14.560335747731948], 21.236067777924614),
            ([-22.422217132786436], 22.07905020522178),
            ([15.188977084281639], 20.305800636107715),
            ([-16.785269743490744], 20.775703244150886),
            ([13.934792698367325], 18.983749979008575),
            ([14.760948499910342], 20.602622683330253),
        ];

        let f = Ackley::new(unsafe { NonZeroUsize::new_unchecked(1) });
        for (x, y) in &data {
            assert_eq!(f.evaluate(x), *y);
        }
    }

    #[test]
    fn ackley_3dim_works() {
        let xs = [
            [18.719632463482156, -0.6431917600828285, -1.6420925390764083],
            [-24.817099865163662, -0.7678978587176744, 11.256566726292299],
            [24.74164440209877, 30.989548376880073, 0.3580624196288227],
            [28.776175552457232, 20.523823521128882, 29.237727357019942],
            [
                -6.0075786570959195,
                -21.598615917324736,
                -6.7801928353539935,
            ],
            [10.849630928460776, -30.805317816408774, 24.974975154745387],
            [-21.32067491951709, -31.974200080684923, 25.585930725139583],
            [-28.173568319960786, -28.715037273701363, 6.5827299192418565],
            [-12.098217704277282, 13.412887012806117, -14.388327805014782],
            [-17.85912015987642, -17.095441918255453, -5.18480283164304],
        ];
        let ys = [
            19.8182593057153,
            20.68583853841581,
            21.401415214188678,
            21.83881635654065,
            20.247646030235106,
            20.652010759915264,
            21.71681391368819,
            21.72538192075483,
            20.56310057087851,
            19.779121186267872,
        ];

        let f = Ackley::new(unsafe { NonZeroUsize::new_unchecked(3) });
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(f.evaluate(x), *y);
        }
    }
}
