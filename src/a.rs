use crate::{Interval, SingleObjective};
use std::f64::consts::{E, PI};
use std::num::NonZeroUsize;

const fn interval(low: f64, high: f64) -> Interval {
    unsafe { Interval::new_unchecked(low, high) }
}

/// Ackley Function.
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
            .map(|_| interval(-32.0, 32.0))
            .collect();
        Self { input_domain }
    }
}
impl SingleObjective for Ackley {
    fn input_domain(&self) -> &[Interval] {
        &self.input_domain
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), self.dimension().get());

        const A: f64 = 20.0;
        const B: f64 = 0.2;
        const C: f64 = 2.0 * PI;

        let n = xs.len() as f64;

        let temp0 = -B * (xs.iter().map(|&x| x * x).sum::<f64>() / n).sqrt();
        let temp1 = xs.iter().map(|&x| (C * x).cos()).sum::<f64>() / n;
        -A * temp0.exp() - temp1.exp() + A + E
    }
}

/// Ackley N. 2 Function.
///
/// # References
///
/// - [BenchmarkFcns: Ackley N. 2 Function](http://http://benchmarkfcns.xyz/benchmarkfcns/ackleyn2fcn.html)
#[derive(Debug, Clone)]
pub struct AckleyN2;
impl SingleObjective for AckleyN2 {
    fn input_domain(&self) -> &[Interval] {
        const I: Interval = interval(-32.0, 32.0);
        &[I, I]
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), 2);

        -200.0 * (-0.2 * (xs[0].powi(2) + xs[1].powi(2)).sqrt()).exp()
    }
}

/// Ackley N. 3 Function.
///
/// # References
///
/// - [BenchmarkFcns: Ackley N. 3 Function](http://http://benchmarkfcns.xyz/benchmarkfcns/ackleyn3fcn.html)
#[derive(Debug, Clone)]
pub struct AckleyN3;
impl SingleObjective for AckleyN3 {
    fn input_domain(&self) -> &[Interval] {
        const I: Interval = interval(-32.0, 32.0);
        &[I, I]
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), 2);

        AckleyN2.evaluate(xs) + 5.0 * ((3.0 * xs[0]).cos() + (3.0 * xs[1]).sin()).exp()
    }
}

/// Ackley N. 4 Function.
///
/// # References
///
/// - [BenchmarkFcns: Ackley N. 4 Function](http://http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html)
#[derive(Debug, Clone)]
pub struct AckleyN4 {
    input_domain: Vec<Interval>,
}
impl AckleyN4 {
    /// Makes a new `AckleyN4` instance.
    pub fn new(dimension: NonZeroUsize) -> Self {
        let input_domain = (0..dimension.get())
            .map(|_| interval(-35.0, 35.0))
            .collect();
        Self { input_domain }
    }
}
impl SingleObjective for AckleyN4 {
    fn input_domain(&self) -> &[Interval] {
        &self.input_domain
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), self.dimension().get());

        (0..xs.len() - 1)
            .map(|i| {
                let x0 = xs[i];
                let x1 = xs[i + 1];
                let a = (x0 * x0 + x1 * x1).sqrt();
                let b = 3.0 * ((2.0 * x0).cos() + (2.0 * x1).sin());
                (-0.2f64).exp() * a + b
            })
            .sum()
    }
}

/// Adjiman Function.
///
/// # References
///
/// - [BenchmarkFcns: Adjiman Function](http://http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html)
#[derive(Debug, Clone)]
pub struct Adjiman;
impl SingleObjective for Adjiman {
    fn input_domain(&self) -> &[Interval] {
        const X: Interval = interval(-1.0, 2.0);
        const Y: Interval = interval(-1.0, 1.0);
        &[X, Y]
    }

    fn evaluate(&self, xs: &[f64]) -> f64 {
        assert_eq!(xs.len(), 2);

        xs[0].cos() * xs[1].sin() - xs[0] / (xs[1].powi(2) + 1.0)
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

    #[test]
    fn ackley_n2_works() {
        let global_minimum = -200.0;
        assert_eq!(AckleyN2.evaluate(&[0.0, 0.0]), global_minimum);
    }

    #[test]
    fn ackley_n4_works() {
        let global_minimum = -4.5901006651507235;
        let f = AckleyN4::new(unsafe { NonZeroUsize::new_unchecked(2) });
        assert_eq!(f.evaluate(&[-1.51, -0.755]), global_minimum);
    }

    #[test]
    fn adjiman_works() {
        let global_minimum = -2.0218067833370204;
        assert_eq!(Adjiman.evaluate(&[2.0, 0.10578]), global_minimum);

        let xs = [
            [1.6988039632644485, 0.19813149356712767],
            [1.9495584023968218, -0.9390715549110211],
            [1.001954306865192, 0.6260208920119086],
            [-0.04794751315575374, 0.797488825309429],
            [-0.93529327876237, 0.46046782236002093],
            [1.6553628440694266, 0.2125948350545459],
            [0.2165112459358971, 0.5655298124927721],
            [1.2011847204083157, -0.9633588309312502],
            [-0.7206031794145588, 0.0931198693185391],
            [1.9424144068370417, -0.15917902447315369],
        ];
        let ys = [
            -1.659762578387234,
            -0.7375681290602631,
            -0.4042329508407953,
            0.7440898402043912,
            1.0354432243712317,
            -1.601603391660295,
            0.3593067922342904,
            -0.9196315410004139,
            0.7842782459976936,
            -1.836855958099792,
        ];
        let f = Adjiman;
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(f.evaluate(x), *y);
        }
    }
}
