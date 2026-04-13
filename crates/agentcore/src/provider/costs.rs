/// Estimated costs rates per million tokens.
///
/// Estimated costs are calculated as:
/// `(input_tokens × input_rate + output_tokens × output_rate) / 1,000,000`
///
/// Cache tokens are not included. Actual billing may differ.
#[derive(Debug, Clone, Copy)]
pub struct ModelCosts {
    pub input_per_million: f64,
    pub output_per_million: f64,
}

impl ModelCosts {
    /// Set both input and output rates (USD per million tokens).
    pub fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
        }
    }

    /// Set only the input rate. Output defaults to 0.
    pub fn input(rate: f64) -> Self {
        Self {
            input_per_million: rate,
            output_per_million: 0.0,
        }
    }

    /// Set only the output rate. Input defaults to 0.
    pub fn output(rate: f64) -> Self {
        Self {
            input_per_million: 0.0,
            output_per_million: rate,
        }
    }

    /// No estimated costs.
    pub fn zero() -> Self {
        Self {
            input_per_million: 0.0,
            output_per_million: 0.0,
        }
    }

    /// Estimate costs from token counts.
    pub fn estimate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_million
            + output_tokens as f64 * self.output_per_million)
            / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_with_both_rates() {
        let costs = ModelCosts::new(3.0, 15.0);
        let est = costs.estimate(1_000_000, 500_000);
        assert!((est - 10.5).abs() < 0.001); // 3.0 + 7.5
    }

    #[test]
    fn estimate_input_only() {
        let costs = ModelCosts::input(3.0);
        assert!((costs.estimate(1_000_000, 1_000_000) - 3.0).abs() < 0.001);
    }

    #[test]
    fn estimate_zero() {
        let costs = ModelCosts::zero();
        assert_eq!(costs.estimate(1_000_000, 1_000_000), 0.0);
    }
}
