use super::costs::ModelCosts;

/// How an agent specifies which model to use.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// A specific model ID with optional estimated costs.
    Exact { id: String, costs: Option<ModelCosts> },
    /// Use the parent agent's model.
    Inherit,
}

impl ModelSpec {
    pub fn parse(s: &str) -> Self {
        match s {
            "inherit" => Self::Inherit,
            other => Self::Exact { id: other.to_string(), costs: None },
        }
    }

    pub fn resolve(&self, parent_model: &str) -> String {
        match self {
            Self::Exact { id, .. } => id.clone(),
            Self::Inherit => parent_model.to_string(),
        }
    }

    pub fn costs(&self) -> ModelCosts {
        match self {
            Self::Exact { costs: Some(c), .. } => *c,
            _ => ModelCosts::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_exact_returns_id() {
        let spec = ModelSpec::Exact { id: "custom".into(), costs: None };
        assert_eq!(spec.resolve("parent"), "custom");
    }

    #[test]
    fn resolve_inherit_uses_parent() {
        assert_eq!(ModelSpec::Inherit.resolve("my-parent-model"), "my-parent-model");
    }

    #[test]
    fn parse_inherit() {
        assert!(matches!(ModelSpec::parse("inherit"), ModelSpec::Inherit));
    }

    #[test]
    fn parse_exact_model_id() {
        match ModelSpec::parse("claude-sonnet-4-20250514") {
            ModelSpec::Exact { id, costs } => {
                assert_eq!(id, "claude-sonnet-4-20250514");
                assert!(costs.is_none());
            }
            other => panic!("Expected Exact, got {other:?}"),
        }
    }

    #[test]
    fn costs_returns_zero_when_none() {
        let spec = ModelSpec::Exact { id: "m".into(), costs: None };
        assert_eq!(spec.costs().estimate(1_000_000, 1_000_000), 0.0);
    }

    #[test]
    fn costs_returns_value_when_set() {
        let spec = ModelSpec::Exact {
            id: "m".into(),
            costs: Some(ModelCosts::new(3.0, 15.0)),
        };
        assert!((spec.costs().estimate(1_000_000, 0) - 3.0).abs() < 0.001);
    }
}
