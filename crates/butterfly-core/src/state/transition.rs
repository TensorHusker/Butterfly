//! Transition validation and proof obligations

use super::{ComponentState, StateError};
use std::collections::HashMap;
use std::fmt::Debug;

/// A transition rule that validates state changes
pub trait TransitionRule<S: ComponentState>: Send + Sync {
    /// Check if transition is allowed
    fn is_valid(&self, from: &S, to: &S) -> bool;

    /// Get reason why transition is invalid (if it is)
    fn invalid_reason(&self, from: &S, to: &S) -> Option<String>;

    /// Human-readable name for this rule
    fn name(&self) -> &str;
}

/// Set of transition rules for a state machine
pub struct TransitionRules<S: ComponentState> {
    rules: Vec<Box<dyn TransitionRule<S>>>,
}

impl<S: ComponentState> TransitionRules<S> {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule<R>(&mut self, rule: R)
    where
        R: TransitionRule<S> + 'static,
    {
        self.rules.push(Box::new(rule));
    }

    /// Validate a transition against all rules
    pub fn validate(&self, from: &S, to: &S) -> Result<(), StateError> {
        for rule in &self.rules {
            if !rule.is_valid(from, to) {
                return Err(StateError::InvalidTransition {
                    from: format!("{:?}", from),
                    to: format!("{:?}", to),
                    reason: rule
                        .invalid_reason(from, to)
                        .unwrap_or_else(|| format!("Transition rule '{}' failed", rule.name())),
                });
            }
        }
        Ok(())
    }
}

impl<S: ComponentState> Default for TransitionRules<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof obligation that must be satisfied before transition
pub trait ProofObligation<S: ComponentState>: Send + Sync {
    /// Verify proof obligation is satisfied
    fn verify(&self, context: &TransitionContext<S>) -> Result<(), StateError>;

    /// Description of what this proof obligation ensures
    fn description(&self) -> &str;
}

/// Context provided to proof obligations for verification
pub struct TransitionContext<S: ComponentState> {
    pub from: S,
    pub to: S,
    pub epoch: u64,
    pub metadata: HashMap<String, String>,
}

impl<S: ComponentState> TransitionContext<S> {
    pub fn new(from: S, to: S, epoch: u64) -> Self {
        Self {
            from,
            to,
            epoch,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test state for validation
    #[derive(Debug, Clone, PartialEq)]
    enum TestState {
        A,
        B,
        C,
    }

    impl ComponentState for TestState {
        fn component_type(&self) -> &str {
            "Test"
        }

        fn description(&self) -> &str {
            "Test state"
        }
    }

    // Test rule: Only A->B allowed
    struct OnlyAtoB;
    impl TransitionRule<TestState> for OnlyAtoB {
        fn is_valid(&self, from: &TestState, to: &TestState) -> bool {
            matches!((from, to), (TestState::A, TestState::B))
        }

        fn invalid_reason(&self, _from: &TestState, _to: &TestState) -> Option<String> {
            Some("Only A->B transition allowed".to_string())
        }

        fn name(&self) -> &str {
            "OnlyAtoB"
        }
    }

    #[test]
    fn test_transition_rules() {
        let mut rules = TransitionRules::new();
        rules.add_rule(OnlyAtoB);

        // Valid transition
        assert!(rules.validate(&TestState::A, &TestState::B).is_ok());

        // Invalid transition
        assert!(rules.validate(&TestState::A, &TestState::C).is_err());
        assert!(rules.validate(&TestState::B, &TestState::C).is_err());
    }

    #[test]
    fn test_transition_context() {
        let ctx = TransitionContext::new(TestState::A, TestState::B, 5)
            .with_metadata("reason", "test");

        assert_eq!(ctx.epoch, 5);
        assert_eq!(ctx.get_metadata("reason"), Some(&"test".to_string()));
    }
}
