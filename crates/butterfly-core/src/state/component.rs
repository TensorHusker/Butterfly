//! Trait for components that maintain state machines

use super::{StateTransitionRecord, TransitionTrigger};
use std::fmt::Debug;
use thiserror::Error;

/// Error types for state management
#[derive(Debug, Error)]
pub enum StateError {
    #[error("Invalid transition from {from:?} to {to:?}: {reason}")]
    InvalidTransition {
        from: String,
        to: String,
        reason: String,
    },

    #[error("Transition precondition failed: {0}")]
    PreconditionFailed(String),

    #[error("State invariant violated: {0}")]
    InvariantViolation(String),

    #[error("Concurrent modification detected")]
    ConcurrentModification,

    #[error("State persistence failed: {0}")]
    PersistenceError(String),
}

/// Trait for types that represent component state
pub trait ComponentState: Debug + Clone + PartialEq + Send + Sync + 'static {
    /// Type of component this state belongs to
    fn component_type(&self) -> &str;

    /// Human-readable description of this state
    fn description(&self) -> &str;

    /// Check if this state is a failure state
    fn is_failure_state(&self) -> bool {
        false
    }

    /// Check if this state is operational
    fn is_operational(&self) -> bool {
        !self.is_failure_state()
    }

    /// Check if this state is terminal (no more transitions possible)
    fn is_terminal(&self) -> bool {
        false
    }
}

/// Trait for components that manage state transitions
pub trait StateMachine: Send + Sync {
    /// The state type this machine manages
    type State: ComponentState;

    /// Get current state
    fn current_state(&self) -> Self::State;

    /// Attempt to transition to a new state
    fn transition_to(
        &mut self,
        new_state: Self::State,
        trigger: TransitionTrigger,
    ) -> Result<StateTransitionRecord<Self::State>, StateError>;

    /// Check if a transition is valid without performing it
    fn can_transition_to(&self, new_state: &Self::State) -> Result<(), StateError>;

    /// Get state transition history (last N transitions)
    fn transition_history(&self) -> &[StateTransitionRecord<Self::State>];

    /// Get time spent in current state
    fn time_in_current_state(&self) -> std::time::Duration;
}

/// Trait for state machines that support persistence
pub trait PersistentStateMachine: StateMachine {
    /// Serialize state to bytes
    fn serialize_state(&self) -> Result<Vec<u8>, StateError>;

    /// Deserialize state from bytes
    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), StateError>;

    /// Save state to persistent storage
    fn save_state(&self) -> Result<(), StateError> {
        let data = self.serialize_state()?;
        // Implementation-specific persistence logic
        // For now, return Ok - actual impl in specific crates
        let _ = data;
        Ok(())
    }

    /// Load state from persistent storage
    fn load_state(&mut self) -> Result<(), StateError> {
        // Implementation-specific loading logic
        // For now, return Ok - actual impl in specific crates
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example component state for testing
    #[derive(Debug, Clone, PartialEq)]
    enum TestState {
        Starting,
        Running,
        Stopped,
        Failed,
    }

    impl ComponentState for TestState {
        fn component_type(&self) -> &str {
            "Test"
        }

        fn description(&self) -> &str {
            match self {
                Self::Starting => "Starting up",
                Self::Running => "Running normally",
                Self::Stopped => "Stopped",
                Self::Failed => "Failed",
            }
        }

        fn is_failure_state(&self) -> bool {
            matches!(self, Self::Failed)
        }

        fn is_terminal(&self) -> bool {
            matches!(self, Self::Stopped | Self::Failed)
        }
    }

    #[test]
    fn test_component_state_traits() {
        let state = TestState::Running;
        assert_eq!(state.component_type(), "Test");
        assert!(!state.is_failure_state());
        assert!(state.is_operational());
        assert!(!state.is_terminal());

        let failed = TestState::Failed;
        assert!(failed.is_failure_state());
        assert!(!failed.is_operational());
        assert!(failed.is_terminal());
    }
}
