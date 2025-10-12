//! State invariant checking and validation

use super::{ComponentState, StateError};
use std::fmt;

/// Severity of an invariant violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational - not a problem
    Info,
    /// Warning - should investigate
    Warning,
    /// High - likely problem, manual intervention may be needed
    High,
    /// Critical - immediate action required
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::High => write!(f, "HIGH"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Types of invariant violations
#[derive(Debug, Clone)]
pub enum InvariantViolation {
    /// Multiple coordinators detected
    MultipleCoordinators {
        count: usize,
    },
    /// Quorum lost
    QuorumLost {
        operational: usize,
        required: usize,
    },
    /// Epoch skew between nodes
    EpochSkew {
        min: u64,
        max: u64,
    },
    /// Work assignment doesn't cover all layers
    IncompleteAssignment {
        assigned: usize,
        total: usize,
    },
    /// Same layer assigned to multiple nodes
    OverlappingAssignment {
        layer_id: usize,
    },
    /// Vertical coherence violated (component state invalid for system state)
    VerticalIncoherence {
        system_state: String,
        component_type: String,
        component_id: String,
        component_state: String,
    },
    /// Custom invariant violation
    Custom {
        name: String,
        description: String,
    },
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultipleCoordinators { count } => {
                write!(f, "Multiple coordinators detected: {}", count)
            }
            Self::QuorumLost {
                operational,
                required,
            } => {
                write!(
                    f,
                    "Quorum lost: {} operational nodes, {} required",
                    operational, required
                )
            }
            Self::EpochSkew { min, max } => {
                write!(f, "Epoch skew: {} to {}", min, max)
            }
            Self::IncompleteAssignment { assigned, total } => {
                write!(
                    f,
                    "Incomplete assignment: {} of {} layers assigned",
                    assigned, total
                )
            }
            Self::OverlappingAssignment { layer_id } => {
                write!(f, "Layer {} assigned to multiple nodes", layer_id)
            }
            Self::VerticalIncoherence {
                system_state,
                component_type,
                component_id,
                component_state,
            } => {
                write!(
                    f,
                    "Vertical incoherence: {} {} in state {} while system in {}",
                    component_type, component_id, component_state, system_state
                )
            }
            Self::Custom { name, description } => {
                write!(f, "{}: {}", name, description)
            }
        }
    }
}

/// Trait for state invariants that must hold
pub trait StateInvariant: Send + Sync {
    /// Check if invariant holds
    fn check(&self) -> Result<(), InvariantViolation>;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Description of what this invariant ensures
    fn description(&self) -> &str;

    /// Severity if violated
    fn severity(&self) -> Severity;
}

/// Helper struct to create inline invariants
pub struct Invariant<F>
where
    F: Fn() -> Result<(), InvariantViolation> + Send + Sync,
{
    name: &'static str,
    description: &'static str,
    severity: Severity,
    check_fn: F,
}

impl<F> Invariant<F>
where
    F: Fn() -> Result<(), InvariantViolation> + Send + Sync,
{
    pub fn new(
        name: &'static str,
        description: &'static str,
        severity: Severity,
        check_fn: F,
    ) -> Self {
        Self {
            name,
            description,
            severity,
            check_fn,
        }
    }
}

impl<F> StateInvariant for Invariant<F>
where
    F: Fn() -> Result<(), InvariantViolation> + Send + Sync,
{
    fn check(&self) -> Result<(), InvariantViolation> {
        (self.check_fn)()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn severity(&self) -> Severity {
        self.severity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_invariant() {
        let invariant = Invariant::new(
            "TestInvariant",
            "Test invariant that always passes",
            Severity::Info,
            || Ok(()),
        );

        assert_eq!(invariant.name(), "TestInvariant");
        assert_eq!(invariant.severity(), Severity::Info);
        assert!(invariant.check().is_ok());
    }

    #[test]
    fn test_failing_invariant() {
        let invariant = Invariant::new(
            "AlwaysFails",
            "Test invariant that always fails",
            Severity::Critical,
            || {
                Err(InvariantViolation::Custom {
                    name: "AlwaysFails".to_string(),
                    description: "This invariant is designed to fail".to_string(),
                })
            },
        );

        assert!(invariant.check().is_err());
    }
}
