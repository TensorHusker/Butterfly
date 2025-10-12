//! System-level state definitions shared across all components

use serde::{Deserialize, Serialize};
use std::fmt;

/// The overall state of the Butterfly distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SystemState {
    /// System exists but not initialized
    Uninitialized,
    /// Discovering nodes and establishing cluster
    Bootstrapping,
    /// Operational and waiting for requests
    Ready,
    /// Executing distributed inference
    Computing,
    /// Operating with reduced capacity due to failures
    Degraded,
    /// Committing inference result via Byzantine agreement
    Committing,
    /// Gracefully shutting down
    Shutdown,
    /// Fully terminated
    Terminated,
}

impl SystemState {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            Self::Uninitialized => "System not initialized",
            Self::Bootstrapping => "Establishing cluster",
            Self::Ready => "Ready for inference requests",
            Self::Computing => "Executing distributed computation",
            Self::Degraded => "Operating with failures",
            Self::Committing => "Finalizing results",
            Self::Shutdown => "Shutting down",
            Self::Terminated => "Terminated",
        }
    }

    /// Check if state is operational (can process requests)
    pub fn is_operational(&self) -> bool {
        matches!(
            self,
            Self::Ready | Self::Computing | Self::Degraded | Self::Committing
        )
    }

    /// Check if state is terminal (no more transitions)
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// Check if state allows accepting new work
    pub fn accepts_new_work(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

impl fmt::Display for SystemState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Execution phase within a computation cycle
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord,
)]
pub enum Phase {
    /// Work assignment distribution
    Assignment = 0,
    /// Pipelined computation execution
    Computation = 1,
    /// Result aggregation at barriers
    Aggregation = 2,
    /// Byzantine agreement and commitment
    Commitment = 3,
}

impl Phase {
    /// Get next phase in sequence
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Assignment => Some(Self::Computation),
            Self::Computation => Some(Self::Aggregation),
            Self::Aggregation => Some(Self::Commitment),
            Self::Commitment => None, // Cycles back to Assignment with new epoch
        }
    }

    /// Get phase index for ordering
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Calculate distance between phases
    pub fn distance_to(&self, other: &Phase) -> usize {
        (other.index() as isize - self.index() as isize).unsigned_abs()
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Unique identifier for state transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionId(pub u64);

/// Trigger that caused a state transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionTrigger {
    /// Explicit command from operator
    Command(String),
    /// Automatic transition due to system event
    Event(String),
    /// Triggered by another component's state change
    Cascade {
        source: String,
        reason: String,
    },
    /// Timeout or deadline reached
    Timeout {
        duration_ms: u64,
    },
    /// Failure detected
    Failure {
        node: String,
        evidence: String,
    },
    /// Recovery completed
    Recovery {
        recovered_nodes: Vec<String>,
    },
}

impl fmt::Display for TransitionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command(cmd) => write!(f, "Command: {}", cmd),
            Self::Event(evt) => write!(f, "Event: {}", evt),
            Self::Cascade { source, reason } => {
                write!(f, "Cascade from {}: {}", source, reason)
            }
            Self::Timeout { duration_ms } => write!(f, "Timeout after {}ms", duration_ms),
            Self::Failure { node, evidence } => write!(f, "Failure of {}: {}", node, evidence),
            Self::Recovery { recovered_nodes } => {
                write!(f, "Recovery of {:?}", recovered_nodes)
            }
        }
    }
}

/// Record of a state transition with full context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionRecord<S> {
    /// Unique transition identifier
    pub id: TransitionId,
    /// State before transition
    pub from: S,
    /// State after transition
    pub to: S,
    /// What triggered the transition
    pub trigger: TransitionTrigger,
    /// Monotonic timestamp (nanoseconds since system boot)
    pub timestamp: i64,
    /// Wall clock time for human readability
    pub wall_time: chrono::DateTime<chrono::Utc>,
    /// Epoch at time of transition
    pub epoch: u64,
    /// How long the transition took
    pub duration_us: u64,
}

impl<S: fmt::Debug> fmt::Display for StateTransitionRecord<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {:?} → {:?} ({}) [{}µs]",
            self.wall_time.format("%H:%M:%S%.3f"),
            self.from,
            self.to,
            self.trigger,
            self.duration_us
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_state_operational() {
        assert!(SystemState::Ready.is_operational());
        assert!(SystemState::Computing.is_operational());
        assert!(!SystemState::Uninitialized.is_operational());
        assert!(!SystemState::Terminated.is_operational());
    }

    #[test]
    fn test_phase_progression() {
        assert_eq!(Phase::Assignment.next(), Some(Phase::Computation));
        assert_eq!(Phase::Computation.next(), Some(Phase::Aggregation));
        assert_eq!(Phase::Aggregation.next(), Some(Phase::Commitment));
        assert_eq!(Phase::Commitment.next(), None);
    }

    #[test]
    fn test_phase_distance() {
        assert_eq!(Phase::Assignment.distance_to(&Phase::Commitment), 3);
        assert_eq!(Phase::Computation.distance_to(&Phase::Aggregation), 1);
        assert_eq!(Phase::Aggregation.distance_to(&Phase::Computation), 1);
    }
}
