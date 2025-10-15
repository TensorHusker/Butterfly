//! Coordination state machine implementation

use crate::types::{CoordinationError, Epoch, FailureEvidence, WorkAssignment};
use butterfly_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// States a node can be in during coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is initializing and loading model
    Initializing,
    /// Node is ready to receive work
    Ready,
    /// Node is computing assigned layers
    Computing,
    /// Node is waiting for dependencies at aggregation barrier
    Aggregating,
    /// Node is committing result after Byzantine agreement
    Committing,
    /// Node is in degraded mode due to peer failure
    Degraded,
    /// Node is recovering from failure
    Recovering,
    /// Node has failed
    Failed,
}

/// Phases of distributed inference execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Phase 1: Work assignment and distribution
    Assignment,
    /// Phase 2: Pipelined computation
    Computation,
    /// Phase 3: Result aggregation
    Aggregation,
    /// Phase 4: Byzantine agreement and commitment
    Commitment,
}

/// Main coordination state machine
#[derive(Debug)]
pub struct CoordinationStateMachine {
    /// This node's ID
    node_id: NodeId,
    /// Current state of this node
    state: NodeState,
    /// Current execution epoch
    epoch: Epoch,
    /// Current execution phase
    phase: Phase,
    /// Total number of nodes in cluster
    cluster_size: usize,
    /// Maximum number of Byzantine failures tolerated
    max_byzantine: usize,
    /// Quorum size (2f + 1)
    quorum_size: usize,
    /// States of all nodes in cluster
    node_states: HashMap<NodeId, NodeState>,
    /// Current work assignment
    work_assignment: Option<WorkAssignment>,
    /// Nodes that completed current phase
    phase_completed: HashSet<NodeId>,
    /// Suspected failed nodes
    suspected_failures: HashSet<NodeId>,
    /// Confirmed failed nodes
    confirmed_failures: HashSet<NodeId>,
}

impl CoordinationStateMachine {
    /// Create new state machine
    pub fn new(node_id: NodeId, cluster_size: usize, max_byzantine: usize) -> Self {
        let quorum_size = 2 * max_byzantine + 1;

        if cluster_size < quorum_size {
            panic!(
                "Cluster size {} too small for {} Byzantine failures (need >= {})",
                cluster_size, max_byzantine, quorum_size
            );
        }

        Self {
            node_id,
            state: NodeState::Initializing,
            epoch: 0,
            phase: Phase::Assignment,
            cluster_size,
            max_byzantine,
            quorum_size,
            node_states: HashMap::new(),
            work_assignment: None,
            phase_completed: HashSet::new(),
            suspected_failures: HashSet::new(),
            confirmed_failures: HashSet::new(),
        }
    }

    /// Get current node state
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Get current phase
    pub fn current_phase(&self) -> Phase {
        self.phase
    }

    /// Get current epoch
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Transition to ready state
    pub fn transition_to_ready(&mut self) -> Result<(), CoordinationError> {
        match self.state {
            NodeState::Initializing | NodeState::Committing => {
                info!(node_id = ?self.node_id, "Transitioning to Ready state");
                self.state = NodeState::Ready;
                Ok(())
            }
            _ => Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.state),
                to: "Ready".to_string(),
            }),
        }
    }

    /// Apply work assignment and transition to computing
    pub fn apply_work_assignment(
        &mut self,
        assignment: WorkAssignment,
    ) -> Result<(), CoordinationError> {
        if self.phase != Phase::Assignment {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.phase),
                to: "Computing".to_string(),
            });
        }

        if self.state != NodeState::Ready {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.state),
                to: "Computing".to_string(),
            });
        }

        info!(
            node_id = ?self.node_id,
            epoch = assignment.epoch,
            "Applying work assignment"
        );

        self.work_assignment = Some(assignment);
        self.state = NodeState::Computing;
        self.phase = Phase::Computation;
        self.phase_completed.clear();

        Ok(())
    }

    /// Mark local computation complete, transition to aggregating
    pub fn complete_computation(&mut self) -> Result<(), CoordinationError> {
        if self.state != NodeState::Computing {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.state),
                to: "Aggregating".to_string(),
            });
        }

        debug!(node_id = ?self.node_id, "Computation complete, entering Aggregating");
        self.state = NodeState::Aggregating;
        Ok(())
    }

    /// Record that a node completed the current phase
    pub fn mark_node_phase_complete(&mut self, node_id: NodeId) {
        debug!(?node_id, "Node completed phase");
        self.phase_completed.insert(node_id);
    }

    /// Check if enough nodes completed phase to proceed
    pub fn is_phase_complete(&self) -> bool {
        let completed_count = self.phase_completed.len();
        completed_count >= self.quorum_size
    }

    /// Advance to next phase
    pub fn advance_phase(&mut self) -> Result<Phase, CoordinationError> {
        if !self.is_phase_complete() {
            return Err(CoordinationError::QuorumNotReached {
                current: self.phase_completed.len(),
                required: self.quorum_size,
            });
        }

        let next_phase = match self.phase {
            Phase::Assignment => Phase::Computation,
            Phase::Computation => Phase::Aggregation,
            Phase::Aggregation => Phase::Commitment,
            Phase::Commitment => {
                // Cycle back to assignment for next inference
                self.epoch += 1;
                info!(epoch = self.epoch, "Starting new epoch");
                Phase::Assignment
            }
        };

        info!(
            from = ?self.phase,
            to = ?next_phase,
            "Advancing phase"
        );

        self.phase = next_phase;
        self.phase_completed.clear();

        Ok(next_phase)
    }

    /// Transition to committing state
    pub fn transition_to_committing(&mut self) -> Result<(), CoordinationError> {
        if self.state != NodeState::Aggregating {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.state),
                to: "Committing".to_string(),
            });
        }

        if self.phase != Phase::Commitment {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.phase),
                to: "Committing".to_string(),
            });
        }

        debug!(node_id = ?self.node_id, "Transitioning to Committing");
        self.state = NodeState::Committing;
        Ok(())
    }

    /// Handle suspected node failure
    pub async fn handle_suspected_failure(
        &mut self,
        node_id: NodeId,
        evidence: FailureEvidence,
    ) -> Result<(), CoordinationError> {
        warn!(?node_id, ?evidence, "Node failure suspected");

        self.suspected_failures.insert(node_id);

        // If enough nodes suspect this node, confirm the failure
        if self.suspected_failures.len() >= self.quorum_size {
            self.confirm_failure(node_id).await?;
        }

        // Transition to degraded mode if we're not already handling recovery
        if self.state != NodeState::Recovering && self.state != NodeState::Failed {
            self.state = NodeState::Degraded;
        }

        Ok(())
    }

    /// Confirm node failure
    async fn confirm_failure(&mut self, node_id: NodeId) -> Result<(), CoordinationError> {
        warn!(?node_id, "Node failure confirmed");

        self.confirmed_failures.insert(node_id);
        self.node_states.insert(node_id, NodeState::Failed);

        // Check if we still have quorum
        let operational_count = self.cluster_size - self.confirmed_failures.len();
        if operational_count < self.quorum_size {
            warn!(
                operational = operational_count,
                required = self.quorum_size,
                "Lost quorum, system cannot proceed"
            );
            return Err(CoordinationError::QuorumNotReached {
                current: operational_count,
                required: self.quorum_size,
            });
        }

        Ok(())
    }

    /// Initiate recovery from failure
    pub fn start_recovery(&mut self) -> Result<(), CoordinationError> {
        info!(node_id = ?self.node_id, "Starting recovery");
        self.state = NodeState::Recovering;
        Ok(())
    }

    /// Complete recovery and return to ready state
    pub fn complete_recovery(&mut self) -> Result<(), CoordinationError> {
        if self.state != NodeState::Recovering {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.state),
                to: "Ready".to_string(),
            });
        }

        info!(node_id = ?self.node_id, "Recovery complete");
        self.state = NodeState::Ready;
        self.suspected_failures.clear();

        Ok(())
    }

    /// Get operational node count
    pub fn operational_node_count(&self) -> usize {
        self.cluster_size - self.confirmed_failures.len()
    }

    /// Check if node is operational
    pub fn is_node_operational(&self, node_id: NodeId) -> bool {
        !self.confirmed_failures.contains(&node_id)
    }

    /// Get current work assignment
    pub fn work_assignment(&self) -> Option<&WorkAssignment> {
        self.work_assignment.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_initialization() {
        let sm = CoordinationStateMachine::new(NodeId(0), 5, 2);
        assert_eq!(sm.state(), NodeState::Initializing);
        assert_eq!(sm.current_phase(), Phase::Assignment);
        assert_eq!(sm.epoch(), 0);
    }

    #[test]
    fn test_phase_advancement() {
        let mut sm = CoordinationStateMachine::new(NodeId(0), 5, 2);

        // Mark quorum as complete
        for i in 0..5 {
            sm.mark_node_phase_complete(NodeId(i));
        }

        assert!(sm.is_phase_complete());

        let next = sm.advance_phase().unwrap();
        assert_eq!(next, Phase::Computation);
    }

    #[test]
    fn test_quorum_calculation() {
        let sm = CoordinationStateMachine::new(NodeId(0), 7, 2);
        assert_eq!(sm.quorum_size, 5); // 2 * 2 + 1
    }

    #[test]
    fn test_state_transitions() {
        let mut sm = CoordinationStateMachine::new(NodeId(0), 5, 2);

        sm.transition_to_ready().unwrap();
        assert_eq!(sm.state(), NodeState::Ready);

        // Can't transition to computing without work assignment
        assert!(sm.complete_computation().is_err());
    }

    #[tokio::test]
    async fn test_failure_handling() {
        let mut sm = CoordinationStateMachine::new(NodeId(0), 5, 2);

        let evidence = FailureEvidence::Unresponsive {
            last_seen: 1000,
            phi_value: 10.0,
        };

        sm.handle_suspected_failure(NodeId(1), evidence).await.unwrap();

        assert!(sm.suspected_failures.contains(&NodeId(1)));
        assert_eq!(sm.state(), NodeState::Degraded);
    }
}
