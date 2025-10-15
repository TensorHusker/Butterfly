//! # Butterfly Coordination
//!
//! Byzantine-fault-tolerant coordination protocol for distributed transformer inference.
//!
//! This crate implements the Butterfly Coordination Protocol (BCP), which provides:
//! - Multi-phase execution coordination (assignment, computation, aggregation, commitment)
//! - Byzantine fault tolerance (up to f failures in 2f+1 nodes)
//! - Minimal synchronization via pipelined execution
//! - Checkpoint-based recovery
//! - Adaptive failure detection
//!
//! See `docs/coordination_protocol.md` for detailed specification.

pub mod agreement;
pub mod barrier;
pub mod checkpoint;
pub mod failure_detector;
pub mod state_machine;
pub mod types;
pub mod work_assignment;

pub use agreement::ByzantineAgreement;
pub use barrier::BarrierCoordinator;
pub use checkpoint::{Checkpoint, CheckpointManager};
pub use failure_detector::PhiAccrualFailureDetector;
pub use state_machine::{CoordinationStateMachine, NodeState, Phase};
pub use types::{CoordinationMessage, CoordinationError, Proof, WorkAssignment};
pub use work_assignment::WorkAssigner;

use butterfly_core::NodeId;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main coordinator that orchestrates distributed inference
pub struct DistributedCoordinator {
    state_machine: Arc<RwLock<CoordinationStateMachine>>,
    agreement: Arc<RwLock<ByzantineAgreement>>,
    barrier: Arc<BarrierCoordinator>,
    checkpoint_manager: Arc<RwLock<CheckpointManager>>,
    failure_detector: Arc<RwLock<PhiAccrualFailureDetector>>,
    work_assigner: Arc<WorkAssigner>,
}

impl DistributedCoordinator {
    pub fn new(
        node_id: NodeId,
        cluster_size: usize,
        max_byzantine: usize,
    ) -> Self {
        let state_machine = Arc::new(RwLock::new(
            CoordinationStateMachine::new(node_id, cluster_size, max_byzantine)
        ));

        let agreement = Arc::new(RwLock::new(
            ByzantineAgreement::new(cluster_size, max_byzantine)
        ));

        let barrier = Arc::new(BarrierCoordinator::new(
            cluster_size,
            max_byzantine,
        ));

        let checkpoint_manager = Arc::new(RwLock::new(
            CheckpointManager::new(10) // Keep last 10 checkpoints
        ));

        let failure_detector = Arc::new(RwLock::new(
            PhiAccrualFailureDetector::new(
                100, // 100ms base heartbeat interval
                8.0, // φ threshold for suspicion
                12.0, // φ threshold for failure
            )
        ));

        let work_assigner = Arc::new(WorkAssigner::new());

        Self {
            state_machine,
            agreement,
            barrier,
            checkpoint_manager,
            failure_detector,
            work_assigner,
        }
    }

    /// Handle incoming coordination message
    pub async fn handle_message(
        &self,
        message: CoordinationMessage,
    ) -> Result<(), CoordinationError> {
        match message {
            CoordinationMessage::WorkAssignment(assignment) => {
                let mut sm = self.state_machine.write().await;
                sm.apply_work_assignment(assignment)?;
            }
            CoordinationMessage::BarrierReady(node_id, hash) => {
                self.barrier.node_ready(node_id, hash).await?;
            }
            CoordinationMessage::PrePrepare(result, proof) => {
                let mut agreement = self.agreement.write().await;
                agreement.handle_pre_prepare(result, proof).await?;
            }
            CoordinationMessage::Prepare(result_hash) => {
                let mut agreement = self.agreement.write().await;
                agreement.handle_prepare(result_hash).await?;
            }
            CoordinationMessage::Commit(result) => {
                let mut agreement = self.agreement.write().await;
                agreement.handle_commit(result).await?;
            }
            CoordinationMessage::Heartbeat(node_id, _phi) => {
                let mut fd = self.failure_detector.write().await;
                fd.record_heartbeat(node_id);
            }
            CoordinationMessage::Suspicion(node_id, evidence) => {
                let mut sm = self.state_machine.write().await;
                sm.handle_suspected_failure(node_id, evidence).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Assign work for a new inference request
    pub async fn assign_work(
        &self,
        layers: &[butterfly_core::LayerInfo],
        nodes: &[NodeId],
    ) -> Result<WorkAssignment, CoordinationError> {
        self.work_assigner
            .assign(layers, nodes)
            .await
            .map_err(|e| CoordinationError::Internal(e))
    }

    /// Check if system is ready for next phase
    pub async fn is_phase_complete(&self) -> bool {
        let sm = self.state_machine.read().await;
        sm.is_phase_complete()
    }

    /// Advance to next execution phase
    pub async fn advance_phase(&self) -> Result<Phase, CoordinationError> {
        let mut sm = self.state_machine.write().await;
        sm.advance_phase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_initialization() {
        let coordinator = DistributedCoordinator::new(
            NodeId(0),
            5, // 5 nodes
            2, // tolerate 2 Byzantine
        );

        let sm = coordinator.state_machine.read().await;
        assert_eq!(sm.current_phase(), Phase::Assignment);
    }
}
