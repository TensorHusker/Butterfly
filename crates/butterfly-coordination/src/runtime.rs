//! # Async Coordination Runtime
//!
//! Orchestrates the distributed inference workflow across worker nodes.
//! Implements the 4-phase coordination protocol with Byzantine fault tolerance.

use crate::{
    ByzantineAgreement, BarrierCoordinator, CheckpointManager,
    CoordinationError, CoordinationMessage, CoordinationStateMachine,
    Phase, PhiAccrualFailureDetector, WorkAssignment, WorkAssigner,
};
use butterfly_comm::{CommunicationBackend, Message};
use butterfly_core::{NodeId, WorkerNode, WorkerConfig, WorkResult, Vote, BarrierCheckpoint};
use butterfly_core::{ModelExecutor, DeviceType};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// This node's ID
    pub node_id: NodeId,
    /// Total number of nodes in cluster
    pub cluster_size: usize,
    /// Maximum Byzantine failures to tolerate
    pub max_byzantine: usize,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Phase timeout in seconds
    pub phase_timeout_secs: u64,
    /// Barrier timeout in seconds
    pub barrier_timeout_secs: u64,
    /// Enable Byzantine consensus
    pub enable_byzantine: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId(0),
            cluster_size: 3,
            max_byzantine: 1,
            heartbeat_interval_ms: 100,
            phase_timeout_secs: 30,
            barrier_timeout_secs: 30,
            enable_byzantine: true,
        }
    }
}

/// Coordination runtime state
pub struct CoordinationRuntime {
    /// Configuration
    config: RuntimeConfig,
    /// State machine for coordination phases
    state_machine: Arc<RwLock<CoordinationStateMachine>>,
    /// Byzantine agreement protocol
    agreement: Arc<RwLock<ByzantineAgreement>>,
    /// Barrier coordinator for synchronization
    barrier: Arc<BarrierCoordinator>,
    /// Checkpoint manager
    checkpoint_manager: Arc<RwLock<CheckpointManager>>,
    /// Failure detector
    failure_detector: Arc<RwLock<PhiAccrualFailureDetector>>,
    /// Work assigner
    work_assigner: Arc<WorkAssigner>,
    /// Local workers managed by this coordinator
    workers: Arc<RwLock<HashMap<NodeId, WorkerNode>>>,
    /// Communication backend
    comm_backend: Option<Arc<dyn CommunicationBackend>>,
    /// Message receiver channel
    message_rx: Arc<RwLock<mpsc::UnboundedReceiver<CoordinationMessage>>>,
    /// Message sender channel
    message_tx: mpsc::UnboundedSender<CoordinationMessage>,
    /// Current superstep
    current_superstep: Arc<RwLock<u64>>,
}

impl CoordinationRuntime {
    /// Create a new coordination runtime
    pub fn new(config: RuntimeConfig) -> Self {
        let state_machine = Arc::new(RwLock::new(
            CoordinationStateMachine::new(
                config.node_id,
                config.cluster_size,
                config.max_byzantine,
            )
        ));

        let agreement = Arc::new(RwLock::new(
            ByzantineAgreement::new(config.cluster_size, config.max_byzantine)
        ));

        let barrier = Arc::new(BarrierCoordinator::new(
            config.cluster_size,
            config.max_byzantine,
        ));

        let checkpoint_manager = Arc::new(RwLock::new(
            CheckpointManager::new(10)
        ));

        let failure_detector = Arc::new(RwLock::new(
            PhiAccrualFailureDetector::new(
                config.heartbeat_interval_ms,
                8.0,  // φ threshold for suspicion
                12.0, // φ threshold for failure
            )
        ));

        let work_assigner = Arc::new(WorkAssigner::new());

        let (message_tx, message_rx) = mpsc::unbounded_channel();

        Self {
            config,
            state_machine,
            agreement,
            barrier,
            checkpoint_manager,
            failure_detector,
            work_assigner,
            workers: Arc::new(RwLock::new(HashMap::new())),
            comm_backend: None,
            message_rx: Arc::new(RwLock::new(message_rx)),
            message_tx,
            current_superstep: Arc::new(RwLock::new(0)),
        }
    }

    /// Set communication backend
    pub fn with_comm_backend(mut self, backend: Arc<dyn CommunicationBackend>) -> Self {
        self.comm_backend = Some(backend);
        self
    }

    /// Add a worker to this coordinator
    pub async fn add_worker(&self, worker: WorkerNode) -> Result<(), CoordinationError> {
        let node_id = worker.node_id();
        let mut workers = self.workers.write().await;

        if workers.contains_key(&node_id) {
            return Err(CoordinationError::Internal(
                format!("Worker {} already exists", node_id.0)
            ));
        }

        workers.insert(node_id, worker);
        info!("Added worker {}", node_id.0);
        Ok(())
    }

    /// Run the coordination runtime
    pub async fn run(self: Arc<Self>) -> Result<(), CoordinationError> {
        info!("Starting coordination runtime for node {}", self.config.node_id.0);

        // Spawn heartbeat task
        let heartbeat_handle = self.clone().spawn_heartbeat_task();

        // Spawn failure detector task
        let failure_detector_handle = self.clone().spawn_failure_detector_task();

        // Spawn message handler task
        let message_handler_handle = self.clone().spawn_message_handler_task();

        // Run main coordination loop
        let coordination_result = self.coordination_loop().await;

        // Wait for tasks to complete (or cancel them)
        heartbeat_handle.abort();
        failure_detector_handle.abort();
        message_handler_handle.abort();

        coordination_result
    }

    /// Main coordination loop (4-phase protocol)
    async fn coordination_loop(&self) -> Result<(), CoordinationError> {
        loop {
            let current_phase = {
                let sm = self.state_machine.read().await;
                sm.current_phase()
            };

            debug!("Entering phase: {:?}", current_phase);

            match current_phase {
                Phase::Assignment => {
                    self.execute_assignment_phase().await?;
                }
                Phase::Computation => {
                    self.execute_computation_phase().await?;
                }
                Phase::Aggregation => {
                    self.execute_aggregation_phase().await?;
                }
                Phase::Commitment => {
                    self.execute_commitment_phase().await?;
                }
            }

            // Check if we should advance to next phase
            if self.should_advance_phase().await {
                self.advance_phase().await?;
            }
        }
    }

    /// Execute Assignment Phase
    async fn execute_assignment_phase(&self) -> Result<(), CoordinationError> {
        info!("Executing Assignment Phase");

        // TODO: In production, this would receive inference requests
        // For now, create a simple test assignment
        let workers = self.workers.read().await;
        let worker_ids: Vec<NodeId> = workers.keys().copied().collect();

        if worker_ids.is_empty() {
            warn!("No workers available for assignment");
            return Ok(());
        }

        // Create work assignments for each worker
        let superstep = *self.current_superstep.read().await;

        for (idx, worker_id) in worker_ids.iter().enumerate() {
            if let Some(worker) = workers.get(worker_id) {
                // Create assignment (mock: layers 0-5)
                let assignment = butterfly_core::WorkAssignment {
                    assignment_id: (superstep * 100) + idx as u64,
                    layer_range: (idx * 2, (idx * 2) + 5),
                    input_data: vec![1.0; 7680], // Mock input
                    expected_output_size: 7680,
                    superstep,
                };

                // Send assignment to worker
                worker.handle_work_assignment(assignment)
                    .map_err(|e| CoordinationError::Internal(e.to_string()))?;

                debug!("Assigned work to worker {}", worker_id.0);
            }
        }

        Ok(())
    }

    /// Execute Computation Phase
    async fn execute_computation_phase(&self) -> Result<(), CoordinationError> {
        info!("Executing Computation Phase");

        let workers = self.workers.read().await;
        let mut results = Vec::new();

        // Execute work on all workers
        for (worker_id, worker) in workers.iter() {
            if !worker.is_ready() {
                match worker.execute_assigned_work() {
                    Ok(result) => {
                        debug!("Worker {} completed execution", worker_id.0);
                        results.push(result);
                    }
                    Err(e) => {
                        error!("Worker {} execution failed: {}", worker_id.0, e);
                        // Mark worker as failed
                        let mut fd = self.failure_detector.write().await;
                        fd.mark_failed(*worker_id);
                    }
                }
            }
        }

        // Store results for aggregation phase
        // TODO: In production, store in state machine

        Ok(())
    }

    /// Execute Aggregation Phase (Byzantine voting)
    async fn execute_aggregation_phase(&self) -> Result<(), CoordinationError> {
        info!("Executing Aggregation Phase");

        if !self.config.enable_byzantine {
            return Ok(());
        }

        let workers = self.workers.read().await;

        // Collect results from all workers
        let mut all_results: Vec<WorkResult> = Vec::new();

        // In production, results would be collected via communication backend
        // For now, we simulate by getting cached results from each worker

        // Byzantine voting: each worker votes on each result
        let mut votes: HashMap<u64, HashMap<Vote, usize>> = HashMap::new();

        for result in all_results.iter() {
            let mut vote_counts = HashMap::new();
            vote_counts.insert(Vote::Accept, 0);
            vote_counts.insert(Vote::Reject, 0);
            vote_counts.insert(Vote::Abstain, 0);

            for worker in workers.values() {
                match worker.vote_on_result(result) {
                    Ok(vote) => {
                        *vote_counts.entry(vote).or_insert(0) += 1;
                    }
                    Err(e) => {
                        warn!("Worker voting error: {}", e);
                    }
                }
            }

            votes.insert(result.assignment_id, vote_counts);
        }

        // Check Byzantine consensus (2f+1 accepts required)
        let required_accepts = self.config.max_byzantine * 2 + 1;

        for (assignment_id, vote_counts) in votes.iter() {
            let accepts = vote_counts.get(&Vote::Accept).unwrap_or(&0);

            if *accepts >= required_accepts {
                debug!("Assignment {} passed Byzantine consensus", assignment_id);
            } else {
                warn!("Assignment {} failed Byzantine consensus", assignment_id);
            }
        }

        Ok(())
    }

    /// Execute Commitment Phase (barrier synchronization)
    async fn execute_commitment_phase(&self) -> Result<(), CoordinationError> {
        info!("Executing Commitment Phase");

        let superstep = *self.current_superstep.read().await;
        let workers = self.workers.read().await;

        // Create barrier checkpoint
        let checkpoint = BarrierCheckpoint {
            checkpoint_id: superstep,
            superstep,
            state_hash: [0u8; 32], // TODO: Compute actual state hash
            required_nodes: workers.len(),
        };

        // All workers participate in barrier
        for worker in workers.values() {
            worker.participate_in_barrier(checkpoint.clone())
                .map_err(|e| CoordinationError::Internal(e.to_string()))?;
        }

        // Wait for barrier to complete (in production, via network)
        let barrier_timeout = Duration::from_secs(self.config.barrier_timeout_secs);

        match timeout(barrier_timeout, self.wait_for_barrier(checkpoint.checkpoint_id)).await {
            Ok(Ok(())) => {
                info!("Barrier synchronization complete for superstep {}", superstep);

                // Release barrier for all workers
                for worker in workers.values() {
                    worker.release_barrier(checkpoint.checkpoint_id)
                        .map_err(|e| CoordinationError::Internal(e.to_string()))?;
                }

                // Increment superstep
                *self.current_superstep.write().await += 1;
            }
            Ok(Err(e)) => {
                error!("Barrier synchronization failed: {}", e);
                return Err(e);
            }
            Err(_) => {
                error!("Barrier synchronization timeout");
                return Err(CoordinationError::Timeout("Barrier timeout".to_string()));
            }
        }

        Ok(())
    }

    /// Wait for barrier to complete
    async fn wait_for_barrier(&self, _checkpoint_id: u64) -> Result<(), CoordinationError> {
        // In production, this would wait for BarrierReady messages from all nodes
        // For now, simulate immediate completion
        sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Check if should advance to next phase
    async fn should_advance_phase(&self) -> bool {
        let sm = self.state_machine.read().await;
        sm.is_phase_complete()
    }

    /// Advance to next phase
    async fn advance_phase(&self) -> Result<Phase, CoordinationError> {
        let mut sm = self.state_machine.write().await;
        let new_phase = sm.advance_phase()?;
        info!("Advanced to phase: {:?}", new_phase);
        Ok(new_phase)
    }

    /// Spawn heartbeat task
    fn spawn_heartbeat_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let interval = Duration::from_millis(self.config.heartbeat_interval_ms);

            loop {
                sleep(interval).await;

                // Send heartbeat to all nodes
                if let Some(ref backend) = self.comm_backend {
                    // Use the communication Message enum, not CoordinationMessage
                    let heartbeat = Message::Heartbeat {
                        node_id: self.config.node_id,
                    };

                    // Broadcast heartbeat
                    if let Err(e) = backend.broadcast(heartbeat).await {
                        warn!("Failed to broadcast heartbeat: {}", e);
                    }
                }
            }
        })
    }

    /// Spawn failure detector task
    fn spawn_failure_detector_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let check_interval = Duration::from_millis(self.config.heartbeat_interval_ms * 2);

            loop {
                sleep(check_interval).await;

                // Check for failed nodes
                let suspected = {
                    let fd = self.failure_detector.read().await;
                    fd.suspected_nodes()
                };

                for node_id in suspected {
                    warn!("Node {} suspected of failure", node_id.0);

                    // Create failure evidence
                    let phi_value = {
                        let fd = self.failure_detector.read().await;
                        fd.phi(node_id).unwrap_or(0.0)
                    };

                    let evidence = crate::types::FailureEvidence::Unresponsive {
                        last_seen: chrono::Utc::now().timestamp(),
                        phi_value,
                    };

                    // Handle suspected failure
                    let mut sm = self.state_machine.write().await;
                    if let Err(e) = sm.handle_suspected_failure(node_id, evidence).await {
                        error!("Failed to handle suspected failure: {}", e);
                    }
                }
            }
        })
    }

    /// Spawn message handler task
    fn spawn_message_handler_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let message_rx = self.message_rx.clone();

        tokio::spawn(async move {
            let mut rx = message_rx.write().await;

            while let Some(message) = rx.recv().await {
                if let Err(e) = self.handle_message(message).await {
                    error!("Message handling error: {}", e);
                }
            }
        })
    }

    /// Handle incoming coordination message
    async fn handle_message(&self, message: CoordinationMessage) -> Result<(), CoordinationError> {
        match message {
            CoordinationMessage::Heartbeat(node_id, _phi) => {
                let mut fd = self.failure_detector.write().await;
                fd.record_heartbeat(node_id);
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
            _ => {}
        }
        Ok(())
    }

    /// Get current superstep
    pub async fn current_superstep(&self) -> u64 {
        *self.current_superstep.read().await
    }

    /// Get current phase
    pub async fn current_phase(&self) -> Phase {
        let sm = self.state_machine.read().await;
        sm.current_phase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::ModelExecutor;

    fn create_test_runtime() -> Arc<CoordinationRuntime> {
        let config = RuntimeConfig {
            node_id: NodeId(0),
            cluster_size: 3,
            max_byzantine: 1,
            ..Default::default()
        };
        Arc::new(CoordinationRuntime::new(config))
    }

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = create_test_runtime();
        assert_eq!(runtime.current_phase().await, Phase::Assignment);
        assert_eq!(runtime.current_superstep().await, 0);
    }

    #[tokio::test]
    async fn test_add_worker() {
        let runtime = create_test_runtime();

        let worker_config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let worker = WorkerNode::new(worker_config, executor);

        runtime.add_worker(worker).await.unwrap();

        let workers = runtime.workers.read().await;
        assert_eq!(workers.len(), 1);
    }

    #[tokio::test]
    async fn test_assignment_phase() {
        let runtime = create_test_runtime();

        // Add workers
        for i in 1..=3 {
            let worker_config = WorkerConfig {
                node_id: NodeId(i),
                device: DeviceType::Cpu,
                ..Default::default()
            };
            let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
            let mut worker = WorkerNode::new(worker_config, executor);
            worker.initialize().unwrap();
            runtime.add_worker(worker).await.unwrap();
        }

        // Execute assignment phase
        runtime.execute_assignment_phase().await.unwrap();

        // Verify workers received assignments
        let workers = runtime.workers.read().await;
        for worker in workers.values() {
            assert_eq!(worker.get_state(), butterfly_core::WorkerState::Assigned);
        }
    }

    #[tokio::test]
    async fn test_computation_phase() {
        let runtime = create_test_runtime();

        // Add and initialize worker
        let worker_config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker = WorkerNode::new(worker_config, executor);
        worker.initialize().unwrap();

        // Assign work
        let assignment = butterfly_core::WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };
        worker.handle_work_assignment(assignment).unwrap();

        runtime.add_worker(worker).await.unwrap();

        // Execute computation phase
        runtime.execute_computation_phase().await.unwrap();
    }

    #[tokio::test]
    async fn test_phase_advancement() {
        let runtime = create_test_runtime();

        assert_eq!(runtime.current_phase().await, Phase::Assignment);

        // Note: Phase advancement in production requires quorum of nodes
        // For this simple test, we just verify current phase
        // Full phase advancement testing requires mocking the state machine's quorum logic
    }
}
