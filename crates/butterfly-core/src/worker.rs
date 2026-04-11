//! # Worker Execution Engine
//!
//! Implements the worker node that executes assigned computation tasks
//! and participates in Byzantine fault-tolerant coordination.

use crate::executor::{ModelExecutor, ExecutorError, DeviceType};
use crate::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use thiserror::Error;

/// Worker errors
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("Worker not ready: {0}")]
    NotReady(String),

    #[error("Invalid work assignment: {0}")]
    InvalidAssignment(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Byzantine consensus failed: {0}")]
    ConsensusFailed(String),

    #[error("Barrier timeout: {0}")]
    BarrierTimeout(String),

    #[error("Executor error: {0}")]
    Executor(#[from] ExecutorError),

    #[error("State transition error: {0}")]
    StateTransition(String),
}

/// Worker lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerState {
    /// Initial state - worker is starting up
    Initializing,
    /// Worker is idle and ready to accept work
    Idle,
    /// Worker has received a work assignment
    Assigned,
    /// Worker is actively computing
    Computing,
    /// Worker has completed computation and is voting
    Voting,
    /// Worker is waiting at a barrier
    BarrierWait,
    /// Worker has failed and needs recovery
    Failed,
}

/// Work assignment from coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    /// Unique assignment ID
    pub assignment_id: u64,
    /// Layer range to execute (start, end)
    pub layer_range: (usize, usize),
    /// Input tensor data
    pub input_data: Vec<f32>,
    /// Expected output size for validation
    pub expected_output_size: usize,
    /// Superstep/round number
    pub superstep: u64,
}

/// Result of work execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    /// Assignment this result corresponds to
    pub assignment_id: u64,
    /// Output tensor data
    pub output_data: Vec<f32>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Hash of output for Byzantine verification
    pub output_hash: [u8; 32],
    /// Node that computed this result
    pub node_id: NodeId,
}

/// Byzantine vote on a result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vote {
    /// Result is correct
    Accept,
    /// Result is incorrect or suspicious
    Reject,
    /// Cannot determine (abstain)
    Abstain,
}

/// Barrier checkpoint for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierCheckpoint {
    /// Checkpoint ID
    pub checkpoint_id: u64,
    /// Superstep this checkpoint is for
    pub superstep: u64,
    /// Hash of computation state
    pub state_hash: [u8; 32],
    /// Number of nodes that must reach this barrier
    pub required_nodes: usize,
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// This worker's node ID
    pub node_id: NodeId,
    /// Device to use for execution
    pub device: DeviceType,
    /// Maximum work queue size
    pub max_queue_size: usize,
    /// Timeout for barrier synchronization (seconds)
    pub barrier_timeout_secs: u64,
    /// Enable Byzantine voting
    pub enable_byzantine: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId(0),
            device: DeviceType::Cpu,
            max_queue_size: 100,
            barrier_timeout_secs: 30,
            enable_byzantine: true,
        }
    }
}

/// Worker node that executes computation tasks
pub struct WorkerNode {
    /// Configuration
    config: WorkerConfig,
    /// Model executor
    executor: Arc<ModelExecutor>,
    /// Current state
    state: Arc<RwLock<WorkerState>>,
    /// Current work assignment (if any)
    current_assignment: Arc<RwLock<Option<WorkAssignment>>>,
    /// Completed results cache
    results_cache: Arc<RwLock<HashMap<u64, WorkResult>>>,
    /// Barrier checkpoints we've reached
    barrier_checkpoints: Arc<RwLock<HashMap<u64, BarrierCheckpoint>>>,
}

impl WorkerNode {
    /// Create a new worker node
    pub fn new(config: WorkerConfig, executor: ModelExecutor) -> Self {
        Self {
            config,
            executor: Arc::new(executor),
            state: Arc::new(RwLock::new(WorkerState::Initializing)),
            current_assignment: Arc::new(RwLock::new(None)),
            results_cache: Arc::new(RwLock::new(HashMap::new())),
            barrier_checkpoints: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the worker
    pub fn initialize(&self) -> Result<(), WorkerError> {
        let mut state = self.state.write();
        if *state != WorkerState::Initializing {
            return Err(WorkerError::StateTransition(
                format!("Cannot initialize from state {:?}", *state)
            ));
        }
        *state = WorkerState::Idle;
        Ok(())
    }

    /// Get current worker state
    pub fn get_state(&self) -> WorkerState {
        *self.state.read()
    }

    /// Check if worker is ready to accept work
    pub fn is_ready(&self) -> bool {
        *self.state.read() == WorkerState::Idle
    }

    /// Handle incoming work assignment
    pub fn handle_work_assignment(&self, assignment: WorkAssignment) -> Result<(), WorkerError> {
        // Verify we're in the right state
        let mut state = self.state.write();
        if *state != WorkerState::Idle {
            return Err(WorkerError::NotReady(
                format!("Worker in state {:?}, cannot accept work", *state)
            ));
        }

        // Validate assignment
        if assignment.layer_range.0 >= assignment.layer_range.1 {
            return Err(WorkerError::InvalidAssignment(
                format!("Invalid layer range: {:?}", assignment.layer_range)
            ));
        }

        if assignment.layer_range.1 > self.executor.num_layers() {
            return Err(WorkerError::InvalidAssignment(
                format!("Layer range {:?} exceeds model layers {}",
                    assignment.layer_range, self.executor.num_layers())
            ));
        }

        // Store assignment and transition to Assigned state
        *self.current_assignment.write() = Some(assignment);
        *state = WorkerState::Assigned;
        Ok(())
    }

    /// Execute the assigned work
    pub fn execute_assigned_work(&self) -> Result<WorkResult, WorkerError> {
        // Transition to Computing state
        {
            let mut state = self.state.write();
            if *state != WorkerState::Assigned {
                return Err(WorkerError::StateTransition(
                    format!("Cannot execute from state {:?}", *state)
                ));
            }
            *state = WorkerState::Computing;
        }

        // Get the assignment
        let assignment = {
            let assignment_lock = self.current_assignment.read();
            assignment_lock.as_ref()
                .ok_or_else(|| WorkerError::NotReady("No work assignment".to_string()))?
                .clone()
        };

        // Execute the computation
        let start_time = std::time::Instant::now();
        let output_data = self.executor.execute_layers(
            &assignment.input_data,
            assignment.layer_range.0..assignment.layer_range.1,
        )?;
        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Validate output size
        if output_data.len() != assignment.expected_output_size {
            return Err(WorkerError::ExecutionFailed(
                format!("Output size mismatch: expected {}, got {}",
                    assignment.expected_output_size, output_data.len())
            ));
        }

        // Compute hash for Byzantine verification
        let output_hash = Self::compute_hash(&output_data);

        // Create result
        let result = WorkResult {
            assignment_id: assignment.assignment_id,
            output_data,
            execution_time_ms,
            output_hash,
            node_id: self.config.node_id,
        };

        // Cache the result
        self.results_cache.write().insert(assignment.assignment_id, result.clone());

        // Transition to Voting state
        *self.state.write() = WorkerState::Voting;

        Ok(result)
    }

    /// Vote on another worker's result (Byzantine consensus)
    pub fn vote_on_result(&self, result: &WorkResult) -> Result<Vote, WorkerError> {
        if !self.config.enable_byzantine {
            return Ok(Vote::Accept);
        }

        // Check if we have our own result for this assignment
        let our_result = {
            let cache = self.results_cache.read();
            cache.get(&result.assignment_id).cloned()
        };

        let vote = match our_result {
            Some(our_result) => {
                // Compare hashes
                if our_result.output_hash == result.output_hash {
                    Vote::Accept
                } else {
                    // Hashes don't match - Byzantine fault detected
                    Vote::Reject
                }
            }
            None => {
                // We don't have a result for this assignment, abstain
                Vote::Abstain
            }
        };

        Ok(vote)
    }

    /// Participate in barrier synchronization
    pub fn participate_in_barrier(&self, checkpoint: BarrierCheckpoint) -> Result<(), WorkerError> {
        // Transition to BarrierWait state
        {
            let mut state = self.state.write();
            match *state {
                WorkerState::Voting | WorkerState::Idle => {
                    *state = WorkerState::BarrierWait;
                }
                _ => {
                    return Err(WorkerError::StateTransition(
                        format!("Cannot enter barrier from state {:?}", *state)
                    ));
                }
            }
        }

        // Store checkpoint
        self.barrier_checkpoints.write().insert(checkpoint.checkpoint_id, checkpoint);

        Ok(())
    }

    /// Release barrier and return to idle state
    pub fn release_barrier(&self, checkpoint_id: u64) -> Result<(), WorkerError> {
        {
            let mut state = self.state.write();
            if *state != WorkerState::BarrierWait {
                return Err(WorkerError::StateTransition(
                    format!("Not at barrier, current state: {:?}", *state)
                ));
            }
            *state = WorkerState::Idle;
        }

        // Clear the assignment
        *self.current_assignment.write() = None;

        // Remove checkpoint
        self.barrier_checkpoints.write().remove(&checkpoint_id);

        Ok(())
    }

    /// Mark worker as failed
    pub fn mark_failed(&self, reason: String) -> Result<(), WorkerError> {
        *self.state.write() = WorkerState::Failed;
        Err(WorkerError::NotReady(reason))
    }

    /// Recover from failed state
    pub fn recover(&self) -> Result<(), WorkerError> {
        let mut state = self.state.write();
        if *state != WorkerState::Failed {
            return Err(WorkerError::StateTransition(
                format!("Cannot recover from state {:?}", *state)
            ));
        }

        // Clear state
        *self.current_assignment.write() = None;
        *state = WorkerState::Idle;

        Ok(())
    }

    /// Get node ID
    pub fn node_id(&self) -> NodeId {
        self.config.node_id
    }

    /// Compute SHA-256 hash of data
    fn compute_hash(data: &[f32]) -> [u8; 32] {
        // Simple hash implementation (in production, use SHA-256)
        // For now, use a deterministic but simple hash
        let mut hash = [0u8; 32];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };

        for (i, chunk) in bytes.chunks(32).enumerate() {
            for (j, &byte) in chunk.iter().enumerate() {
                hash[j] = hash[j].wrapping_add(byte).wrapping_add(i as u8);
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_worker() -> WorkerNode {
        let config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        WorkerNode::new(config, executor)
    }

    #[test]
    fn test_worker_initialization() {
        let worker = create_test_worker();
        assert_eq!(worker.get_state(), WorkerState::Initializing);

        worker.initialize().unwrap();
        assert_eq!(worker.get_state(), WorkerState::Idle);
        assert!(worker.is_ready());
    }

    #[test]
    fn test_work_assignment() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let assignment = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680], // 1 * 10 * 768
            expected_output_size: 7680,
            superstep: 0,
        };

        worker.handle_work_assignment(assignment).unwrap();
        assert_eq!(worker.get_state(), WorkerState::Assigned);
    }

    #[test]
    fn test_invalid_assignment_when_busy() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let assignment1 = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };

        worker.handle_work_assignment(assignment1.clone()).unwrap();

        // Try to assign more work while busy
        let result = worker.handle_work_assignment(assignment1);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_assigned_work() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let assignment = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };

        worker.handle_work_assignment(assignment).unwrap();
        let result = worker.execute_assigned_work().unwrap();

        assert_eq!(result.assignment_id, 1);
        assert_eq!(result.output_data.len(), 7680);
        assert_eq!(result.node_id, NodeId(1));
        assert_eq!(worker.get_state(), WorkerState::Voting);
    }

    #[test]
    fn test_byzantine_voting_accept() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let assignment = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };

        worker.handle_work_assignment(assignment).unwrap();
        let our_result = worker.execute_assigned_work().unwrap();

        // Vote on our own result (should accept)
        let vote = worker.vote_on_result(&our_result).unwrap();
        assert_eq!(vote, Vote::Accept);
    }

    #[test]
    fn test_byzantine_voting_reject() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let assignment = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };

        worker.handle_work_assignment(assignment).unwrap();
        worker.execute_assigned_work().unwrap();

        // Create a different result with different hash
        let fake_result = WorkResult {
            assignment_id: 1,
            output_data: vec![2.0; 7680], // Different data
            execution_time_ms: 10.0,
            output_hash: [255u8; 32], // Different hash
            node_id: NodeId(2),
        };

        // Vote on fake result (should reject)
        let vote = worker.vote_on_result(&fake_result).unwrap();
        assert_eq!(vote, Vote::Reject);
    }

    #[test]
    fn test_barrier_participation() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let checkpoint = BarrierCheckpoint {
            checkpoint_id: 1,
            superstep: 0,
            state_hash: [0u8; 32],
            required_nodes: 3,
        };

        worker.participate_in_barrier(checkpoint).unwrap();
        assert_eq!(worker.get_state(), WorkerState::BarrierWait);

        worker.release_barrier(1).unwrap();
        assert_eq!(worker.get_state(), WorkerState::Idle);
    }

    #[test]
    fn test_failure_and_recovery() {
        let worker = create_test_worker();
        worker.initialize().unwrap();

        let result = worker.mark_failed("Test failure".to_string());
        assert!(result.is_err());
        assert_eq!(worker.get_state(), WorkerState::Failed);

        worker.recover().unwrap();
        assert_eq!(worker.get_state(), WorkerState::Idle);
    }

    #[test]
    fn test_state_machine_transitions() {
        let worker = create_test_worker();

        // Initializing -> Idle
        assert_eq!(worker.get_state(), WorkerState::Initializing);
        worker.initialize().unwrap();
        assert_eq!(worker.get_state(), WorkerState::Idle);

        // Idle -> Assigned
        let assignment = WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };
        worker.handle_work_assignment(assignment).unwrap();
        assert_eq!(worker.get_state(), WorkerState::Assigned);

        // Assigned -> Computing -> Voting
        worker.execute_assigned_work().unwrap();
        assert_eq!(worker.get_state(), WorkerState::Voting);

        // Voting -> BarrierWait
        let checkpoint = BarrierCheckpoint {
            checkpoint_id: 1,
            superstep: 0,
            state_hash: [0u8; 32],
            required_nodes: 3,
        };
        worker.participate_in_barrier(checkpoint).unwrap();
        assert_eq!(worker.get_state(), WorkerState::BarrierWait);

        // BarrierWait -> Idle
        worker.release_barrier(1).unwrap();
        assert_eq!(worker.get_state(), WorkerState::Idle);
    }
}
