//! Checkpoint management for recovery

use crate::types::{CheckpointData, CheckpointMetadata, CoordinationError, Epoch};
use butterfly_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info};

/// Maximum number of checkpoints to retain
const DEFAULT_MAX_CHECKPOINTS: usize = 10;

/// Checkpoint containing system state for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub epoch: Epoch,
    pub token_position: usize,
    pub node_states: HashMap<NodeId, Vec<u8>>,
    pub intermediate_results: HashMap<NodeId, Vec<u8>>,
    pub metadata: CheckpointMetadata,
}

impl Checkpoint {
    /// Create new checkpoint
    pub fn new(
        epoch: Epoch,
        token_position: usize,
        node_states: HashMap<NodeId, Vec<u8>>,
        intermediate_results: HashMap<NodeId, Vec<u8>>,
    ) -> Self {
        use sha2::{Digest, Sha256};

        // Compute checksum of all data
        let mut hasher = Sha256::new();
        hasher.update(&epoch.to_le_bytes());
        hasher.update(&token_position.to_le_bytes());

        for (node_id, state) in &node_states {
            hasher.update(&node_id.0.to_le_bytes());
            hasher.update(state);
        }

        for (node_id, result) in &intermediate_results {
            hasher.update(&node_id.0.to_le_bytes());
            hasher.update(result);
        }

        let checksum: [u8; 32] = hasher.finalize().into();

        let metadata = CheckpointMetadata {
            timestamp: chrono::Utc::now().timestamp(),
            version: 1,
            checksum,
        };

        Self {
            epoch,
            token_position,
            node_states,
            intermediate_results,
            metadata,
        }
    }

    /// Verify checkpoint integrity
    pub fn verify(&self) -> bool {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(&self.token_position.to_le_bytes());

        for (node_id, state) in &self.node_states {
            hasher.update(&node_id.0.to_le_bytes());
            hasher.update(state);
        }

        for (node_id, result) in &self.intermediate_results {
            hasher.update(&node_id.0.to_le_bytes());
            hasher.update(result);
        }

        let computed: [u8; 32] = hasher.finalize().into();

        computed == self.metadata.checksum
    }

    /// Get size in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let states_size: usize = self.node_states.values().map(|v| v.len()).sum();
        let results_size: usize = self.intermediate_results.values().map(|v| v.len()).sum();
        states_size + results_size + std::mem::size_of::<Self>()
    }

    /// Convert to CheckpointData for transmission
    pub fn to_data(&self) -> CheckpointData {
        CheckpointData {
            epoch: self.epoch,
            token_position: self.token_position,
            node_states: self.node_states.clone(),
            intermediate_results: self.intermediate_results.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Create from CheckpointData
    pub fn from_data(data: CheckpointData) -> Self {
        Self {
            epoch: data.epoch,
            token_position: data.token_position,
            node_states: data.node_states,
            intermediate_results: data.intermediate_results,
            metadata: data.metadata,
        }
    }
}

/// Manages checkpoint storage and retrieval
pub struct CheckpointManager {
    /// Maximum checkpoints to retain
    max_checkpoints: usize,
    /// Stored checkpoints (FIFO queue)
    checkpoints: VecDeque<Checkpoint>,
    /// Index by epoch for fast lookup
    epoch_index: HashMap<Epoch, usize>,
    /// Checkpoint frequency (every K tokens)
    checkpoint_frequency: usize,
    /// Total checkpoints created
    total_created: u64,
    /// Total bytes stored
    total_bytes: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            max_checkpoints: max_checkpoints.max(1),
            checkpoints: VecDeque::new(),
            epoch_index: HashMap::new(),
            checkpoint_frequency: 10, // Default: every 10 tokens
            total_created: 0,
            total_bytes: 0,
        }
    }

    /// Store checkpoint
    pub fn store(&mut self, checkpoint: Checkpoint) -> Result<(), CoordinationError> {
        // Verify checkpoint integrity
        if !checkpoint.verify() {
            return Err(CoordinationError::InvalidProof(
                "Checkpoint verification failed".to_string(),
            ));
        }

        let epoch = checkpoint.epoch;
        let size = checkpoint.size_bytes();

        info!(
            epoch,
            token_position = checkpoint.token_position,
            size_kb = size / 1024,
            "Storing checkpoint"
        );

        // Remove oldest if at capacity
        if self.checkpoints.len() >= self.max_checkpoints {
            if let Some(old) = self.checkpoints.pop_front() {
                self.epoch_index.remove(&old.epoch);
                self.total_bytes -= old.size_bytes();
                debug!(epoch = old.epoch, "Evicted old checkpoint");
            }
        }

        // Add new checkpoint
        let index = self.checkpoints.len();
        self.checkpoints.push_back(checkpoint);
        self.epoch_index.insert(epoch, index);
        self.total_created += 1;
        self.total_bytes += size;

        Ok(())
    }

    /// Retrieve checkpoint by epoch
    pub fn get(&self, epoch: Epoch) -> Result<&Checkpoint, CoordinationError> {
        // Check if epoch is in index
        let index = self
            .epoch_index
            .get(&epoch)
            .ok_or(CoordinationError::CheckpointNotFound(epoch))?;

        // Adjust index due to FIFO eviction
        let offset = self.checkpoints.len() - self.epoch_index.len();
        let actual_index = index.saturating_sub(offset);

        self.checkpoints
            .get(actual_index)
            .ok_or(CoordinationError::CheckpointNotFound(epoch))
    }

    /// Get latest checkpoint
    pub fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.back()
    }

    /// Get checkpoint closest to token position
    pub fn get_at_position(&self, token_position: usize) -> Option<&Checkpoint> {
        // Find checkpoint with largest token_position <= target
        self.checkpoints
            .iter()
            .rev()
            .find(|cp| cp.token_position <= token_position)
    }

    /// List all stored epochs
    pub fn epochs(&self) -> Vec<Epoch> {
        let mut epochs: Vec<_> = self.epoch_index.keys().copied().collect();
        epochs.sort();
        epochs
    }

    /// Get number of stored checkpoints
    pub fn count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Get total storage used
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Set checkpoint frequency
    pub fn set_frequency(&mut self, frequency: usize) {
        self.checkpoint_frequency = frequency.max(1);
        debug!(frequency, "Set checkpoint frequency");
    }

    /// Check if should create checkpoint at token position
    pub fn should_checkpoint(&self, token_position: usize) -> bool {
        token_position % self.checkpoint_frequency == 0
    }

    /// Clear all checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.epoch_index.clear();
        self.total_bytes = 0;
        info!("Cleared all checkpoints");
    }

    /// Get statistics
    pub fn stats(&self) -> CheckpointStats {
        CheckpointStats {
            count: self.count(),
            total_created: self.total_created,
            total_bytes: self.total_bytes,
            avg_size_bytes: if self.count() > 0 {
                self.total_bytes / self.count()
            } else {
                0
            },
            oldest_epoch: self.checkpoints.front().map(|cp| cp.epoch),
            latest_epoch: self.checkpoints.back().map(|cp| cp.epoch),
        }
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CHECKPOINTS)
    }
}

/// Statistics about checkpoint storage
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub count: usize,
    pub total_created: u64,
    pub total_bytes: usize,
    pub avg_size_bytes: usize,
    pub oldest_epoch: Option<Epoch>,
    pub latest_epoch: Option<Epoch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_checkpoint(epoch: Epoch, token_position: usize) -> Checkpoint {
        let mut node_states = HashMap::new();
        node_states.insert(NodeId(0), vec![1, 2, 3]);

        let mut intermediate_results = HashMap::new();
        intermediate_results.insert(NodeId(0), vec![4, 5, 6]);

        Checkpoint::new(epoch, token_position, node_states, intermediate_results)
    }

    #[test]
    fn test_checkpoint_verification() {
        let cp = create_test_checkpoint(0, 0);
        assert!(cp.verify());
    }

    #[test]
    fn test_checkpoint_manager_storage() {
        let mut manager = CheckpointManager::new(5);

        for i in 0..3 {
            let cp = create_test_checkpoint(i, i as usize * 10);
            manager.store(cp).unwrap();
        }

        assert_eq!(manager.count(), 3);
        assert!(manager.get(1).is_ok());
    }

    #[test]
    fn test_checkpoint_eviction() {
        let mut manager = CheckpointManager::new(3);

        // Store 5 checkpoints (2 should be evicted)
        for i in 0..5 {
            let cp = create_test_checkpoint(i, i as usize * 10);
            manager.store(cp).unwrap();
        }

        assert_eq!(manager.count(), 3);
        assert!(manager.get(0).is_err()); // Evicted
        assert!(manager.get(4).is_ok()); // Latest
    }

    #[test]
    fn test_get_at_position() {
        let mut manager = CheckpointManager::new(5);

        manager.store(create_test_checkpoint(0, 0)).unwrap();
        manager.store(create_test_checkpoint(1, 10)).unwrap();
        manager.store(create_test_checkpoint(2, 20)).unwrap();

        let cp = manager.get_at_position(15).unwrap();
        assert_eq!(cp.token_position, 10);

        let cp = manager.get_at_position(25).unwrap();
        assert_eq!(cp.token_position, 20);
    }

    #[test]
    fn test_checkpoint_frequency() {
        let mut manager = CheckpointManager::new(10);
        manager.set_frequency(5);

        assert!(manager.should_checkpoint(0));
        assert!(manager.should_checkpoint(5));
        assert!(manager.should_checkpoint(10));
        assert!(!manager.should_checkpoint(7));
    }

    #[test]
    fn test_checkpoint_stats() {
        let mut manager = CheckpointManager::new(5);

        for i in 0..3 {
            manager.store(create_test_checkpoint(i, i as usize * 10)).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_created, 3);
        assert_eq!(stats.oldest_epoch, Some(0));
        assert_eq!(stats.latest_epoch, Some(2));
    }
}
