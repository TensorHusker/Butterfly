//! Barrier synchronization for phase coordination

use crate::types::{CheckpointHash, CoordinationError, Epoch};
use butterfly_core::NodeId;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{debug, info};

/// Barrier coordinator for synchronizing nodes at phase boundaries
pub struct BarrierCoordinator {
    /// Cluster size
    cluster_size: usize,
    /// Quorum size
    quorum_size: usize,
    /// Current epoch
    epoch: Arc<RwLock<Epoch>>,
    /// Nodes that reached the barrier
    ready_nodes: Arc<RwLock<HashSet<NodeId>>>,
    /// Checkpoint hashes from ready nodes
    checkpoint_hashes: Arc<RwLock<HashMap<NodeId, CheckpointHash>>>,
    /// Notify when barrier is released
    release_notify: Arc<Notify>,
    /// Whether barrier is currently released
    released: Arc<RwLock<bool>>,
}

impl BarrierCoordinator {
    /// Create new barrier coordinator
    pub fn new(cluster_size: usize, max_byzantine: usize) -> Self {
        let quorum_size = 2 * max_byzantine + 1;

        Self {
            cluster_size,
            quorum_size,
            epoch: Arc::new(RwLock::new(0)),
            ready_nodes: Arc::new(RwLock::new(HashSet::new())),
            checkpoint_hashes: Arc::new(RwLock::new(HashMap::new())),
            release_notify: Arc::new(Notify::new()),
            released: Arc::new(RwLock::new(false)),
        }
    }

    /// Node signals ready at barrier
    pub async fn node_ready(
        &self,
        node_id: NodeId,
        checkpoint_hash: CheckpointHash,
    ) -> Result<(), CoordinationError> {
        {
            let mut ready = self.ready_nodes.write();
            ready.insert(node_id);

            let mut hashes = self.checkpoint_hashes.write();
            hashes.insert(node_id, checkpoint_hash);

            debug!(
                ?node_id,
                ready_count = ready.len(),
                required = self.quorum_size,
                "Node ready at barrier"
            );
        }

        // Check if we can release the barrier
        self.try_release().await?;

        Ok(())
    }

    /// Try to release barrier if quorum reached
    async fn try_release(&self) -> Result<(), CoordinationError> {
        let (ready_count, hashes) = {
            let ready = self.ready_nodes.read();
            let hashes = self.checkpoint_hashes.read();

            if ready.len() < self.quorum_size {
                return Ok(()); // Not enough nodes yet
            }

            (
                ready.len(),
                hashes.values().cloned().collect::<Vec<_>>(),
            )
        };

        // Verify all checkpoint hashes match (honest nodes agree)
        let unique_hashes: HashSet<_> = hashes.iter().collect();

        if unique_hashes.len() > 1 {
            // Byzantine behavior: nodes have different checkpoints
            return Err(CoordinationError::ByzantineDetected(
                format!(
                    "Checkpoint hash mismatch: {} different hashes from {} nodes",
                    unique_hashes.len(),
                    ready_count
                )
            ));
        }

        // Release the barrier
        {
            let mut released = self.released.write();
            if !*released {
                info!(ready_count, "Barrier released, advancing phase");
                *released = true;
                self.release_notify.notify_waiters();
            }
        }

        Ok(())
    }

    /// Wait for barrier to be released
    pub async fn wait(&self) {
        loop {
            {
                let released = self.released.read();
                if *released {
                    return;
                }
            }

            // Wait for notification
            self.release_notify.notified().await;
        }
    }

    /// Reset barrier for next phase
    pub fn reset(&self) {
        let mut ready = self.ready_nodes.write();
        ready.clear();

        let mut hashes = self.checkpoint_hashes.write();
        hashes.clear();

        let mut released = self.released.write();
        *released = false;

        debug!("Barrier reset for next phase");
    }

    /// Check if barrier is released
    pub fn is_released(&self) -> bool {
        *self.released.read()
    }

    /// Get number of ready nodes
    pub fn ready_count(&self) -> usize {
        self.ready_nodes.read().len()
    }

    /// Advance to next epoch
    pub fn advance_epoch(&self) {
        let mut epoch = self.epoch.write();
        *epoch += 1;
        info!(epoch = *epoch, "Advanced to next epoch");
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> Epoch {
        *self.epoch.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_barrier_synchronization() {
        let barrier = Arc::new(BarrierCoordinator::new(5, 2));

        // Spawn tasks to simulate nodes
        let mut handles = vec![];

        for i in 0..5 {
            let barrier_clone = barrier.clone();
            let handle = tokio::spawn(async move {
                let node_id = NodeId(i);
                let hash = [i as u8; 32];

                // Signal ready
                barrier_clone.node_ready(node_id, hash).await.unwrap();

                // Wait for barrier
                barrier_clone.wait().await;
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        assert!(barrier.is_released());
        assert_eq!(barrier.ready_count(), 5);
    }

    #[tokio::test]
    async fn test_quorum_requirement() {
        let barrier = BarrierCoordinator::new(5, 2); // Need 5 nodes for quorum

        // Only 4 nodes ready (below quorum)
        for i in 0..4 {
            barrier.node_ready(NodeId(i), [0u8; 32]).await.unwrap();
        }

        assert!(!barrier.is_released());

        // 5th node makes quorum
        barrier.node_ready(NodeId(4), [0u8; 32]).await.unwrap();

        // Small delay for async processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(barrier.is_released());
    }

    #[tokio::test]
    async fn test_byzantine_detection() {
        let barrier = BarrierCoordinator::new(5, 2);

        // First 3 nodes with matching hash
        for i in 0..3 {
            barrier.node_ready(NodeId(i), [0u8; 32]).await.unwrap();
        }

        // 4th and 5th nodes with different hashes (Byzantine)
        barrier.node_ready(NodeId(3), [1u8; 32]).await.unwrap();
        let result = barrier.node_ready(NodeId(4), [2u8; 32]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_barrier_reset() {
        let barrier = BarrierCoordinator::new(3, 1);

        // First round
        for i in 0..3 {
            barrier.node_ready(NodeId(i), [0u8; 32]).await.unwrap();
        }

        assert!(barrier.is_released());

        // Reset
        barrier.reset();

        assert!(!barrier.is_released());
        assert_eq!(barrier.ready_count(), 0);
    }

    #[tokio::test]
    async fn test_wait_timeout() {
        let barrier = BarrierCoordinator::new(5, 2);

        // Only 2 nodes ready (below quorum)
        barrier.node_ready(NodeId(0), [0u8; 32]).await.unwrap();
        barrier.node_ready(NodeId(1), [0u8; 32]).await.unwrap();

        // Wait should timeout since quorum not reached
        let result = timeout(Duration::from_millis(100), barrier.wait()).await;

        assert!(result.is_err()); // Timeout
    }
}
