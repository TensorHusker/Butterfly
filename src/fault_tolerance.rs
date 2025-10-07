use crate::node::{NodeId, NodeRegistry};
use crate::communication::{CommunicationLayer, Message, MessageType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time;

/// Status of a node in the distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is healthy and responsive
    Healthy,
    /// Node is experiencing degraded performance
    Degraded,
    /// Node is not responding to health checks
    Unresponsive,
    /// Node has failed
    Failed,
}

/// Health information for a node
#[derive(Debug, Clone)]
pub struct HealthInfo {
    pub node_id: NodeId,
    pub status: NodeStatus,
    pub last_heartbeat: Instant,
    pub consecutive_failures: u32,
}

impl HealthInfo {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            status: NodeStatus::Healthy,
            last_heartbeat: Instant::now(),
            consecutive_failures: 0,
        }
    }

    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
        self.consecutive_failures = 0;
        self.status = NodeStatus::Healthy;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    pub fn is_healthy(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() < timeout && self.status != NodeStatus::Failed
    }
}

/// Monitors the health of nodes in the distributed system
pub struct HealthMonitor {
    /// Health information for each node
    health_info: HashMap<NodeId, HealthInfo>,
    /// Timeout duration for health checks
    health_check_timeout: Duration,
    /// Interval between health checks
    health_check_interval: Duration,
    /// Maximum consecutive failures before marking node as failed
    max_consecutive_failures: u32,
}

impl HealthMonitor {
    pub fn new(
        health_check_timeout: Duration,
        health_check_interval: Duration,
        max_consecutive_failures: u32,
    ) -> Self {
        Self {
            health_info: HashMap::new(),
            health_check_timeout,
            health_check_interval,
            max_consecutive_failures,
        }
    }

    /// Register a node for health monitoring
    pub fn register_node(&mut self, node_id: NodeId) {
        self.health_info.insert(node_id, HealthInfo::new(node_id));
    }

    /// Update heartbeat for a node
    pub fn update_heartbeat(&mut self, node_id: &NodeId) {
        if let Some(info) = self.health_info.get_mut(node_id) {
            info.update_heartbeat();
        }
    }

    /// Check the health of all nodes
    pub fn check_all_nodes(&mut self) -> Vec<NodeId> {
        let mut failed_nodes = Vec::new();

        for (node_id, info) in self.health_info.iter_mut() {
            if !info.is_healthy(self.health_check_timeout) {
                info.record_failure();
                
                if info.consecutive_failures >= self.max_consecutive_failures {
                    info.status = NodeStatus::Failed;
                    failed_nodes.push(*node_id);
                } else {
                    info.status = NodeStatus::Degraded;
                }
            }
        }

        failed_nodes
    }

    /// Get the status of a specific node
    pub fn get_node_status(&self, node_id: &NodeId) -> Option<NodeStatus> {
        self.health_info.get(node_id).map(|info| info.status)
    }

    /// Get all healthy nodes
    pub fn healthy_nodes(&self) -> Vec<NodeId> {
        self.health_info
            .iter()
            .filter(|(_, info)| info.status == NodeStatus::Healthy)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Remove a failed node from monitoring
    pub fn remove_node(&mut self, node_id: &NodeId) {
        self.health_info.remove(node_id);
    }

    /// Start periodic health checking
    pub async fn start_monitoring(
        mut self,
        mut comm_layer: CommunicationLayer,
    ) -> Result<(), FaultToleranceError> {
        let mut interval = time::interval(self.health_check_interval);

        loop {
            interval.tick().await;

            // Check all nodes
            let failed_nodes = self.check_all_nodes();

            // Send health check messages to all nodes
            for (node_id, info) in &self.health_info {
                if info.status != NodeStatus::Failed {
                    let message = Message::new(
                        comm_layer.node_id(),
                        *node_id,
                        MessageType::HealthCheck,
                        vec![],
                    );
                    
                    if let Err(_) = comm_layer.send_message(message).await {
                        // Failed to send health check
                    }
                }
            }

            // Handle failed nodes
            for failed_node in failed_nodes {
                tracing::warn!("Node {:?} has failed", failed_node);
                // In a full implementation, trigger recovery procedures
            }
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new(
            Duration::from_secs(30),
            Duration::from_secs(10),
            3,
        )
    }
}

/// Manages fault tolerance and recovery
pub struct FaultToleranceManager {
    health_monitor: HealthMonitor,
    node_registry: NodeRegistry,
    /// Backup nodes for recovery
    backup_nodes: Vec<NodeId>,
}

impl FaultToleranceManager {
    pub fn new(health_monitor: HealthMonitor, node_registry: NodeRegistry) -> Self {
        Self {
            health_monitor,
            node_registry,
            backup_nodes: Vec::new(),
        }
    }

    /// Handle a node failure by redistributing its workload
    pub fn handle_node_failure(&mut self, failed_node: &NodeId) -> Result<(), FaultToleranceError> {
        // Get the failed node's assigned layers
        let failed_node_obj = self.node_registry.get_node(failed_node)
            .ok_or(FaultToleranceError::NodeNotFound)?;
        
        let assigned_layers = failed_node_obj.assigned_layers.clone();

        // Find healthy nodes to redistribute work
        let healthy_nodes = self.health_monitor.healthy_nodes();
        
        if healthy_nodes.is_empty() {
            return Err(FaultToleranceError::NoHealthyNodes);
        }

        // Redistribute layers to healthy nodes
        for (idx, layer_id) in assigned_layers.iter().enumerate() {
            let target_node_id = &healthy_nodes[idx % healthy_nodes.len()];
            
            if let Some(target_node) = self.node_registry.get_node_mut(target_node_id) {
                target_node.assign_layer(*layer_id);
            }
        }

        // Remove failed node from registry
        self.node_registry.remove_node(failed_node);
        self.health_monitor.remove_node(failed_node);

        Ok(())
    }

    /// Add a backup node for recovery
    pub fn add_backup_node(&mut self, node_id: NodeId) {
        if !self.backup_nodes.contains(&node_id) {
            self.backup_nodes.push(node_id);
        }
    }

    /// Get available backup nodes
    pub fn get_backup_nodes(&self) -> &[NodeId] {
        &self.backup_nodes
    }

    /// Activate a backup node to replace a failed node
    pub fn activate_backup_node(&mut self) -> Option<NodeId> {
        self.backup_nodes.pop()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FaultToleranceError {
    #[error("Node not found")]
    NodeNotFound,
    #[error("No healthy nodes available")]
    NoHealthyNodes,
    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_health_info() {
        let node_id = NodeId::new();
        let mut info = HealthInfo::new(node_id);
        
        assert_eq!(info.status, NodeStatus::Healthy);
        assert_eq!(info.consecutive_failures, 0);
        
        info.record_failure();
        assert_eq!(info.consecutive_failures, 1);
        
        info.update_heartbeat();
        assert_eq!(info.consecutive_failures, 0);
    }

    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new(
            Duration::from_secs(5),
            Duration::from_secs(1),
            3,
        );
        
        let node_id = NodeId::new();
        monitor.register_node(node_id);
        
        assert_eq!(monitor.get_node_status(&node_id), Some(NodeStatus::Healthy));
        
        monitor.update_heartbeat(&node_id);
        assert_eq!(monitor.get_node_status(&node_id), Some(NodeStatus::Healthy));
    }

    #[test]
    fn test_fault_tolerance_manager() {
        let health_monitor = HealthMonitor::default();
        let node_registry = NodeRegistry::new();
        
        let manager = FaultToleranceManager::new(health_monitor, node_registry);
        
        assert_eq!(manager.backup_nodes.len(), 0);
    }
}
