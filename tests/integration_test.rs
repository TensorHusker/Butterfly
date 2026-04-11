//! Integration tests for Butterfly distributed inference system
//!
//! These tests verify the end-to-end functionality of the system,
//! including node communication, task distribution, and result aggregation.

use butterfly_core::{NodeId, WorkerNode, WorkerConfig, DeviceType, ModelExecutor};
use butterfly_coordination::{CoordinationRuntime, RuntimeConfig};
use std::sync::Arc;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_single_node_inference() {
        // Create a worker node
        let worker_config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };

        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker = WorkerNode::new(worker_config, executor);

        // Initialize the worker
        worker.initialize().unwrap();
        assert!(worker.is_ready(), "Worker should be ready after initialization");

        // Create a work assignment
        let assignment = butterfly_core::WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680], // 1 * 10 * 768
            expected_output_size: 7680,
            superstep: 0,
        };

        // Assign and execute work
        worker.handle_work_assignment(assignment).unwrap();
        let result = worker.execute_assigned_work().unwrap();

        // Verify the result
        assert_eq!(result.assignment_id, 1);
        assert_eq!(result.output_data.len(), 7680);
        assert_eq!(result.node_id, NodeId(1));
    }

    #[tokio::test]
    async fn test_multi_node_coordination() {
        // Create coordination runtime
        let config = RuntimeConfig {
            node_id: NodeId(0),
            cluster_size: 3,
            max_byzantine: 1,
            heartbeat_interval_ms: 100,
            phase_timeout_secs: 30,
            barrier_timeout_secs: 30,
            enable_byzantine: true,
        };

        let runtime = Arc::new(CoordinationRuntime::new(config));

        // Add multiple workers
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

        // Verify workers are added
        let workers = runtime.workers.read().await;
        assert_eq!(workers.len(), 3, "Should have 3 workers");
    }

    #[tokio::test]
    async fn test_assignment_phase() {
        let config = RuntimeConfig {
            node_id: NodeId(0),
            cluster_size: 3,
            max_byzantine: 1,
            ..Default::default()
        };

        let runtime = Arc::new(CoordinationRuntime::new(config));

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
    async fn test_byzantine_voting() {
        // Create two workers
        let worker_config1 = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            enable_byzantine: true,
            ..Default::default()
        };
        let executor1 = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker1 = WorkerNode::new(worker_config1, executor1);
        worker1.initialize().unwrap();

        let worker_config2 = WorkerConfig {
            node_id: NodeId(2),
            device: DeviceType::Cpu,
            enable_byzantine: true,
            ..Default::default()
        };
        let executor2 = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker2 = WorkerNode::new(worker_config2, executor2);
        worker2.initialize().unwrap();

        // Both execute the same work
        let assignment = butterfly_core::WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };

        worker1.handle_work_assignment(assignment.clone()).unwrap();
        worker2.handle_work_assignment(assignment).unwrap();

        let result1 = worker1.execute_assigned_work().unwrap();
        let result2 = worker2.execute_assigned_work().unwrap();

        // Each worker votes on the other's result
        let vote1_on_2 = worker1.vote_on_result(&result2).unwrap();
        let vote2_on_1 = worker2.vote_on_result(&result1).unwrap();

        // Both should accept since they computed the same result
        assert_eq!(vote1_on_2, butterfly_core::Vote::Accept);
        assert_eq!(vote2_on_1, butterfly_core::Vote::Accept);
    }

    #[tokio::test]
    async fn test_fault_tolerance() {
        // Test that a worker can be marked as failed and recovered
        let worker_config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker = WorkerNode::new(worker_config, executor);
        worker.initialize().unwrap();

        // Mark as failed
        let _ = worker.mark_failed("Test failure".to_string());
        assert_eq!(worker.get_state(), butterfly_core::WorkerState::Failed);

        // Recover
        worker.recover().unwrap();
        assert_eq!(worker.get_state(), butterfly_core::WorkerState::Idle);
        assert!(worker.is_ready());
    }

    #[tokio::test]
    async fn test_barrier_synchronization() {
        let worker_config = WorkerConfig {
            node_id: NodeId(1),
            device: DeviceType::Cpu,
            ..Default::default()
        };
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let mut worker = WorkerNode::new(worker_config, executor);
        worker.initialize().unwrap();

        // Complete some work first
        let assignment = butterfly_core::WorkAssignment {
            assignment_id: 1,
            layer_range: (0, 5),
            input_data: vec![1.0; 7680],
            expected_output_size: 7680,
            superstep: 0,
        };
        worker.handle_work_assignment(assignment).unwrap();
        worker.execute_assigned_work().unwrap();

        // Create barrier checkpoint
        let checkpoint = butterfly_core::BarrierCheckpoint {
            checkpoint_id: 1,
            superstep: 0,
            state_hash: [0u8; 32],
            required_nodes: 3,
        };

        // Participate in barrier
        worker.participate_in_barrier(checkpoint.clone()).unwrap();
        assert_eq!(worker.get_state(), butterfly_core::WorkerState::BarrierWait);

        // Release barrier
        worker.release_barrier(checkpoint.checkpoint_id).unwrap();
        assert_eq!(worker.get_state(), butterfly_core::WorkerState::Idle);
    }

    #[tokio::test]
    async fn test_full_computation_phase() {
        let config = RuntimeConfig {
            node_id: NodeId(0),
            cluster_size: 2,
            max_byzantine: 0,
            ..Default::default()
        };

        let runtime = Arc::new(CoordinationRuntime::new(config));

        // Add workers
        for i in 1..=2 {
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

        // Execute computation phase
        runtime.execute_computation_phase().await.unwrap();

        // Workers should have completed execution
        // (State checking would require exposing more internals or using message passing)
    }
}

#[cfg(test)]
mod communication_tests {
    use butterfly_comm::{TcpBackend, CommunicationBackend, Message};
    use butterfly_core::NodeId;

    #[tokio::test]
    async fn test_tcp_backend_creation() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let backend = TcpBackend::new(NodeId(1), addr).await.unwrap();
        assert!(backend.bind_addr.port() > 0);
    }

    #[tokio::test]
    async fn test_peer_registration() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let backend = TcpBackend::new(NodeId(1), addr).await.unwrap();

        let peer_addr = "127.0.0.1:9000".parse().unwrap();
        backend.register_peer(NodeId(2), peer_addr);

        assert!(backend.peer_addrs.contains_key(&NodeId(2)));
    }
}

#[cfg(test)]
mod config_tests {
    use butterfly_config::Config;
    use std::path::Path;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_load_config_from_file() {
        let config_path = Path::new("config/default.toml");
        if config_path.exists() {
            let config = Config::load_from_file(config_path).unwrap();
            assert!(config.validate().is_ok());
            assert_eq!(config.byzantine.fault_tolerance_level, 1);
        }
    }
}
