use butterfly::*;
use std::time::Duration;

#[test]
fn test_end_to_end_node_setup() {
    let mut registry = NodeRegistry::new();

    let capability = NodeCapability {
        memory_gb: 32.0,
        compute_flops: 20e12,
        network_bandwidth_gbps: 25.0,
        num_devices: 2,
        device_type: "GPU-A100".to_string(),
    };

    let node = Node::new("node1:8080".to_string(), capability);
    let node_id = registry.register_node(node);

    assert_eq!(registry.active_nodes().len(), 1);
    assert!(registry.get_node(&node_id).is_some());
}

#[test]
fn test_complete_partitioning_workflow() {
    // Create nodes with different capabilities
    let node_caps = vec![
        (
            NodeId::new(),
            NodeCapability {
                memory_gb: 32.0,
                compute_flops: 20e12,
                network_bandwidth_gbps: 25.0,
                num_devices: 2,
                device_type: "GPU-A100".to_string(),
            },
        ),
        (
            NodeId::new(),
            NodeCapability {
                memory_gb: 16.0,
                compute_flops: 10e12,
                network_bandwidth_gbps: 10.0,
                num_devices: 1,
                device_type: "GPU-V100".to_string(),
            },
        ),
        (
            NodeId::new(),
            NodeCapability {
                memory_gb: 64.0,
                compute_flops: 30e12,
                network_bandwidth_gbps: 40.0,
                num_devices: 4,
                device_type: "GPU-A100".to_string(),
            },
        ),
    ];

    // Test sequential partitioning
    let config = PartitionConfig {
        strategy: PartitionStrategy::Sequential,
        num_layers: 24,
        num_nodes: 3,
    };

    let mut partition_manager = PartitionManager::new(config).unwrap();
    partition_manager.partition_layers(&node_caps).unwrap();

    // Verify all layers are assigned
    for layer_id in 0..24 {
        assert!(partition_manager.get_layer_node(layer_id).is_some());
    }

    assert!(partition_manager.is_fully_partitioned());
}

#[test]
fn test_load_balancer_workflow() {
    let mut lb = LoadBalancer::new();

    let node1 = NodeId::new();
    let node2 = NodeId::new();

    let cap1 = NodeCapability {
        memory_gb: 32.0,
        compute_flops: 20e12,
        network_bandwidth_gbps: 25.0,
        num_devices: 2,
        device_type: "GPU-A100".to_string(),
    };

    let cap2 = NodeCapability {
        memory_gb: 16.0,
        compute_flops: 10e12,
        network_bandwidth_gbps: 10.0,
        num_devices: 1,
        device_type: "GPU-V100".to_string(),
    };

    lb.register_node(node1, cap1);
    lb.register_node(node2, cap2);

    // Set different loads
    lb.update_node_load(&node1, 0.8);
    lb.update_node_load(&node2, 0.3);

    let stats = lb.get_load_statistics();
    assert_eq!(stats.num_nodes, 2);
    assert!(stats.avg_load > 0.0);
    assert!(stats.std_dev > 0.0);

    // Test threshold-based node filtering
    let overloaded = lb.get_overloaded_nodes(0.7);
    assert_eq!(overloaded.len(), 1);

    let underloaded = lb.get_underloaded_nodes(0.5);
    assert_eq!(underloaded.len(), 1);
}

#[test]
fn test_health_monitoring() {
    let mut monitor = HealthMonitor::new(
        Duration::from_secs(5),
        Duration::from_secs(1),
        3,
    );

    let node_id = NodeId::new();
    monitor.register_node(node_id);

    assert_eq!(monitor.get_node_status(&node_id), Some(NodeStatus::Healthy));
    assert_eq!(monitor.total_nodes(), 1);
    assert_eq!(monitor.cluster_health_percentage(), 100.0);

    // Update heartbeat
    monitor.update_heartbeat(&node_id);
    assert_eq!(monitor.healthy_nodes().len(), 1);
}

#[test]
fn test_node_capability_validation() {
    // Valid capability
    let result = NodeCapability::new(
        32.0,
        20e12,
        25.0,
        2,
        "GPU-A100".to_string(),
    );
    assert!(result.is_ok());

    // Invalid memory
    let result = NodeCapability::new(
        -1.0,
        20e12,
        25.0,
        2,
        "GPU-A100".to_string(),
    );
    assert!(result.is_err());

    // Invalid compute
    let result = NodeCapability::new(
        32.0,
        0.0,
        25.0,
        2,
        "GPU-A100".to_string(),
    );
    assert!(result.is_err());

    // Invalid devices
    let result = NodeCapability::new(
        32.0,
        20e12,
        25.0,
        0,
        "GPU-A100".to_string(),
    );
    assert!(result.is_err());
}

#[test]
fn test_partition_manager_validation() {
    // Valid config
    let config = PartitionConfig {
        strategy: PartitionStrategy::Balanced,
        num_layers: 24,
        num_nodes: 3,
    };
    assert!(PartitionManager::new(config).is_ok());

    // Invalid config - zero layers
    let config = PartitionConfig {
        strategy: PartitionStrategy::Balanced,
        num_layers: 0,
        num_nodes: 3,
    };
    assert!(PartitionManager::new(config).is_err());

    // Invalid config - zero nodes
    let config = PartitionConfig {
        strategy: PartitionStrategy::Balanced,
        num_layers: 24,
        num_nodes: 0,
    };
    assert!(PartitionManager::new(config).is_err());
}

#[tokio::test]
async fn test_communication_layer_with_timeout() {
    let node_id = NodeId::new();
    let mut comm_layer = CommunicationLayer::new(node_id);

    let receiver = NodeId::new();
    let message = Message::new(
        node_id,
        receiver,
        MessageType::HealthCheck,
        vec![],
    );

    // Send message
    comm_layer.send_message(message).await.unwrap();

    // Receive with timeout
    let result = comm_layer.receive_message_timeout(Duration::from_secs(1)).await;
    assert!(result.is_ok());
}

#[test]
fn test_load_statistics_calculations() {
    let mut lb = LoadBalancer::new();

    let node1 = NodeId::new();
    let node2 = NodeId::new();
    let node3 = NodeId::new();

    let capability = NodeCapability {
        memory_gb: 16.0,
        compute_flops: 10e12,
        network_bandwidth_gbps: 10.0,
        num_devices: 1,
        device_type: "GPU".to_string(),
    };

    lb.register_node(node1, capability.clone());
    lb.register_node(node2, capability.clone());
    lb.register_node(node3, capability);

    lb.update_node_load(&node1, 0.3);
    lb.update_node_load(&node2, 0.5);
    lb.update_node_load(&node3, 0.7);

    let stats = lb.get_load_statistics();

    assert_eq!(stats.min_load, 0.3);
    assert_eq!(stats.max_load, 0.7);
    assert_eq!(stats.avg_load, 0.5);
    assert!(stats.std_dev > 0.0);

    // Test coefficient of variation
    let cv = stats.coefficient_of_variation();
    assert!(cv > 0.0);

    // Test imbalance factor
    let imbalance = stats.imbalance_factor();
    assert_eq!(imbalance, 0.8); // (0.7 - 0.3) / 0.5
}
