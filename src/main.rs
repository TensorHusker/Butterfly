use butterfly::{
    Node, NodeCapability, NodeRegistry,
    PartitionConfig, PartitionStrategy, PartitionManager,
    LoadBalancer,
    HealthMonitor,
};
use std::time::Duration;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Butterfly: Distributed Inference System for Large Language Models");
    println!("==================================================================\n");

    // Create node registry
    let mut node_registry = NodeRegistry::new();

    // Create multiple nodes with different capabilities
    let node1 = Node::new(
        "node1:8080".to_string(),
        NodeCapability {
            memory_gb: 32.0,
            compute_flops: 20e12,
            network_bandwidth_gbps: 25.0,
            num_devices: 2,
            device_type: "GPU-A100".to_string(),
        },
    );

    let node2 = Node::new(
        "node2:8080".to_string(),
        NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU-V100".to_string(),
        },
    );

    let node3 = Node::new(
        "node3:8080".to_string(),
        NodeCapability {
            memory_gb: 64.0,
            compute_flops: 30e12,
            network_bandwidth_gbps: 40.0,
            num_devices: 4,
            device_type: "GPU-A100".to_string(),
        },
    );

    println!("Registered Nodes:");
    println!("- Node 1: {} (2x GPU-A100, 32GB)", node1.info.address);
    println!("- Node 2: {} (1x GPU-V100, 16GB)", node2.info.address);
    println!("- Node 3: {} (4x GPU-A100, 64GB)", node3.info.address);

    let node1_id = node_registry.register_node(node1);
    let node2_id = node_registry.register_node(node2);
    let node3_id = node_registry.register_node(node3);

    // Configure partitioning for a 24-layer transformer
    let config = PartitionConfig {
        strategy: PartitionStrategy::Balanced,
        num_layers: 24,
        num_nodes: 3,
    };

    println!("\nPartitioning Strategy: {:?}", config.strategy);
    println!("Total Layers: {}", config.num_layers);

    // Create partition manager and distribute layers
    let mut partition_manager = PartitionManager::new(config)
        .expect("Failed to create partition manager");
    
    let node_capabilities = vec![
        (node1_id, NodeCapability {
            memory_gb: 32.0,
            compute_flops: 20e12,
            network_bandwidth_gbps: 25.0,
            num_devices: 2,
            device_type: "GPU-A100".to_string(),
        }),
        (node2_id, NodeCapability {
            memory_gb: 16.0,
            compute_flops: 10e12,
            network_bandwidth_gbps: 10.0,
            num_devices: 1,
            device_type: "GPU-V100".to_string(),
        }),
        (node3_id, NodeCapability {
            memory_gb: 64.0,
            compute_flops: 30e12,
            network_bandwidth_gbps: 40.0,
            num_devices: 4,
            device_type: "GPU-A100".to_string(),
        }),
    ];

    partition_manager.partition_layers(&node_capabilities).unwrap();

    println!("\nLayer Distribution:");
    for layer_id in 0..24 {
        if let Some(node_id) = partition_manager.get_layer_node(layer_id) {
            let node_name = if node_id == node1_id {
                "Node 1"
            } else if node_id == node2_id {
                "Node 2"
            } else {
                "Node 3"
            };
            println!("  Layer {}: {}", layer_id, node_name);
        }
    }

    // Initialize load balancer
    let mut load_balancer = LoadBalancer::new();
    load_balancer.register_node(node1_id, node_capabilities[0].1.clone());
    load_balancer.register_node(node2_id, node_capabilities[1].1.clone());
    load_balancer.register_node(node3_id, node_capabilities[2].1.clone());

    // Simulate some load
    load_balancer.update_node_load(&node1_id, 0.6);
    load_balancer.update_node_load(&node2_id, 0.3);
    load_balancer.update_node_load(&node3_id, 0.7);

    let stats = load_balancer.get_load_statistics();
    println!("\nLoad Balancing Statistics:");
    println!("  Average Load: {:.2}%", stats.avg_load * 100.0);
    println!("  Min Load: {:.2}%", stats.min_load * 100.0);
    println!("  Max Load: {:.2}%", stats.max_load * 100.0);
    println!("  Standard Deviation: {:.3}", stats.std_dev);
    println!("  Imbalance Factor: {:.3}", stats.imbalance_factor());
    println!("  Coefficient of Variation: {:.3}", stats.coefficient_of_variation());

    // Initialize health monitor
    let health_monitor = HealthMonitor::new(
        Duration::from_secs(30),
        Duration::from_secs(10),
        3,
    );

    println!("\nFault Tolerance:");
    println!("  Health Check Timeout: 30s");
    println!("  Health Check Interval: 10s");
    println!("  Max Consecutive Failures: 3");

    println!("\n✓ Butterfly system initialized successfully!");
    println!("  - {} active nodes", node_registry.active_nodes().len());
    println!("  - {} transformer layers distributed", 24);
    println!("  - Load balancing enabled");
    println!("  - Fault tolerance monitoring ready");
}
