# Butterfly

Butterfly is a distributed inference system for transformers and large language models. It solves LLM scaling problems by distributing inference across multiple nodes, enabling efficient computation for models too large for single machines. Butterfly uses novel algorithms for partitioning transformers and minimizing communication overhead.

## Features

### Core Architecture
- **Distributed Transformer Layers**: Efficient partitioning of attention and feed-forward computations across multiple nodes
- **Flexible Partitioning Strategies**: 
  - Sequential partitioning for simple layer distribution
  - Balanced partitioning based on hardware capabilities
  - Custom partitioning for advanced use cases

### Load Balancing
- **Hardware-Aware Distribution**: Automatically detects and adapts to heterogeneous hardware configurations
- **Dynamic Load Monitoring**: Real-time tracking of node utilization
- **Optimal Work Distribution**: Distributes computational load proportionally to node capabilities

### Fault Tolerance
- **Health Monitoring**: Continuous monitoring of node health with configurable timeouts
- **Automatic Failure Detection**: Detects unresponsive or failed nodes
- **Workload Redistribution**: Automatically redistributes work from failed nodes to healthy ones

### Communication Layer
- **Efficient Inter-Node Messaging**: Asynchronous communication with minimal overhead
- **Message Types**: Support for forward passes, backward gradients, health checks, and more
- **Broadcast Capabilities**: Efficient one-to-many communication patterns

## Architecture

Butterfly is built with:
- **Rust**: For performance and safety
- **Burn Framework**: For deep learning tensor operations
- **Tokio**: For asynchronous runtime and networking

### Components

1. **Node Management** (`node.rs`): Manages compute nodes and their capabilities
2. **Partitioning** (`partition.rs`): Distributes transformer layers across nodes
3. **Communication** (`communication.rs`): Handles inter-node message passing
4. **Distributed Computation** (`distributed.rs`): Implements distributed attention and feed-forward layers
5. **Fault Tolerance** (`fault_tolerance.rs`): Monitors health and handles failures
6. **Load Balancing** (`load_balancer.rs`): Optimizes work distribution across heterogeneous hardware

## Key Enhancements

### Robust Error Handling
- Comprehensive validation in constructors (NodeCapability, PartitionManager)
- Type-safe error handling with thiserror
- Detailed error messages for debugging

### Advanced Communication
- Retry logic with exponential backoff
- Timeout support for all operations
- Configurable retry attempts and timeouts

### Enhanced Metrics
- Standard deviation and coefficient of variation for load statistics
- Cluster health percentage tracking
- Threshold-based node filtering (overloaded/underloaded nodes)

### Improved Fault Tolerance
- Degraded node state tracking
- Time-since-heartbeat monitoring
- Cluster-wide health percentage calculation

## Getting Started

### Prerequisites
- Rust 1.70 or later
- Cargo

### Building
```bash
cargo build --release
```

### Running
```bash
cargo run
```

### Testing
```bash
cargo test
```

## Example Usage

```rust
use butterfly::{
    Node, NodeCapability, NodeRegistry,
    PartitionConfig, PartitionStrategy, PartitionManager,
    LoadBalancer,
};

// Create nodes with different capabilities
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

// Register nodes
let mut registry = NodeRegistry::new();
let node1_id = registry.register_node(node1);

// Configure partitioning for a 24-layer transformer
let config = PartitionConfig {
    strategy: PartitionStrategy::Balanced,
    num_layers: 24,
    num_nodes: 3,
};

// Distribute layers across nodes
let mut partition_manager = PartitionManager::new(config).unwrap();
partition_manager.partition_layers(&node_capabilities).unwrap();
```

### Advanced Usage: Load Balancing

```rust
use butterfly::{LoadBalancer, NodeId, NodeCapability};

let mut load_balancer = LoadBalancer::new();

// Register nodes with their capabilities
load_balancer.register_node(node_id, capability);

// Update load metrics
load_balancer.update_node_load(&node_id, 0.75); // 75% loaded

// Get load statistics
let stats = load_balancer.get_load_statistics();
println!("Average load: {:.2}%", stats.avg_load * 100.0);
println!("Standard deviation: {:.3}", stats.std_dev);
println!("Coefficient of variation: {:.3}", stats.coefficient_of_variation());

// Check if load is well-balanced
if stats.is_well_balanced(0.2) {
    println!("Cluster load is well-balanced!");
}

// Find overloaded/underloaded nodes
let overloaded = load_balancer.get_overloaded_nodes(0.8);
let underloaded = load_balancer.get_underloaded_nodes(0.3);
```

### Advanced Usage: Health Monitoring

```rust
use butterfly::{HealthMonitor, NodeStatus};
use std::time::Duration;

let mut health_monitor = HealthMonitor::new(
    Duration::from_secs(30),  // health check timeout
    Duration::from_secs(10),  // health check interval
    3,                        // max consecutive failures
);

// Register nodes for monitoring
health_monitor.register_node(node_id);

// Update heartbeat when node responds
health_monitor.update_heartbeat(&node_id);

// Check cluster health
let health_percentage = health_monitor.cluster_health_percentage();
println!("Cluster health: {:.1}%", health_percentage);

// Get nodes by status
let healthy = health_monitor.healthy_nodes();
let degraded = health_monitor.degraded_nodes();
let failed = health_monitor.failed_nodes();
```

### Advanced Usage: Communication with Retry

```rust
use butterfly::{CommunicationLayer, Message, MessageType};
use std::time::Duration;

// Create communication layer with custom config
let comm_layer = CommunicationLayer::with_config(
    node_id,
    Duration::from_secs(30),  // default timeout
    5,                        // max retries
);

// Send message with automatic retry
let message = Message::new(sender, receiver, MessageType::HealthCheck, vec![]);
comm_layer.send_message_with_retry(message).await?;

// Receive with timeout
let message = comm_layer.receive_message_timeout(Duration::from_secs(5)).await?;
```

## Performance Characteristics

- **Minimized Communication Overhead**: Strategic layer placement reduces inter-node data transfer
- **Load-Balanced Execution**: Work is distributed proportionally to node capabilities
- **Fault-Resilient**: Continues operation even when nodes fail
- **Scalable**: Designed to work with 2 to 100+ nodes

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
