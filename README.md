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
let mut partition_manager = PartitionManager::new(config);
partition_manager.partition_layers(&node_capabilities).unwrap();
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
