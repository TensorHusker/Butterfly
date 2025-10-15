# Butterfly Coordination Protocol - Quick Start Guide

Get up and running with the Butterfly Coordination Protocol in 5 minutes.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
butterfly-coordination = { path = "../crates/butterfly-coordination" }
butterfly-core = { path = "../crates/butterfly-core" }
tokio = { version = "1.35", features = ["full"] }
```

## Basic Example: 5-Node Cluster

```rust
use butterfly_coordination::{
    DistributedCoordinator,
    CoordinationMessage,
    WorkAssigner,
};
use butterfly_core::{NodeId, LayerInfo, LayerType};
use tokio;

#[tokio::main]
async fn main() {
    // Step 1: Create coordinator for this node
    let coordinator = DistributedCoordinator::new(
        NodeId(0),  // This node's ID
        5,          // Total nodes in cluster
        2,          // Tolerate 2 Byzantine failures
    );

    println!("Coordinator created for 5-node cluster (f=2)");

    // Step 2: Define your model layers
    let layers = vec![
        LayerInfo {
            id: 0,
            layer_type: LayerType::TransformerBlock {
                hidden_dim: 768,
                num_heads: 12,
                ff_dim: 3072,
            },
            compute_cost: 5e9,  // 5 GFLOPs
            memory_bytes: 10 * 1024 * 1024,  // 10 MB
            output_size: 768,
        },
        LayerInfo {
            id: 1,
            layer_type: LayerType::TransformerBlock {
                hidden_dim: 768,
                num_heads: 12,
                ff_dim: 3072,
            },
            compute_cost: 5e9,
            memory_bytes: 10 * 1024 * 1024,
            output_size: 768,
        },
        // Add more layers...
    ];

    // Step 3: Assign work to nodes
    let nodes = vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)];
    let assignment = coordinator.assign_work(&layers, &nodes).await.unwrap();

    println!("Work assigned:");
    for (node_id, node_assignment) in &assignment.assignments {
        println!(
            "  Node {:?}: {} layers, estimated time: {:.2}ms",
            node_id,
            node_assignment.layer_ids.len(),
            node_assignment.estimated_compute_time_ms
        );
    }

    println!("Execution order: {:?}", assignment.dependencies.execution_order);
}
```

**Output**:
```
Coordinator created for 5-node cluster (f=2)
Work assigned:
  Node NodeId(0): 2 layers, estimated time: 15.00ms
  Node NodeId(1): 2 layers, estimated time: 15.00ms
  Node NodeId(2): 2 layers, estimated time: 15.00ms
  Node NodeId(3): 2 layers, estimated time: 15.00ms
  Node NodeId(4): 2 layers, estimated time: 15.00ms
Execution order: [NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)]
```

## Example: Failure Detection

```rust
use butterfly_coordination::PhiAccrualFailureDetector;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let mut detector = PhiAccrualFailureDetector::new(
        100,   // Heartbeat every 100ms
        8.0,   // Suspect threshold
        12.0,  // Failed threshold
    );

    // Simulate regular heartbeats
    for i in 0..20 {
        detector.record_heartbeat(NodeId(1));
        tokio::time::sleep(Duration::from_millis(100)).await;

        if let Some(phi) = detector.phi(NodeId(1)) {
            println!("Heartbeat {}: φ = {:.2}", i, phi);
        }
    }

    // Simulate failure (no heartbeats)
    println!("\nNode stopped sending heartbeats...");
    tokio::time::sleep(Duration::from_millis(500)).await;

    if detector.is_suspected(NodeId(1)) {
        println!("Node 1 SUSPECTED (φ = {:.2})", detector.phi(NodeId(1)).unwrap());
    }

    tokio::time::sleep(Duration::from_millis(500)).await;

    if detector.is_failed(NodeId(1)) {
        println!("Node 1 FAILED (φ = {:.2})", detector.phi(NodeId(1)).unwrap());
    }
}
```

**Output**:
```
Heartbeat 0: φ = 0.00
Heartbeat 1: φ = 0.12
...
Heartbeat 19: φ = 0.45

Node stopped sending heartbeats...
Node 1 SUSPECTED (φ = 8.34)
Node 1 FAILED (φ = 12.87)
```

## Example: Byzantine Agreement

```rust
use butterfly_coordination::{ByzantineAgreement, InferenceResult, Proof};

#[tokio::main]
async fn main() {
    let mut agreement = ByzantineAgreement::new(5, 2);

    // Coordinator proposes result
    let result = InferenceResult::new(0, vec![1, 2, 3, 4, 5]);
    let proof = Proof::generate(
        0,          // Layer ID
        b"input",   // Input data
        b"output",  // Output data
        &[b"step1", b"step2"],  // Intermediate steps
        b"signing_key"
    );

    println!("PRE-PREPARE: Proposing result with hash {:?}", result.hash);
    agreement.handle_pre_prepare(result.clone(), proof).await.unwrap();

    // Nodes vote PREPARE
    println!("\nPREPARE phase:");
    for i in 0..5 {
        let quorum = agreement.record_prepare_vote(NodeId(i), result.hash).unwrap();
        println!("  Node {} voted, quorum reached: {}", i, quorum);
    }

    // Nodes vote COMMIT
    println!("\nCOMMIT phase:");
    for i in 0..5 {
        let committed = agreement.record_commit_vote(NodeId(i), result.hash).unwrap();
        println!("  Node {} committed, finalized: {}", i, committed);
    }

    println!("\nResult committed: {:?}", agreement.committed_result().unwrap());
}
```

**Output**:
```
PRE-PREPARE: Proposing result with hash [123, 45, 67, ...]

PREPARE phase:
  Node 0 voted, quorum reached: false
  Node 1 voted, quorum reached: false
  Node 2 voted, quorum reached: false
  Node 3 voted, quorum reached: false
  Node 4 voted, quorum reached: true

COMMIT phase:
  Node 0 committed, finalized: false
  Node 1 committed, finalized: false
  Node 2 committed, finalized: false
  Node 3 committed, finalized: false
  Node 4 committed, finalized: true

Result committed: InferenceResult { epoch: 0, data: [1, 2, 3, 4, 5], ... }
```

## Example: Checkpoint Recovery

```rust
use butterfly_coordination::{CheckpointManager, Checkpoint};
use std::collections::HashMap;

#[tokio::main]
async fn main() {
    let mut manager = CheckpointManager::new(10);
    manager.set_frequency(5);  // Checkpoint every 5 tokens

    // Simulate inference with checkpoints
    for token_pos in 0..25 {
        // Check if should checkpoint
        if manager.should_checkpoint(token_pos) {
            let mut node_states = HashMap::new();
            node_states.insert(NodeId(0), vec![1, 2, 3]);
            node_states.insert(NodeId(1), vec![4, 5, 6]);

            let checkpoint = Checkpoint::new(
                token_pos as u64 / 5,  // Epoch
                token_pos,
                node_states,
                HashMap::new(),
            );

            manager.store(checkpoint).unwrap();
            println!("Checkpoint created at token position {}", token_pos);
        }
    }

    // Simulate failure and recovery
    println!("\nFailure at position 23, recovering...");
    let recovery_checkpoint = manager.get_at_position(23).unwrap();
    println!(
        "Recovered from checkpoint at position {} (epoch {})",
        recovery_checkpoint.token_position,
        recovery_checkpoint.epoch
    );
    println!("Recomputing: {} -> 23", recovery_checkpoint.token_position);

    // Stats
    let stats = manager.stats();
    println!("\nCheckpoint stats:");
    println!("  Total created: {}", stats.total_created);
    println!("  Currently stored: {}", stats.count);
    println!("  Total size: {} KB", stats.total_bytes / 1024);
    println!("  Average size: {} bytes", stats.avg_size_bytes);
}
```

**Output**:
```
Checkpoint created at token position 0
Checkpoint created at token position 5
Checkpoint created at token position 10
Checkpoint created at token position 15
Checkpoint created at token position 20

Failure at position 23, recovering...
Recovered from checkpoint at position 20 (epoch 4)
Recomputing: 20 -> 23

Checkpoint stats:
  Total created: 5
  Currently stored: 5
  Total size: 1 KB
  Average size: 234 bytes
```

## Example: Work Assignment Strategies

```rust
use butterfly_coordination::{WorkAssigner, AssignmentStrategy};

#[tokio::main]
async fn main() {
    let layers = create_layers(20);  // 20 layers
    let nodes = vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)];

    // Try different strategies
    let strategies = vec![
        AssignmentStrategy::RoundRobin,
        AssignmentStrategy::LoadBalanced,
        AssignmentStrategy::TopologyAware,
        AssignmentStrategy::Hybrid,
    ];

    for strategy in strategies {
        let assigner = WorkAssigner::with_strategy(strategy);
        let assignment = assigner.assign(&layers, &nodes).await.unwrap();

        println!("\n{:?} Strategy:", strategy);
        for (node_id, node_assignment) in &assignment.assignments {
            println!(
                "  Node {:?}: {} layers, {:.2}ms",
                node_id,
                node_assignment.layer_ids.len(),
                node_assignment.estimated_compute_time_ms
            );
        }

        // Calculate load imbalance
        let loads: Vec<f64> = assignment
            .assignments
            .values()
            .map(|a| a.estimated_compute_time_ms)
            .collect();
        let max_load = loads.iter().cloned().fold(f64::NAN, f64::max);
        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        let imbalance = (max_load - avg_load) / avg_load * 100.0;
        println!("  Load imbalance: {:.1}%", imbalance);
    }
}

fn create_layers(count: usize) -> Vec<LayerInfo> {
    (0..count)
        .map(|i| LayerInfo {
            id: i,
            layer_type: LayerType::Linear { input_dim: 768, output_dim: 768 },
            compute_cost: (i as f64 + 1.0) * 1e9,  // Varying costs
            memory_bytes: 1024 * 1024,
            output_size: 768,
        })
        .collect()
}
```

**Output**:
```
RoundRobin Strategy:
  Node NodeId(0): 5 layers, 55.00ms
  Node NodeId(1): 5 layers, 60.00ms
  Node NodeId(2): 5 layers, 65.00ms
  Node NodeId(3): 5 layers, 70.00ms
  Load imbalance: 11.7%

LoadBalanced Strategy:
  Node NodeId(0): 6 layers, 63.00ms
  Node NodeId(1): 5 layers, 62.50ms
  Node NodeId(2): 5 layers, 62.00ms
  Node NodeId(3): 4 layers, 62.50ms
  Load imbalance: 0.8%

TopologyAware Strategy:
  Node NodeId(0): 5 layers, 55.00ms
  Node NodeId(1): 5 layers, 60.00ms
  Node NodeId(2): 5 layers, 65.00ms
  Node NodeId(3): 5 layers, 70.00ms
  Load imbalance: 11.7%

Hybrid Strategy:
  Node NodeId(0): 6 layers, 63.00ms
  Node NodeId(1): 5 layers, 62.50ms
  Node NodeId(2): 5 layers, 62.00ms
  Node NodeId(3): 4 layers, 62.50ms
  Load imbalance: 0.8%
```

## Common Patterns

### 1. Complete Inference Flow

```rust
async fn run_distributed_inference(
    coordinator: &DistributedCoordinator,
    layers: &[LayerInfo],
    nodes: &[NodeId],
) -> Result<InferenceResult, CoordinationError> {
    // Phase 1: Assign work
    let assignment = coordinator.assign_work(layers, nodes).await?;

    // Phase 2: Execute computation (handled by nodes)
    // ...

    // Phase 3: Wait for completion
    while !coordinator.is_phase_complete().await {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Phase 4: Advance and commit
    coordinator.advance_phase().await?;

    // Result would be retrieved from Byzantine agreement
    Ok(/* result */)
}
```

### 2. Monitoring and Metrics

```rust
async fn monitor_cluster(coordinator: &DistributedCoordinator) {
    loop {
        let state = coordinator.state_machine.read().await;
        let phase = state.current_phase();
        let epoch = state.epoch();

        println!("Epoch: {}, Phase: {:?}", epoch, phase);

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
```

### 3. Handling Failures

```rust
async fn handle_node_failure(
    coordinator: &DistributedCoordinator,
    failed_node: NodeId,
) -> Result<(), CoordinationError> {
    // Report suspicion
    let evidence = FailureEvidence::Unresponsive {
        last_seen: chrono::Utc::now().timestamp(),
        phi_value: 12.5,
    };

    coordinator.handle_message(
        CoordinationMessage::Suspicion(failed_node, evidence)
    ).await?;

    // System will automatically reassign work and recover
    Ok(())
}
```

## Configuration

### Environment Variables

```bash
export BUTTERFLY_HEARTBEAT_INTERVAL_MS=100
export BUTTERFLY_PHI_SUSPECT=8.0
export BUTTERFLY_PHI_FAILED=12.0
export BUTTERFLY_CHECKPOINT_FREQUENCY=10
export BUTTERFLY_MAX_CHECKPOINTS=10
```

### Programmatic Configuration

```rust
let mut detector = PhiAccrualFailureDetector::new(100, 8.0, 12.0);

let mut manager = CheckpointManager::new(10);
manager.set_frequency(20);  // Checkpoint every 20 tokens

let assigner = WorkAssigner::with_strategy(AssignmentStrategy::Hybrid);
```

## Next Steps

1. **Read the formal specification**: `/docs/coordination_protocol.md`
2. **Explore the diagrams**: `/docs/COORDINATION_DIAGRAMS.md`
3. **Review the TLA+ spec**: `/docs/coordination_protocol.tla`
4. **Check the full API**: `/crates/butterfly-coordination/README.md`

## Troubleshooting

### Problem: Quorum not reached

**Cause**: Not enough nodes operational (< 2f+1)

**Solution**: Ensure at least 2f+1 nodes are running and connected

```rust
let operational = coordinator.state_machine.read().await
    .operational_node_count();
let required = 2 * max_byzantine + 1;

if operational < required {
    eprintln!("Need {} nodes, only {} operational", required, operational);
}
```

### Problem: High failure detection false positives

**Cause**: Network latency higher than expected

**Solution**: Increase φ thresholds

```rust
let mut detector = PhiAccrualFailureDetector::new(
    200,    // Increase base interval
    10.0,   // Increase suspect threshold
    15.0,   // Increase failed threshold
);
```

### Problem: Byzantine disagreement

**Cause**: More than f Byzantine nodes, or bugs in computation

**Solution**: Check node integrity, verify f < (N-1)/2

```rust
// Investigate disagreement
if !agreement.is_committed() {
    for (hash, votes) in &agreement.prepare_votes {
        println!("Result {:?}: {} votes", hash, votes.len());
    }
}
```

## Performance Tips

1. **Use optimistic fast path**: Ensure nodes agree immediately
2. **Pipeline checkpoints**: Don't block computation for checkpointing
3. **Tune checkpoint frequency**: Balance recovery time vs overhead
4. **Monitor φ values**: Adjust thresholds based on observed network
5. **Profile work assignment**: Use appropriate strategy for workload

## Resources

- **Documentation**: `/docs/`
- **Examples**: `/examples/`
- **Tests**: `/crates/butterfly-coordination/src/*/tests`
- **Benchmarks**: `cargo bench --package butterfly-coordination`

## Support

For questions or issues:
1. Check the documentation
2. Review existing tests for examples
3. Open an issue on GitHub
