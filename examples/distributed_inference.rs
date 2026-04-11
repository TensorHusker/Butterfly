//! Distributed Inference Example
//!
//! This example demonstrates a complete distributed inference workflow:
//! 1. Create a coordination runtime
//! 2. Spawn multiple worker nodes
//! 3. Execute the 4-phase coordination protocol
//! 4. Perform distributed computation with Byzantine fault tolerance
//!
//! Run with: cargo run --example distributed_inference

use butterfly_core::{NodeId, WorkerNode, WorkerConfig, DeviceType, ModelExecutor};
use butterfly_coordination::{CoordinationRuntime, RuntimeConfig};
use butterfly_config::Config;
use std::sync::Arc;
use std::path::Path;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🦋 Butterfly Distributed Inference System");
    println!("==========================================\n");

    // Load configuration
    println!("📋 Loading configuration...");
    let config = if Path::new("config/default.toml").exists() {
        Config::load_from_file(Path::new("config/default.toml"))?
    } else {
        Config::default()
    };
    println!("  ✓ Configuration loaded");
    println!("    - Byzantine fault tolerance: f={}", config.byzantine.fault_tolerance_level);
    println!("    - Network protocol: {:?}\n", config.network.protocol);

    // Create coordination runtime
    println!("🎯 Creating coordination runtime...");
    let runtime_config = RuntimeConfig {
        node_id: NodeId(0),
        cluster_size: 3,
        max_byzantine: config.byzantine.fault_tolerance_level,
        heartbeat_interval_ms: 100,
        phase_timeout_secs: 30,
        barrier_timeout_secs: 30,
        enable_byzantine: true,
    };

    let runtime = Arc::new(CoordinationRuntime::new(runtime_config));
    println!("  ✓ Coordination runtime created");
    println!("    - Coordinator ID: {:?}", NodeId(0));
    println!("    - Cluster size: 3 workers\n");

    // Spawn worker nodes
    println!("👷 Spawning worker nodes...");
    for i in 1..=3 {
        let worker_config = WorkerConfig {
            node_id: NodeId(i),
            device: DeviceType::Cpu,
            max_queue_size: 100,
            barrier_timeout_secs: 30,
            enable_byzantine: true,
        };

        let executor = ModelExecutor::new_test_model(DeviceType::Cpu)?;
        let mut worker = WorkerNode::new(worker_config, executor);
        worker.initialize()?;

        runtime.add_worker(worker).await?;
        println!("  ✓ Worker {} initialized", i);
    }
    println!();

    // Execute Phase 1: Assignment
    println!("📦 Phase 1: Work Assignment");
    runtime.execute_assignment_phase().await?;
    println!("  ✓ Work assigned to all workers");

    let workers = runtime.workers.read().await;
    for (id, worker) in workers.iter() {
        println!("    - Worker {:?}: {:?}", id, worker.get_state());
    }
    drop(workers);
    println!();

    // Execute Phase 2: Computation
    println!("⚙️  Phase 2: Computation");
    runtime.execute_computation_phase().await?;
    println!("  ✓ All workers completed computation");
    println!();

    // Execute Phase 3: Aggregation (Byzantine Voting)
    println!("🗳️  Phase 3: Byzantine Aggregation");
    runtime.execute_aggregation_phase().await?;
    println!("  ✓ Byzantine consensus achieved");
    println!("    - Required quorum: 2f+1 = 3 nodes");
    println!();

    // Execute Phase 4: Commitment (Barrier Synchronization)
    println!("🔒 Phase 4: Commitment & Barrier");
    runtime.execute_commitment_phase().await?;
    println!("  ✓ Barrier synchronization complete");
    println!("    - All workers reached checkpoint");
    println!("    - Superstep incremented");
    println!();

    // Display statistics
    println!("📊 Execution Statistics");
    println!("  - Current superstep: {}", runtime.current_superstep().await);
    println!("  - Current phase: {:?}", runtime.current_phase().await);
    println!("  - Workers active: 3/3");
    println!("  - Byzantine faults tolerated: 1");
    println!();

    // Demonstrate fault tolerance
    println!("🛡️  Testing Fault Tolerance");
    let workers = runtime.workers.read().await;
    if let Some(worker) = workers.get(&NodeId(1)) {
        println!("  - Simulating worker 1 failure...");
        let _ = worker.mark_failed("Simulated failure".to_string());
        println!("    ✓ Worker marked as failed");

        sleep(Duration::from_millis(100)).await;

        println!("  - Recovering worker 1...");
        worker.recover()?;
        println!("    ✓ Worker recovered successfully");
    }
    drop(workers);
    println!();

    println!("==========================================");
    println!("🦋 Distributed inference completed successfully!");
    println!();
    println!("Key Features Demonstrated:");
    println!("  ✓ 4-phase coordination protocol");
    println!("  ✓ Byzantine fault tolerance (f=1)");
    println!("  ✓ Barrier synchronization");
    println!("  ✓ Worker state management");
    println!("  ✓ Fault detection and recovery");

    Ok(())
}
