//! Multi-Node Distributed Inference Example
//!
//! This example demonstrates how to set up a distributed inference cluster
//! with multiple Butterfly nodes working together.
//!
//! Architecture:
//! - 1 Coordinator node (orchestrates tasks)
//! - N Worker nodes (perform inference)
//!
//! Run with: cargo run --example multi_node

use std::error::Error;
use std::time::Duration;
use tokio::time::sleep;

// TODO: Import butterfly crate once available
// use butterfly::{Node, CoordinatorConfig, WorkerConfig, Task};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    println!("🦋 Butterfly Multi-Node Distributed Inference");
    println!("==============================================\n");

    // Configuration
    let num_workers = 3;
    let coordinator_addr = "127.0.0.1:9000";

    println!("Configuration:");
    println!("  - Coordinator: {}", coordinator_addr);
    println!("  - Worker nodes: {}", num_workers);
    println!();

    // Step 1: Start coordinator node
    println!("Step 1: Starting coordinator node...");
    // TODO: Implement coordinator initialization
    // let coordinator_config = CoordinatorConfig::builder()
    //     .listen_addr(coordinator_addr)
    //     .max_workers(num_workers)
    //     .build()?;
    //
    // let coordinator = Node::coordinator(coordinator_config).await?;
    // coordinator.start().await?;

    println!("  ✓ Coordinator started at {}\n", coordinator_addr);

    // Step 2: Start worker nodes
    println!("Step 2: Starting {} worker nodes...", num_workers);
    // TODO: Implement worker initialization
    // let mut workers = Vec::new();
    //
    // for i in 0..num_workers {
    //     let worker_addr = format!("127.0.0.1:{}", 9001 + i);
    //     let worker_config = WorkerConfig::builder()
    //         .node_id(format!("worker-{}", i))
    //         .listen_addr(&worker_addr)
    //         .coordinator_addr(coordinator_addr)
    //         .build()?;
    //
    //     let worker = Node::worker(worker_config).await?;
    //     worker.start().await?;
    //     workers.push(worker);
    //
    //     println!("  ✓ Worker {} started at {}", i, worker_addr);
    // }

    for i in 0..num_workers {
        println!("  ✓ Worker {} started", i);
    }
    println!();

    // Step 3: Wait for cluster to stabilize
    println!("Step 3: Waiting for cluster to stabilize...");
    sleep(Duration::from_secs(2)).await;
    println!("  ✓ Cluster ready\n");

    // Step 4: Submit distributed workload
    println!("Step 4: Submitting distributed workload...");
    let num_tasks = 10;
    // TODO: Implement distributed task submission
    // let mut task_ids = Vec::new();
    //
    // for i in 0..num_tasks {
    //     let task = Task::builder()
    //         .input(format!("Task {}", i))
    //         .model("distributed-model")
    //         .build()?;
    //
    //     let task_id = coordinator.submit_task(task).await?;
    //     task_ids.push(task_id);
    // }

    println!("  ✓ Submitted {} tasks for distributed processing\n", num_tasks);

    // Step 5: Monitor progress
    println!("Step 5: Monitoring task progress...");
    // TODO: Implement progress monitoring
    // for (idx, task_id) in task_ids.iter().enumerate() {
    //     let status = coordinator.get_task_status(task_id).await?;
    //     println!("  Task {}: {:?}", idx, status);
    // }

    println!("  ✓ All tasks completed\n");

    // Step 6: Collect and aggregate results
    println!("Step 6: Collecting results...");
    // TODO: Implement result collection
    // let mut results = Vec::new();
    // for task_id in task_ids {
    //     let result = coordinator.get_result(task_id).await?;
    //     results.push(result);
    // }
    //
    // let aggregated = aggregate_results(results)?;
    // println!("  ✓ Aggregated result: {:?}\n", aggregated);

    println!("  ✓ Results collected\n");

    // Step 7: Display cluster statistics
    println!("Step 7: Cluster statistics:");
    // TODO: Implement statistics collection
    // let stats = coordinator.get_cluster_stats().await?;
    // println!("  - Total tasks processed: {}", stats.total_tasks);
    // println!("  - Average task latency: {:?}", stats.avg_latency);
    // println!("  - Throughput: {:.2} tasks/sec", stats.throughput);
    // println!("  - Active workers: {}", stats.active_workers);

    println!("  [Statistics pending implementation]");
    println!();

    // Step 8: Graceful shutdown
    println!("Step 8: Shutting down cluster...");
    // TODO: Implement graceful shutdown
    // for (idx, worker) in workers.into_iter().enumerate() {
    //     worker.shutdown().await?;
    //     println!("  ✓ Worker {} stopped", idx);
    // }
    // coordinator.shutdown().await?;
    // println!("  ✓ Coordinator stopped");

    println!("  ✓ Cluster shutdown complete\n");

    println!("==============================================");
    println!("🦋 Multi-node example completed successfully!");
    println!("\n[NOTE: This is a template. Actual implementation pending.]");

    Ok(())
}

/// Placeholder for result aggregation logic
#[allow(dead_code)]
fn aggregate_results(_results: Vec<String>) -> Result<String, Box<dyn Error>> {
    // TODO: Implement actual aggregation logic
    Ok("Aggregated result".to_string())
}

/// Helper to demonstrate load balancing strategies
#[allow(dead_code)]
fn demonstrate_load_balancing() {
    println!("Load Balancing Strategies:");
    println!("  - Round-robin: Distribute tasks evenly across workers");
    println!("  - Least-loaded: Send tasks to least busy worker");
    println!("  - Locality-aware: Prefer workers with cached data");
    println!("  - Performance-based: Weight by worker capabilities");
}
