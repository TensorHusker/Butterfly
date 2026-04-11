//! Hello Butterfly - A simple example demonstrating basic usage
//!
//! This example shows how to:
//! 1. Create a single worker node
//! 2. Assign and execute work
//! 3. Verify results
//!
//! Run with: cargo run --example hello_butterfly

use butterfly_core::{NodeId, WorkerNode, WorkerConfig, DeviceType, ModelExecutor, WorkAssignment};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🦋 Welcome to Butterfly Distributed Inference!");
    println!("================================================\n");

    // Step 1: Create worker configuration
    println!("Step 1: Configuring worker node...");
    let worker_config = WorkerConfig {
        node_id: NodeId(1),
        device: DeviceType::Cpu,
        max_queue_size: 100,
        barrier_timeout_secs: 30,
        enable_byzantine: false,
    };
    println!("  ✓ Worker configured");
    println!("    - Node ID: {:?}", worker_config.node_id);
    println!("    - Device: {:?}", worker_config.device);
    println!();

    // Step 2: Create model executor
    println!("Step 2: Loading model executor...");
    let executor = ModelExecutor::new_test_model(DeviceType::Cpu)?;
    println!("  ✓ Model executor loaded");
    println!("    - Layers: {}", executor.num_layers());
    println!("    - Input shape: {:?}", executor.input_shape());
    println!();

    // Step 3: Initialize worker
    println!("Step 3: Initializing worker...");
    let mut worker = WorkerNode::new(worker_config, executor);
    worker.initialize()?;
    println!("  ✓ Worker initialized and ready");
    println!("    - State: {:?}", worker.get_state());
    println!();

    // Step 4: Create work assignment
    println!("Step 4: Creating work assignment...");
    let assignment = WorkAssignment {
        assignment_id: 1,
        layer_range: (0, 6),
        input_data: vec![1.0; 7680], // Mock input tensor
        expected_output_size: 7680,
        superstep: 0,
    };
    println!("  ✓ Work assignment created");
    println!("    - Assignment ID: {}", assignment.assignment_id);
    println!("    - Layer range: {:?}", assignment.layer_range);
    println!("    - Input size: {} elements", assignment.input_data.len());
    println!();

    // Step 5: Assign work to worker
    println!("Step 5: Assigning work...");
    worker.handle_work_assignment(assignment)?;
    println!("  ✓ Work assigned");
    println!("    - State: {:?}", worker.get_state());
    println!();

    // Step 6: Execute computation
    println!("Step 6: Executing computation...");
    let result = worker.execute_assigned_work()?;
    println!("  ✓ Computation completed");
    println!("    - Execution time: {:.2}ms", result.execution_time_ms);
    println!("    - Output size: {} elements", result.output_data.len());
    println!("    - State: {:?}", worker.get_state());
    println!();

    // Step 7: Verify result
    println!("Step 7: Verifying result...");
    assert_eq!(result.assignment_id, 1, "Assignment ID mismatch");
    assert_eq!(result.output_data.len(), 7680, "Output size mismatch");
    assert_eq!(result.node_id, NodeId(1), "Node ID mismatch");
    println!("  ✓ Result verified successfully");
    println!();

    println!("================================================");
    println!("🦋 Hello Butterfly example completed successfully!");
    println!();
    println!("What just happened:");
    println!("  1. Created a worker node with CPU executor");
    println!("  2. Assigned layers 0-6 for computation");
    println!("  3. Executed inference on mock input data");
    println!("  4. Verified output matches expectations");
    println!();
    println!("Next steps:");
    println!("  - Try: cargo run --example distributed_inference");
    println!("  - Read: docs/architecture.md");
    println!("  - Explore: docs/coordination_protocol.md");

    Ok(())
}
