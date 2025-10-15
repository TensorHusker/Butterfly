//! Hello Butterfly - A simple example demonstrating basic usage
//!
//! This example shows how to:
//! 1. Initialize a Butterfly node
//! 2. Submit a simple inference task
//! 3. Retrieve and process results
//!
//! Run with: cargo run --example hello_butterfly

use std::error::Error;

// TODO: Import butterfly crate once available
// use butterfly::{Node, Task, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init();

    println!("🦋 Welcome to Butterfly Distributed Inference!");
    println!("================================================\n");

    // Step 1: Create a node configuration
    println!("Step 1: Configuring node...");
    // TODO: Implement node configuration
    // let config = Config::builder()
    //     .node_id("hello-node-1")
    //     .listen_addr("127.0.0.1:8080")
    //     .build()?;

    println!("  ✓ Node configured\n");

    // Step 2: Initialize the node
    println!("Step 2: Initializing node...");
    // TODO: Implement node initialization
    // let node = Node::new(config).await?;
    // node.start().await?;

    println!("  ✓ Node started and ready\n");

    // Step 3: Create a simple inference task
    println!("Step 3: Creating inference task...");
    // TODO: Implement task creation
    // let task = Task::builder()
    //     .input("Hello, Butterfly!")
    //     .model("simple-echo")
    //     .build()?;

    println!("  ✓ Task created\n");

    // Step 4: Submit the task
    println!("Step 4: Submitting task for inference...");
    // TODO: Implement task submission
    // let task_id = node.submit_task(task).await?;
    // println!("  ✓ Task submitted with ID: {}\n", task_id);

    // Step 5: Wait for results
    println!("Step 5: Waiting for results...");
    // TODO: Implement result retrieval
    // let result = node.get_result(task_id).await?;
    // println!("  ✓ Received result: {:?}\n", result);

    // Step 6: Clean shutdown
    println!("Step 6: Shutting down node...");
    // TODO: Implement graceful shutdown
    // node.shutdown().await?;
    println!("  ✓ Node shutdown complete\n");

    println!("================================================");
    println!("🦋 Hello Butterfly example completed successfully!");

    // Placeholder success for now
    println!("\n[NOTE: This is a template. Actual implementation pending.]");

    Ok(())
}

/// Helper function to demonstrate error handling
#[allow(dead_code)]
fn demonstrate_error_handling() -> Result<(), Box<dyn Error>> {
    // Example of proper error handling patterns
    // TODO: Add actual error scenarios once APIs are available
    Ok(())
}

/// Helper function to show configuration options
#[allow(dead_code)]
fn show_configuration_options() {
    println!("Available configuration options:");
    println!("  - node_id: Unique identifier for this node");
    println!("  - listen_addr: Address to listen for connections");
    println!("  - peer_addrs: Addresses of peer nodes");
    println!("  - model_path: Path to model weights");
    println!("  - batch_size: Batch size for inference");
    println!("  - timeout: Task timeout duration");
}
