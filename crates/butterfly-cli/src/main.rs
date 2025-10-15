//! # Butterfly CLI
//!
//! Command-line interface for the Butterfly distributed inference system.
//! Provides commands for starting nodes, managing clusters, and monitoring performance.

use clap::{Parser, Subcommand};
use butterfly_core::NodeId;

#[derive(Parser)]
#[command(name = "butterfly")]
#[command(about = "Butterfly distributed inference system", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a node in the distributed system
    Start {
        /// Node ID for this instance
        #[arg(short, long)]
        node_id: u64,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Query the status of the cluster
    Status {
        /// API endpoint to query
        #[arg(short, long, default_value = "http://localhost:8080")]
        endpoint: String,
    },

    /// Submit an inference task
    Submit {
        /// Input data file
        #[arg(short, long)]
        input: String,

        /// API endpoint to submit to
        #[arg(short, long, default_value = "http://localhost:8080")]
        endpoint: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Start { node_id, port } => {
            println!("Starting Butterfly node {} on port {}", node_id, port);
            start_node(NodeId(node_id), port).await;
        }
        Commands::Status { endpoint } => {
            println!("Querying status from {}", endpoint);
            query_status(&endpoint).await;
        }
        Commands::Submit { input, endpoint } => {
            println!("Submitting inference task from {} to {}", input, endpoint);
            submit_task(&input, &endpoint).await;
        }
    }
}

async fn start_node(node_id: NodeId, port: u16) {
    println!("Node {:?} listening on port {}", node_id, port);

    let state = butterfly_api::ApiState::new();
    let app = butterfly_api::create_router(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();

    println!("Server running on http://0.0.0.0:{}", port);
    axum::serve(listener, app).await.unwrap();
}

async fn query_status(endpoint: &str) {
    println!("Status query to {} not yet implemented", endpoint);
}

async fn submit_task(input: &str, endpoint: &str) {
    println!("Task submission from {} to {} not yet implemented", input, endpoint);
}
