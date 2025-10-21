//! Graph processing commands for Butterfly CLI
//!
//! Provides commands for running graph algorithms, partitioning graphs,
//! and analyzing graph statistics.

use anyhow::{Context, Result};
use butterfly_core::{
    Graph, GraphBuilder, Vertex, VertexId, Edge, EdgeId, VertexCompute,
    VertexData, Message, MessagePayload, ComputationResult,
};
use butterfly_graph::{PageRank, PageRankConfig, ConnectedComponents, ShortestPaths};
use butterfly_partition::{
    GraphPartitioner, HashPartitioner, StreamPartitioner, METISPartitioner,
    GraphPartitionQuality,
};
use clap::{Subcommand, ValueEnum, Args};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Graph processing subcommands
#[derive(Subcommand)]
pub enum GraphCommands {
    /// Run a graph algorithm
    Run {
        #[command(flatten)]
        algorithm: AlgorithmArgs,

        /// Input graph file (edge list format: source target [weight])
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Partition a graph for distributed processing
    Partition {
        /// Input graph file
        #[arg(short, long)]
        input: PathBuf,

        /// Number of partitions
        #[arg(short = 'n', long, default_value = "4")]
        partitions: usize,

        /// Partitioning strategy
        #[arg(short, long, value_enum, default_value = "stream")]
        strategy: PartitionStrategy,

        /// Output directory for partition files
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Display graph statistics
    Stats {
        /// Input graph file
        #[arg(short, long)]
        input: PathBuf,
    },
}

/// Algorithm selection and parameters
#[derive(Args)]
pub struct AlgorithmArgs {
    /// Algorithm to run
    #[arg(value_enum)]
    pub algorithm: Algorithm,

    /// Maximum iterations (for iterative algorithms)
    #[arg(long, default_value = "100")]
    pub iterations: u32,

    /// Damping factor for PageRank
    #[arg(long, default_value = "0.85")]
    pub damping: f64,

    /// Source vertex for SSSP
    #[arg(long, default_value = "0")]
    pub source: u64,

    /// Convergence threshold
    #[arg(long, default_value = "0.001")]
    pub threshold: f64,
}

/// Available graph algorithms
#[derive(Clone, ValueEnum)]
pub enum Algorithm {
    /// PageRank algorithm
    PageRank,
    /// Connected Components
    ConnectedComponents,
    /// Single-Source Shortest Paths
    SSSP,
}

/// Partitioning strategies
#[derive(Clone, ValueEnum)]
pub enum PartitionStrategy {
    /// Hash-based partitioning
    Hash,
    /// Stream-based partitioning
    Stream,
    /// METIS-style partitioning
    METIS,
}

/// Execute graph processing commands
pub fn execute_graph_command(command: GraphCommands) -> Result<()> {
    match command {
        GraphCommands::Run {
            algorithm,
            input,
            output,
        } => run_algorithm(algorithm, input, output),
        GraphCommands::Partition {
            input,
            partitions,
            strategy,
            output,
        } => partition_graph(input, partitions, strategy, output),
        GraphCommands::Stats { input } => display_stats(input),
    }
}

/// Run a graph algorithm
fn run_algorithm(args: AlgorithmArgs, input: PathBuf, output: PathBuf) -> Result<()> {
    println!("Loading graph from {:?}...", input);
    let graph = load_graph(&input)?;

    println!("Running {:?} algorithm...", args.algorithm);
    println!("  Vertices: {}", graph.vertex_count());
    println!("  Edges: {}", graph.edge_count());

    match args.algorithm {
        Algorithm::PageRank => {
            let config = PageRankConfig {
                damping_factor: args.damping,
                convergence_threshold: args.threshold,
                max_iterations: args.iterations,
            };
            let pagerank = PageRank::new(config);
            let results = run_vertex_centric(&graph, pagerank, args.iterations)?;
            export_f64_results(&results, "rank", &output)?;
            println!("PageRank results written to {:?}", output);
        }
        Algorithm::ConnectedComponents => {
            let cc = ConnectedComponents::new();
            let results = run_vertex_centric(&graph, cc, args.iterations)?;
            export_u64_results(&results, "component", &output)?;
            println!("Connected components results written to {:?}", output);
        }
        Algorithm::SSSP => {
            let sssp = ShortestPaths::new(VertexId(args.source));
            let results = run_vertex_centric(&graph, sssp, args.iterations)?;
            export_f64_results(&results, "distance", &output)?;
            println!("Shortest paths results written to {:?}", output);
        }
    }

    Ok(())
}

/// Run a vertex-centric computation
fn run_vertex_centric<V: VertexCompute>(
    graph: &Graph,
    mut compute: V,
    max_iterations: u32,
) -> Result<HashMap<VertexId, VertexData>> {
    let mut vertices: HashMap<VertexId, Vertex> = graph
        .vertices
        .iter()
        .map(|(id, v)| (*id, v.clone()))
        .collect();

    // Initialize all vertices
    for vertex in vertices.values_mut() {
        compute.init(vertex)?;
    }

    let mut message_buffer: HashMap<VertexId, Vec<Message>> = HashMap::new();
    let mut superstep = 0u32;

    while superstep < max_iterations {
        message_buffer.clear();
        let mut active_count = 0;
        let mut results = Vec::new();

        // Compute phase
        for (vertex_id, vertex) in &vertices {
            let messages = message_buffer
                .get(vertex_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let result = compute.compute(vertex, messages, superstep)?;

            if result.is_active {
                active_count += 1;
            }

            // Collect outgoing messages
            for msg in &result.outgoing_messages {
                message_buffer
                    .entry(msg.target)
                    .or_insert_with(Vec::new)
                    .push(msg.clone());
            }

            results.push((vertex_id, result));
        }

        // Update vertex state
        for (vertex_id, result) in results {
            if let Some(vertex) = vertices.get_mut(vertex_id) {
                vertex.data = result.new_value.clone();
                if result.is_active {
                    vertex.activate();
                } else {
                    vertex.deactivate();
                }
            }
        }

        // Check termination
        if compute.should_terminate(superstep, active_count) {
            println!(
                "Algorithm converged at superstep {} with {} active vertices",
                superstep, active_count
            );
            break;
        }

        superstep += 1;
        if superstep % 10 == 0 {
            println!("Superstep {}: {} active vertices", superstep, active_count);
        }
    }

    if superstep >= max_iterations {
        println!("Reached maximum iterations ({})", max_iterations);
    }

    // Extract results
    Ok(vertices
        .into_iter()
        .map(|(id, v)| (id, v.data))
        .collect())
}

/// Partition a graph
fn partition_graph(
    input: PathBuf,
    num_partitions: usize,
    strategy: PartitionStrategy,
    output_dir: PathBuf,
) -> Result<()> {
    println!("Loading graph from {:?}...", input);
    let graph = load_graph(&input)?;

    println!("Partitioning graph into {} parts using {:?} strategy...", num_partitions, strategy);

    let partitioner: Box<dyn GraphPartitioner> = match strategy {
        PartitionStrategy::Hash => Box::new(HashPartitioner::new()),
        PartitionStrategy::Stream => Box::new(StreamPartitioner::new()),
        PartitionStrategy::METIS => Box::new(METISPartitioner::new()),
    };

    let subgraphs = partitioner
        .partition(&graph, num_partitions)
        .context("Failed to partition graph")?;

    // Calculate quality metrics
    let quality = partitioner.estimate_quality(&subgraphs);
    println!("\nPartition Quality Metrics:");
    println!("  Edge cut: {}", quality.edge_cut);
    println!("  Load balance: {:.3}", quality.load_balance);
    println!("  Communication volume: {}", quality.communication_volume);
    println!("  Replication factor: {:.3}", quality.replication_factor);

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Write partition files
    for (i, subgraph) in subgraphs.iter().enumerate() {
        let partition_file = output_dir.join(format!("partition_{}.txt", i));
        export_partition(&subgraph.graph, &partition_file)?;
        println!("Partition {} written to {:?} ({} vertices, {} edges)",
                 i, partition_file, subgraph.local_vertex_count(), subgraph.edge_count());
    }

    // Write partition metadata
    let metadata_file = output_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "num_partitions": num_partitions,
        "strategy": format!("{:?}", strategy),
        "quality": {
            "edge_cut": quality.edge_cut,
            "load_balance": quality.load_balance,
            "communication_volume": quality.communication_volume,
            "replication_factor": quality.replication_factor,
        },
        "partitions": subgraphs.iter().enumerate().map(|(i, sg)| {
            serde_json::json!({
                "id": i,
                "vertices": sg.local_vertex_count(),
                "edges": sg.edge_count(),
                "boundary_vertices": sg.boundary_vertices.len(),
                "ghost_vertices": sg.ghost_vertex_count(),
            })
        }).collect::<Vec<_>>(),
    });

    let mut meta_file = File::create(&metadata_file)?;
    serde_json::to_writer_pretty(&meta_file, &metadata)?;
    println!("\nMetadata written to {:?}", metadata_file);

    Ok(())
}

/// Display graph statistics
fn display_stats(input: PathBuf) -> Result<()> {
    println!("Loading graph from {:?}...", input);
    let graph = load_graph(&input)?;

    let vertex_count = graph.vertex_count();
    let edge_count = graph.edge_count();

    // Calculate degree statistics
    let degrees: Vec<usize> = graph
        .vertices
        .values()
        .map(|v| v.degree())
        .collect();

    let total_degree: usize = degrees.iter().sum();
    let avg_degree = if vertex_count > 0 {
        total_degree as f64 / vertex_count as f64
    } else {
        0.0
    };

    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let min_degree = degrees.iter().copied().min().unwrap_or(0);

    // Calculate degree distribution
    let mut degree_counts: HashMap<usize, usize> = HashMap::new();
    for &deg in &degrees {
        *degree_counts.entry(deg).or_insert(0) += 1;
    }

    println!("\n=== Graph Statistics ===");
    println!("Vertices: {}", vertex_count);
    println!("Edges: {}", edge_count);
    println!("Directed: {}", graph.metadata.directed);
    println!("Weighted: {}", graph.metadata.weighted);
    println!("\n=== Degree Statistics ===");
    println!("Average degree: {:.2}", avg_degree);
    println!("Max degree: {}", max_degree);
    println!("Min degree: {}", min_degree);

    // Show top degree vertices
    if vertex_count > 0 {
        let mut vertex_degrees: Vec<_> = graph
            .vertices
            .iter()
            .map(|(id, v)| (id.0, v.degree()))
            .collect();
        vertex_degrees.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\n=== Top 5 Vertices by Degree ===");
        for (id, deg) in vertex_degrees.iter().take(5) {
            println!("Vertex {}: degree {}", id, deg);
        }
    }

    Ok(())
}

/// Load a graph from an edge list file
///
/// Format: Each line contains: source_id target_id [weight]
/// Lines starting with # are comments
fn load_graph(path: &PathBuf) -> Result<Graph> {
    let file = File::open(path).context("Failed to open input file")?;
    let reader = BufReader::new(file);

    let mut builder = GraphBuilder::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("graph")
    );

    let mut edge_id_counter = 0u64;
    let mut has_weights = false;
    let mut vertex_ids = std::collections::HashSet::new();

    // First pass: collect all vertex IDs
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let source = parts[0]
            .parse::<u64>()
            .context("Invalid source vertex ID")?;
        let target = parts[1]
            .parse::<u64>()
            .context("Invalid target vertex ID")?;

        vertex_ids.insert(source);
        vertex_ids.insert(target);

        if parts.len() >= 3 {
            has_weights = true;
        }
    }

    // Add all vertices
    for vid in vertex_ids {
        builder.add_vertex(VertexId(vid));
    }

    // Second pass: add edges
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    if has_weights {
        builder.weighted();
    }

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let source = VertexId(parts[0].parse()?);
        let target = VertexId(parts[1].parse()?);

        if has_weights && parts.len() >= 3 {
            let weight = parts[2].parse::<f64>().context("Invalid edge weight")?;
            builder.add_weighted_edge(source, target, weight);
        } else {
            builder.add_edge(source, target);
        }

        edge_id_counter += 1;
    }

    Ok(builder.build())
}

/// Export results with f64 values to CSV
fn export_f64_results(
    results: &HashMap<VertexId, VertexData>,
    field_name: &str,
    path: &PathBuf,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "vertex_id,{}", field_name)?;

    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by_key(|(id, _)| id.0);

    for (vertex_id, data) in sorted_results {
        let value = data.get_f64(field_name).unwrap_or(0.0);
        writeln!(file, "{},{:.6}", vertex_id.0, value)?;
    }

    Ok(())
}

/// Export results with u64 values to CSV
fn export_u64_results(
    results: &HashMap<VertexId, VertexData>,
    field_name: &str,
    path: &PathBuf,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "vertex_id,{}", field_name)?;

    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by_key(|(id, _)| id.0);

    for (vertex_id, data) in sorted_results {
        let value = data.get_u64(field_name).unwrap_or(0);
        writeln!(file, "{},{}", vertex_id.0, value)?;
    }

    Ok(())
}

/// Export a partition as an edge list
fn export_partition(graph: &Graph, path: &PathBuf) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# Vertices: {}", graph.vertex_count())?;
    writeln!(file, "# Edges: {}", graph.edge_count())?;

    for edge in graph.edges.values() {
        if graph.metadata.weighted {
            writeln!(
                file,
                "{} {} {:.6}",
                edge.source.0, edge.target.0, edge.data.weight
            )?;
        } else {
            writeln!(file, "{} {}", edge.source.0, edge.target.0)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_simple_graph() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "1 2").unwrap();
        writeln!(file, "2 3").unwrap();
        writeln!(file, "3 1").unwrap();
        file.flush().unwrap();

        let graph = load_graph(&file.path().to_path_buf()).unwrap();
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_load_weighted_graph() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "1 2 0.5").unwrap();
        writeln!(file, "2 3 1.5").unwrap();
        file.flush().unwrap();

        let graph = load_graph(&file.path().to_path_buf()).unwrap();
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.metadata.weighted);
    }

    #[test]
    fn test_load_graph_with_comments() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# This is a comment").unwrap();
        writeln!(file, "1 2").unwrap();
        writeln!(file, "# Another comment").unwrap();
        writeln!(file, "2 3").unwrap();
        file.flush().unwrap();

        let graph = load_graph(&file.path().to_path_buf()).unwrap();
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }
}
