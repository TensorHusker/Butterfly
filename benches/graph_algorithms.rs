//! Benchmarks for distributed graph algorithms
//!
//! Measures performance of graph algorithms in both single-partition and
//! multi-partition configurations. Tracks convergence speed and throughput.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use butterfly_core::{Graph, VertexId, VertexData};
use butterfly_graph::{PageRank, ConnectedComponents, ShortestPaths};
use butterfly_partition::{HashPartitioner, StreamPartitioner, GraphPartitioner};

mod graph_generators;
use graph_generators::*;

/// Simulate BSP execution for single partition
///
/// This runs the algorithm locally without actual distribution,
/// measuring the pure computational cost.
fn run_pagerank_single_partition(graph: &Graph, max_iterations: u32) -> usize {
    use butterfly_core::VertexCompute;

    let pagerank = PageRank::with_defaults();
    let mut vertices: Vec<_> = graph.vertices.values().cloned().collect();

    // Initialize all vertices
    let mut pr = PageRank::with_defaults();
    for vertex in &mut vertices {
        pr.init(vertex).unwrap();
    }

    let mut iteration = 0;
    let mut active_count = vertices.len();

    // Run supersteps
    while active_count > 0 && iteration < max_iterations {
        let mut messages = std::collections::HashMap::new();

        // Compute phase
        for vertex in &vertices {
            if !vertex.is_active() {
                continue;
            }

            let vertex_messages = messages.get(&vertex.id).map(|v: &Vec<_>| v.as_slice()).unwrap_or(&[]);
            let result = pagerank.compute(vertex, vertex_messages, iteration).unwrap();

            // Route messages
            for msg in result.outgoing_messages {
                messages.entry(msg.target_id)
                    .or_insert_with(Vec::new)
                    .push(msg);
            }
        }

        // Update phase
        active_count = 0;
        for vertex in &mut vertices {
            let vertex_messages = messages.get(&vertex.id).map(|v: &Vec<_>| v.as_slice()).unwrap_or(&[]);
            if !vertex_messages.is_empty() || vertex.is_active() {
                let result = pagerank.compute(vertex, vertex_messages, iteration).unwrap();
                vertex.data = result.new_value;
                if result.is_active {
                    vertex.activate();
                    active_count += 1;
                } else {
                    vertex.deactivate();
                }
            }
        }

        iteration += 1;
    }

    iteration as usize
}

/// Simulate BSP execution for connected components
fn run_connected_components_single_partition(graph: &Graph, max_iterations: u32) -> usize {
    use butterfly_core::VertexCompute;

    let mut cc = ConnectedComponents::new();
    let mut vertices: Vec<_> = graph.vertices.values().cloned().collect();

    // Initialize
    for vertex in &mut vertices {
        cc.init(vertex).unwrap();
    }

    let mut iteration = 0;
    let mut active_count = vertices.len();

    while active_count > 0 && iteration < max_iterations {
        let mut messages = std::collections::HashMap::new();

        // Compute phase
        for vertex in &vertices {
            if !vertex.is_active() {
                continue;
            }

            let vertex_messages = messages.get(&vertex.id).map(|v: &Vec<_>| v.as_slice()).unwrap_or(&[]);
            let result = cc.compute(vertex, vertex_messages, iteration).unwrap();

            for msg in result.outgoing_messages {
                messages.entry(msg.target_id)
                    .or_insert_with(Vec::new)
                    .push(msg);
            }
        }

        // Update phase
        active_count = 0;
        for vertex in &mut vertices {
            let vertex_messages = messages.get(&vertex.id).map(|v: &Vec<_>| v.as_slice()).unwrap_or(&[]);
            if !vertex_messages.is_empty() || vertex.is_active() {
                let result = cc.compute(vertex, vertex_messages, iteration).unwrap();
                vertex.data = result.new_value;

                // Check if component changed
                let old_component = vertex.data.get_u64("component").unwrap_or(vertex.id.0);
                let new_component = result.new_value.get_u64("component").unwrap_or(vertex.id.0);

                if new_component < old_component {
                    vertex.activate();
                    active_count += 1;
                } else {
                    vertex.deactivate();
                }
            }
        }

        iteration += 1;
    }

    iteration as usize
}

/// Benchmark PageRank on different graph sizes
fn pagerank_single_partition_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank_single_partition");

    let sizes = [1000, 5000, 10000];

    for &size in &sizes {
        let graph = generate_power_law_graph(size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &graph,
            |b, g| {
                b.iter(|| run_pagerank_single_partition(black_box(g), 100))
            }
        );
    }

    group.finish();
}

/// Benchmark PageRank with different graph topologies
fn pagerank_topology_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank_topology");

    let n = 5000;

    let topologies = [
        ("power_law", generate_power_law_graph(n)),
        ("random", generate_random_graph(n, 0.001)),
        ("grid", generate_grid_graph(71, 71)),
        ("star", generate_star_graph(n)),
    ];

    for (name, graph) in &topologies {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            graph,
            |b, g| {
                b.iter(|| run_pagerank_single_partition(black_box(g), 50))
            }
        );
    }

    group.finish();
}

/// Benchmark Connected Components on different graph sizes
fn connected_components_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");

    let sizes = [1000, 5000, 10000];

    for &size in &sizes {
        // Use random graph with low edge probability to create multiple components
        let graph = generate_random_graph(size, 0.001);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &graph,
            |b, g| {
                b.iter(|| run_connected_components_single_partition(black_box(g), 100))
            }
        );
    }

    group.finish();
}

/// Benchmark Connected Components on different topologies
fn connected_components_topology_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components_topology");

    let n = 5000;

    let topologies = [
        ("power_law", generate_power_law_graph(n)),
        ("random_sparse", generate_random_graph(n, 0.0005)),
        ("random_dense", generate_random_graph(n, 0.002)),
        ("grid", generate_grid_graph(71, 71)),
        ("cycle", generate_cycle_graph(n)),
    ];

    for (name, graph) in &topologies {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            graph,
            |b, g| {
                b.iter(|| run_connected_components_single_partition(black_box(g), 100))
            }
        );
    }

    group.finish();
}

/// Benchmark convergence rate (iterations to convergence)
fn convergence_rate_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_rate");

    let graph = generate_power_law_graph(5000);

    group.bench_function("pagerank_convergence", |b| {
        b.iter(|| {
            let iterations = run_pagerank_single_partition(black_box(&graph), 100);
            black_box(iterations)
        })
    });

    group.bench_function("connected_components_convergence", |b| {
        b.iter(|| {
            let iterations = run_connected_components_single_partition(black_box(&graph), 100);
            black_box(iterations)
        })
    });

    group.finish();
}

/// Benchmark throughput (edges processed per second)
fn throughput_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let sizes = [1000, 5000, 10000];

    for &size in &sizes {
        let graph = generate_power_law_graph(size);
        let edge_count = graph.edge_count();

        group.bench_with_input(
            BenchmarkId::new("pagerank", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let iterations = run_pagerank_single_partition(black_box(g), 50);
                    // Throughput = edges * iterations
                    black_box(edge_count * iterations)
                })
            }
        );

        group.bench_with_input(
            BenchmarkId::new("connected_components", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let iterations = run_connected_components_single_partition(black_box(g), 50);
                    black_box(edge_count * iterations)
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    pagerank_single_partition_benchmarks,
    pagerank_topology_benchmarks,
    connected_components_benchmarks,
    connected_components_topology_benchmarks,
    convergence_rate_benchmarks,
    throughput_benchmarks
);

criterion_main!(benches);
