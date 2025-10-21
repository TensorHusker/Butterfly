//! Benchmarks for graph partitioning algorithms
//!
//! Compares performance and quality of different partitioning strategies:
//! - HashPartitioner: Simple hash-based partitioning
//! - StreamPartitioner: Greedy streaming partitioning
//! - METISPartitioner: Multilevel partitioning (currently uses stream as fallback)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use butterfly_partition::{HashPartitioner, StreamPartitioner, METISPartitioner, GraphPartitioner};

mod graph_generators;
use graph_generators::*;

/// Benchmark partitioning time for different graph sizes and partitioners
fn partition_time_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_time");

    let sizes = [1000, 10000, 100000];
    let num_partitions = 4;

    for &size in &sizes {
        let graph = generate_power_law_graph(size);

        group.bench_with_input(
            BenchmarkId::new("hash", size),
            &graph,
            |b, g| {
                let partitioner = HashPartitioner::new();
                b.iter(|| partitioner.partition(black_box(g), num_partitions))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("stream", size),
            &graph,
            |b, g| {
                let partitioner = StreamPartitioner::new();
                b.iter(|| partitioner.partition(black_box(g), num_partitions))
            }
        );

        // Skip METIS for largest size to avoid long benchmark times
        if size <= 10000 {
            group.bench_with_input(
                BenchmarkId::new("metis", size),
                &graph,
                |b, g| {
                    let partitioner = METISPartitioner::new();
                    b.iter(|| partitioner.partition(black_box(g), num_partitions))
                }
            );
        }
    }

    group.finish();
}

/// Benchmark edge cut quality across different graph topologies
fn partition_quality_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_quality");

    let n = 5000;
    let num_partitions = 8;

    // Power-law graph (realistic)
    let power_law = generate_power_law_graph(n);
    group.bench_function("power_law_hash", |b| {
        let partitioner = HashPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&power_law), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    group.bench_function("power_law_stream", |b| {
        let partitioner = StreamPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&power_law), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    // Grid graph (structured)
    let grid = generate_grid_graph(71, 71); // ~5000 vertices
    group.bench_function("grid_hash", |b| {
        let partitioner = HashPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&grid), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    group.bench_function("grid_stream", |b| {
        let partitioner = StreamPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&grid), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    // Random graph
    let random = generate_random_graph(n, 0.001);
    group.bench_function("random_hash", |b| {
        let partitioner = HashPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&random), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    group.bench_function("random_stream", |b| {
        let partitioner = StreamPartitioner::new();
        b.iter(|| {
            let subgraphs = partitioner.partition(black_box(&random), num_partitions).unwrap();
            partitioner.estimate_quality(&subgraphs)
        })
    });

    group.finish();
}

/// Benchmark load balancing across different partition counts
fn partition_scalability_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_scalability");

    let graph = generate_power_law_graph(10000);
    let partition_counts = [2, 4, 8, 16, 32];

    for &num_partitions in &partition_counts {
        group.bench_with_input(
            BenchmarkId::new("hash", num_partitions),
            &num_partitions,
            |b, &np| {
                let partitioner = HashPartitioner::new();
                b.iter(|| partitioner.partition(black_box(&graph), np))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("stream", num_partitions),
            &num_partitions,
            |b, &np| {
                let partitioner = StreamPartitioner::new();
                b.iter(|| partitioner.partition(black_box(&graph), np))
            }
        );
    }

    group.finish();
}

/// Benchmark partitioning of different graph topologies
fn partition_topology_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_topology");

    let n = 5000;
    let num_partitions = 8;

    // Different graph topologies
    let topologies = [
        ("power_law", generate_power_law_graph(n)),
        ("random", generate_random_graph(n, 0.001)),
        ("grid", generate_grid_graph(71, 71)),
        ("cycle", generate_cycle_graph(n)),
        ("star", generate_star_graph(n)),
    ];

    for (name, graph) in &topologies {
        group.bench_with_input(
            BenchmarkId::new("hash", name),
            graph,
            |b, g| {
                let partitioner = HashPartitioner::new();
                b.iter(|| partitioner.partition(black_box(g), num_partitions))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("stream", name),
            graph,
            |b, g| {
                let partitioner = StreamPartitioner::new();
                b.iter(|| partitioner.partition(black_box(g), num_partitions))
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    partition_time_benchmarks,
    partition_quality_benchmarks,
    partition_scalability_benchmarks,
    partition_topology_benchmarks
);

criterion_main!(benches);
