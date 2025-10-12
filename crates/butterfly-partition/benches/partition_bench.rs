//! Benchmarks for partition algorithms

use butterfly_partition::*;
use butterfly_partition::cost_estimation::*;
use butterfly_core::{LayerType, NodeId};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn create_benchmark_nodes(count: usize) -> Vec<NodeCapability> {
    (0..count)
        .map(|i| NodeCapability {
            node_id: NodeId(i as u64),
            compute_power: 1.0 + (i as f64 * 0.1),
            memory_gb: 16.0 + (i as f64 * 4.0),
            bandwidth_mbps: 1000.0,
        })
        .collect()
}

fn create_gpt2_like_model(num_layers: usize) -> Vec<butterfly_core::LayerInfo> {
    let mut layers = Vec::new();

    // Embedding layer
    layers.push(create_layer_info(
        0,
        LayerType::Embedding {
            vocab_size: 50257,
            hidden_dim: 768,
        },
        DEFAULT_SEQ_LEN,
    ));

    // Transformer blocks
    for i in 1..=num_layers {
        layers.push(create_layer_info(
            i,
            LayerType::TransformerBlock {
                hidden_dim: 768,
                num_heads: 12,
                ff_dim: 3072,
            },
            DEFAULT_SEQ_LEN,
        ));
    }

    // Output head
    layers.push(create_layer_info(
        num_layers + 1,
        LayerType::OutputHead {
            hidden_dim: 768,
            vocab_size: 50257,
        },
        DEFAULT_SEQ_LEN,
    ));

    layers
}

fn bench_uniform_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_partitioning");

    for num_nodes in [2, 4, 8, 16].iter() {
        for num_layers in [12, 24, 48, 96].iter() {
            group.throughput(Throughput::Elements(*num_layers as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("nodes_{}_layers_{}", num_nodes, num_layers)),
                &(num_nodes, num_layers),
                |b, &(nodes, layers)| {
                    let node_caps = create_benchmark_nodes(*nodes);
                    let model_layers = create_gpt2_like_model(*layers);
                    let partitioner = UniformPartitioner;

                    b.iter(|| {
                        let partitions = partitioner
                            .partition(black_box(&model_layers), black_box(&node_caps))
                            .unwrap();
                        black_box(partitions)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_load_balanced_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_balanced_partitioning");

    for num_nodes in [2, 4, 8].iter() {
        for num_layers in [12, 24, 48].iter() {
            group.throughput(Throughput::Elements(*num_layers as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("nodes_{}_layers_{}", num_nodes, num_layers)),
                &(num_nodes, num_layers),
                |b, &(nodes, layers)| {
                    let node_caps = create_benchmark_nodes(*nodes);
                    let model_layers = create_gpt2_like_model(*layers);
                    let partitioner = LoadBalancedPartitioner;

                    b.iter(|| {
                        let partitions = partitioner
                            .partition(black_box(&model_layers), black_box(&node_caps))
                            .unwrap();
                        black_box(partitions)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_topology_aware_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_aware_partitioning");
    group.sample_size(20); // Fewer samples due to simulated annealing

    for num_nodes in [2, 4].iter() {
        for num_layers in [12, 24].iter() {
            group.throughput(Throughput::Elements(*num_layers as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("nodes_{}_layers_{}", num_nodes, num_layers)),
                &(num_nodes, num_layers),
                |b, &(nodes, layers)| {
                    let node_caps = create_benchmark_nodes(*nodes);
                    let model_layers = create_gpt2_like_model(*layers);
                    let topology = NetworkTopology::uniform(*nodes, 1000.0, 1.0);
                    let partitioner = TopologyAwarePartitioner::new(topology);

                    b.iter(|| {
                        let partitions = partitioner
                            .partition(black_box(&model_layers), black_box(&node_caps))
                            .unwrap();
                        black_box(partitions)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_cost_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cost_estimation");

    let layer_types = vec![
        LayerType::Embedding {
            vocab_size: 50257,
            hidden_dim: 768,
        },
        LayerType::TransformerBlock {
            hidden_dim: 768,
            num_heads: 12,
            ff_dim: 3072,
        },
        LayerType::OutputHead {
            hidden_dim: 768,
            vocab_size: 50257,
        },
    ];

    for (idx, layer_type) in layer_types.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::from_parameter(idx),
            layer_type,
            |b, layer_type| {
                b.iter(|| {
                    let cost = estimate_layer_cost(black_box(layer_type), DEFAULT_SEQ_LEN);
                    let memory = estimate_layer_memory(black_box(layer_type));
                    black_box((cost, memory))
                });
            },
        );
    }

    group.finish();
}

fn bench_quality_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_estimation");

    let nodes = create_benchmark_nodes(4);
    let layers = create_gpt2_like_model(24);
    let partitioner = LoadBalancedPartitioner;
    let partitions = partitioner.partition(&layers, &nodes).unwrap();

    group.bench_function("estimate_quality", |b| {
        b.iter(|| {
            let quality =
                partitioner.estimate_quality(black_box(&partitions), black_box(&layers), black_box(&nodes));
            black_box(quality)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uniform_partitioning,
    bench_load_balanced_partitioning,
    bench_topology_aware_partitioning,
    bench_cost_estimation,
    bench_quality_estimation
);

criterion_main!(benches);
