# Butterfly Metrics Specification

This document provides a complete catalog of all Prometheus metrics exposed by Butterfly components.

## Naming Conventions

All metrics follow Prometheus naming best practices:

- Prefix: `butterfly_`
- Use snake_case
- Suffix with unit: `_seconds`, `_bytes`, `_total`, `_ratio`
- Counters end with `_total`
- No redundant prefixes in labels

## Label Cardinality Guidelines

| Label | Cardinality | Example Values |
|-------|-------------|----------------|
| `node_id` | ~1000 | coordinator-1, compute-node-42, observer-7 |
| `node_role` | 3 | coordinator, compute, observer |
| `layer_id` | ~200 | 0, 1, 2, ..., 199 |
| `model` | ~20 | llama-70b, gpt-4, claude-3 |
| `error_type` | ~50 | timeout, oom, network_error, ... |
| `operation` | ~30 | inference, checkpoint, consensus, ... |
| `protocol` | 2 | quic, grpc |
| `device` | ~10 | cpu, cuda:0, cuda:1, ... |

**Avoid high-cardinality labels:**
- ❌ `request_id` (unbounded)
- ❌ `user_id` (unbounded)
- ❌ `trace_id` (unbounded - use tracing instead)
- ❌ `timestamp` (time is series dimension)

## Metric Categories

### 1. Request Metrics (RED Method)

#### butterfly_requests_total
**Type:** Counter
**Description:** Total number of inference requests received
**Labels:**
- `node_role`: coordinator, compute, observer
- `method`: inference, checkpoint, health_check
- `status`: success, error

```promql
# Request rate by status
rate(butterfly_requests_total[5m])

# Error rate
rate(butterfly_requests_total{status="error"}[5m])
  / rate(butterfly_requests_total[5m])
```

#### butterfly_requests_in_flight
**Type:** Gauge
**Description:** Current number of in-flight requests
**Labels:**
- `node_role`: coordinator, compute
- `method`: inference, checkpoint

```promql
# P99 concurrent requests
quantile(0.99, butterfly_requests_in_flight)
```

#### butterfly_request_errors_total
**Type:** Counter
**Description:** Total number of request errors
**Labels:**
- `node_role`: coordinator, compute
- `method`: inference, checkpoint
- `error_type`: timeout, oom, network_error, byzantine_fault, ...

```promql
# Error rate by type
rate(butterfly_request_errors_total[5m]) by (error_type)
```

#### butterfly_request_duration_seconds
**Type:** Histogram
**Description:** Request duration from submission to completion
**Labels:**
- `node_role`: coordinator, compute
- `method`: inference, checkpoint

**Buckets:** [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, +Inf]

```promql
# P99 latency
histogram_quantile(0.99,
  rate(butterfly_request_duration_seconds_bucket[5m])
)

# SLO: 99% of requests < 5s
histogram_quantile(0.99,
  rate(butterfly_request_duration_seconds_bucket{le="5.0"}[5m])
) < 5.0
```

### 2. Consensus Layer Metrics

#### butterfly_consensus_term
**Type:** Gauge
**Description:** Current Raft consensus term
**Labels:**
- `node_id`: coordinator-1, coordinator-2, ...

```promql
# Detect term divergence (should all be same)
stddev(butterfly_consensus_term) > 0
```

#### butterfly_consensus_leader_changes_total
**Type:** Counter
**Description:** Total number of leader elections
**Labels:** None

```promql
# Frequent leader changes indicate instability
rate(butterfly_consensus_leader_changes_total[1h]) > 5
```

#### butterfly_consensus_election_duration_seconds
**Type:** Histogram
**Description:** Duration of leader elections
**Buckets:** [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, +Inf]

```promql
# P99 election time
histogram_quantile(0.99,
  rate(butterfly_consensus_election_duration_seconds_bucket[1h])
)
```

#### butterfly_consensus_log_entries_total
**Type:** Gauge
**Description:** Total number of entries in Raft log
**Labels:**
- `node_id`: coordinator-1, coordinator-2, ...

```promql
# Log divergence detection
max(butterfly_consensus_log_entries_total)
  - min(butterfly_consensus_log_entries_total) > 100
```

#### butterfly_consensus_commit_index
**Type:** Gauge
**Description:** Index of highest committed log entry
**Labels:**
- `node_id`: coordinator-1, coordinator-2, ...

```promql
# Commit lag
max(butterfly_consensus_log_entries_total)
  - min(butterfly_consensus_commit_index)
```

#### butterfly_consensus_applied_index
**Type:** Gauge
**Description:** Index of highest applied log entry
**Labels:**
- `node_id`: coordinator-1, coordinator-2, ...

```promql
# Apply lag (should be close to commit index)
butterfly_consensus_commit_index
  - butterfly_consensus_applied_index > 10
```

#### butterfly_consensus_heartbeat_failures_total
**Type:** Counter
**Description:** Failed heartbeats between nodes
**Labels:**
- `from_node`: coordinator-1, ...
- `to_node`: coordinator-2, ...

```promql
# Network partition detection
rate(butterfly_consensus_heartbeat_failures_total[5m]) > 0.1
```

#### butterfly_cluster_size
**Type:** Gauge
**Description:** Number of nodes in cluster
**Labels:**
- `state`: active, degraded, recovering

```promql
# Alert if cluster size drops below quorum
butterfly_cluster_size{state="active"} < 3
```

#### butterfly_quorum_size
**Type:** Gauge
**Description:** Minimum nodes needed for quorum (2f+1)
**Labels:** None

### 3. Computation Metrics

#### butterfly_layer_compute_duration_seconds
**Type:** Histogram
**Description:** Time to execute a model layer
**Labels:**
- `layer_id`: 0, 1, 2, ...
- `node_id`: compute-node-1, ...
- `model`: llama-70b, gpt-4, ...

**Buckets:** [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, +Inf]

```promql
# P99 layer execution time
histogram_quantile(0.99,
  rate(butterfly_layer_compute_duration_seconds_bucket[5m])
) by (layer_id)

# Slowest layers
topk(10,
  histogram_quantile(0.99,
    rate(butterfly_layer_compute_duration_seconds_bucket[5m])
  ) by (layer_id)
)
```

#### butterfly_layer_memory_bytes
**Type:** Gauge
**Description:** Memory used by layer computation
**Labels:**
- `layer_id`: 0, 1, 2, ...
- `node_id`: compute-node-1, ...
- `operation`: forward, activation, gradient

```promql
# Total memory by layer
sum(butterfly_layer_memory_bytes) by (layer_id)

# Memory per node
sum(butterfly_layer_memory_bytes) by (node_id)
```

#### butterfly_tensor_size_bytes
**Type:** Histogram
**Description:** Size of tensors passed between layers
**Labels:**
- `layer_id`: 0, 1, 2, ...
- `direction`: input, output

**Buckets:** [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, +Inf]  (1KB to 1GB)

```promql
# Average tensor size
histogram_quantile(0.5,
  rate(butterfly_tensor_size_bytes_bucket[5m])
) by (layer_id)
```

#### butterfly_pipeline_depth
**Type:** Gauge
**Description:** Number of stages in execution pipeline
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Average pipeline depth
avg(butterfly_pipeline_depth)
```

#### butterfly_pipeline_stalls_total
**Type:** Counter
**Description:** Number of pipeline stalls (waiting for dependencies)
**Labels:**
- `node_id`: compute-node-1, ...
- `reason`: memory_pressure, network_wait, consensus_wait

```promql
# Stall rate
rate(butterfly_pipeline_stalls_total[5m]) by (reason)
```

#### butterfly_batch_size
**Type:** Gauge
**Description:** Current batch size for inference
**Labels:**
- `node_id`: compute-node-1, ...

#### butterfly_tokens_processed_total
**Type:** Counter
**Description:** Total tokens processed by the system
**Labels:**
- `model`: llama-70b, gpt-4, ...
- `node_id`: compute-node-1, ...

```promql
# Cluster-wide throughput (tokens/sec)
sum(rate(butterfly_tokens_processed_total[1m]))

# Per-model throughput
sum(rate(butterfly_tokens_processed_total[1m])) by (model)
```

#### butterfly_sequences_processed_total
**Type:** Counter
**Description:** Total sequences (requests) processed
**Labels:**
- `model`: llama-70b, ...
- `node_id`: compute-node-1, ...

```promql
# Sequences per second
sum(rate(butterfly_sequences_processed_total[1m]))
```

#### butterfly_attention_heads_active
**Type:** Gauge
**Description:** Number of active attention heads in layer
**Labels:**
- `layer_id`: 0, 1, 2, ...
- `node_id`: compute-node-1, ...

### 4. Communication Metrics

#### butterfly_network_bytes_sent
**Type:** Counter
**Description:** Total bytes sent over network
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...
- `protocol`: quic, grpc

```promql
# Network throughput (bytes/sec)
sum(rate(butterfly_network_bytes_sent[5m]))

# Bandwidth per connection
rate(butterfly_network_bytes_sent[5m]) by (from_node, to_node)
```

#### butterfly_network_bytes_received
**Type:** Counter
**Description:** Total bytes received over network
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...
- `protocol`: quic, grpc

```promql
# Total cluster network traffic
sum(rate(butterfly_network_bytes_sent[5m]))
  + sum(rate(butterfly_network_bytes_received[5m]))
```

#### butterfly_network_messages_sent_total
**Type:** Counter
**Description:** Total messages sent
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...
- `message_type`: tensor, checkpoint, consensus, heartbeat

```promql
# Message rate by type
sum(rate(butterfly_network_messages_sent_total[5m])) by (message_type)
```

#### butterfly_network_messages_received_total
**Type:** Counter
**Description:** Total messages received
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...
- `message_type`: tensor, checkpoint, consensus, heartbeat

#### butterfly_connection_errors_total
**Type:** Counter
**Description:** Connection errors
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...
- `error_type`: timeout, refused, reset, encryption_error

```promql
# Connection error rate
rate(butterfly_connection_errors_total[5m]) by (error_type)
```

#### butterfly_connection_latency_seconds
**Type:** Histogram
**Description:** Round-trip latency between nodes
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...

**Buckets:** [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, +Inf]

```promql
# P99 latency matrix
histogram_quantile(0.99,
  rate(butterfly_connection_latency_seconds_bucket[5m])
) by (from_node, to_node)
```

#### butterfly_connection_active
**Type:** Gauge
**Description:** Whether connection is active (1) or not (0)
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...

```promql
# Detect network partition
butterfly_connection_active == 0
```

#### butterfly_quic_streams_active
**Type:** Gauge
**Description:** Number of active QUIC streams
**Labels:**
- `node_id`: compute-node-1, ...

#### butterfly_quic_congestion_window_bytes
**Type:** Gauge
**Description:** QUIC congestion window size
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Detect congestion
rate(butterfly_quic_congestion_window_bytes[5m]) < 0
```

#### butterfly_quic_rtt_seconds
**Type:** Gauge
**Description:** QUIC round-trip time
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...

```promql
# RTT over time
butterfly_quic_rtt_seconds
```

#### butterfly_quic_packet_loss_ratio
**Type:** Gauge
**Description:** Ratio of lost packets (0.0 - 1.0)
**Labels:**
- `from_node`: compute-node-1, ...
- `to_node`: compute-node-2, ...

```promql
# Alert on high packet loss
butterfly_quic_packet_loss_ratio > 0.05
```

### 5. Resource Utilization Metrics

#### butterfly_memory_allocated_bytes
**Type:** Gauge
**Description:** Memory currently allocated
**Labels:**
- `node_id`: compute-node-1, ...
- `pool`: model, tensor, buffer, system

```promql
# Memory usage per node
sum(butterfly_memory_allocated_bytes) by (node_id)

# Memory usage by pool
sum(butterfly_memory_allocated_bytes) by (pool)
```

#### butterfly_memory_peak_bytes
**Type:** Gauge
**Description:** Peak memory usage since start
**Labels:**
- `node_id`: compute-node-1, ...

#### butterfly_memory_fragmentation_ratio
**Type:** Gauge
**Description:** Memory fragmentation (1.0 = no fragmentation)
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Alert on high fragmentation
butterfly_memory_fragmentation_ratio < 0.7
```

#### butterfly_cpu_usage_ratio
**Type:** Gauge
**Description:** CPU usage ratio (0.0 - 1.0)
**Labels:**
- `node_id`: compute-node-1, ...
- `core`: 0, 1, 2, ... (optional, can be aggregated)

```promql
# Average CPU usage
avg(butterfly_cpu_usage_ratio)

# Per-node CPU usage
butterfly_cpu_usage_ratio by (node_id)
```

#### butterfly_cpu_throttle_events_total
**Type:** Counter
**Description:** Number of CPU throttle events
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Throttle rate
rate(butterfly_cpu_throttle_events_total[5m])
```

#### butterfly_gpu_utilization_ratio
**Type:** Gauge
**Description:** GPU utilization (0.0 - 1.0)
**Labels:**
- `node_id`: compute-node-1, ...
- `gpu_id`: 0, 1, 2, ...

```promql
# Average GPU utilization
avg(butterfly_gpu_utilization_ratio)

# Underutilized GPUs
butterfly_gpu_utilization_ratio < 0.5
```

#### butterfly_gpu_memory_bytes
**Type:** Gauge
**Description:** GPU memory usage
**Labels:**
- `node_id`: compute-node-1, ...
- `gpu_id`: 0, 1, 2, ...
- `state`: used, free

```promql
# GPU memory usage ratio
butterfly_gpu_memory_bytes{state="used"}
  / (butterfly_gpu_memory_bytes{state="used"} + butterfly_gpu_memory_bytes{state="free"})
```

#### butterfly_gpu_temperature_celsius
**Type:** Gauge
**Description:** GPU temperature in Celsius
**Labels:**
- `node_id`: compute-node-1, ...
- `gpu_id`: 0, 1, 2, ...

```promql
# Alert on high GPU temperature
butterfly_gpu_temperature_celsius > 85
```

#### butterfly_gpu_compute_unit_active
**Type:** Gauge
**Description:** Number of active GPU compute units
**Labels:**
- `node_id`: compute-node-1, ...
- `gpu_id`: 0, 1, 2, ...
- `unit_type`: sm, tensor_core

#### butterfly_disk_checkpoint_bytes
**Type:** Gauge
**Description:** Disk space used by checkpoints
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Total checkpoint storage
sum(butterfly_disk_checkpoint_bytes)
```

#### butterfly_disk_write_duration_seconds
**Type:** Histogram
**Description:** Disk write duration
**Labels:**
- `node_id`: compute-node-1, ...
- `operation`: checkpoint_write, log_append

**Buckets:** [0.001, 0.01, 0.1, 1.0, 10.0, +Inf]

```promql
# P99 write latency
histogram_quantile(0.99,
  rate(butterfly_disk_write_duration_seconds_bucket[5m])
)
```

#### butterfly_disk_io_errors_total
**Type:** Counter
**Description:** Disk I/O errors
**Labels:**
- `node_id`: compute-node-1, ...
- `operation`: read, write

```promql
# Disk error rate
rate(butterfly_disk_io_errors_total[5m])
```

### 6. Checkpoint & Recovery Metrics

#### butterfly_checkpoint_count
**Type:** Gauge
**Description:** Number of checkpoints
**Labels:**
- `node_id`: compute-node-1, ...
- `status`: pending, complete, failed

```promql
# Pending checkpoints
butterfly_checkpoint_count{status="pending"}
```

#### butterfly_checkpoint_duration_seconds
**Type:** Histogram
**Description:** Checkpoint creation duration
**Labels:**
- `node_id`: compute-node-1, ...
- `phase`: snapshot, write, verify

**Buckets:** [0.1, 1.0, 10.0, 60.0, 300.0, +Inf]

```promql
# P99 checkpoint time
histogram_quantile(0.99,
  rate(butterfly_checkpoint_duration_seconds_bucket[5m])
) by (phase)
```

#### butterfly_checkpoint_size_bytes
**Type:** Histogram
**Description:** Checkpoint size
**Labels:**
- `node_id`: compute-node-1, ...
- `checkpoint_id`: ckpt_001, ckpt_002, ...

**Buckets:** [1e6, 1e7, 1e8, 1e9, 1e10, +Inf]  (1MB to 10GB)

```promql
# Average checkpoint size
histogram_quantile(0.5,
  rate(butterfly_checkpoint_size_bytes_bucket[5m])
)
```

#### butterfly_recovery_attempts_total
**Type:** Counter
**Description:** Recovery attempts
**Labels:**
- `node_id`: compute-node-1, ...
- `outcome`: success, partial, failed

```promql
# Recovery success rate
rate(butterfly_recovery_attempts_total{outcome="success"}[1h])
  / rate(butterfly_recovery_attempts_total[1h])
```

#### butterfly_recovery_duration_seconds
**Type:** Histogram
**Description:** Recovery duration
**Labels:**
- `node_id`: compute-node-1, ...

**Buckets:** [1.0, 10.0, 60.0, 300.0, 600.0, +Inf]

```promql
# P99 recovery time
histogram_quantile(0.99,
  rate(butterfly_recovery_duration_seconds_bucket[1h])
)
```

#### butterfly_state_divergence_detected_total
**Type:** Counter
**Description:** State divergences detected between nodes
**Labels:**
- `node_id_pair`: node1_node2

```promql
# Divergence detection rate (should be near 0)
rate(butterfly_state_divergence_detected_total[5m]) > 0
```

### 7. Coordination Metrics

#### butterfly_tasks_queued
**Type:** Gauge
**Description:** Tasks in queue waiting for execution
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Queue depth
butterfly_tasks_queued

# Alert on queue buildup
butterfly_tasks_queued > 100
```

#### butterfly_tasks_assigned_total
**Type:** Counter
**Description:** Tasks assigned to nodes
**Labels:**
- `from_coordinator`: coordinator-1, ...
- `to_node`: compute-node-1, ...

```promql
# Task assignment rate
rate(butterfly_tasks_assigned_total[5m])
```

#### butterfly_tasks_completed_total
**Type:** Counter
**Description:** Tasks completed
**Labels:**
- `node_id`: compute-node-1, ...
- `status`: success, error, timeout

```promql
# Task completion rate
rate(butterfly_tasks_completed_total{status="success"}[5m])

# Task error rate
rate(butterfly_tasks_completed_total{status="error"}[5m])
  / rate(butterfly_tasks_completed_total[5m])
```

#### butterfly_task_reassignments_total
**Type:** Counter
**Description:** Tasks reassigned due to failures
**Labels:**
- `reason`: node_failure, timeout, overload

```promql
# Reassignment rate
rate(butterfly_task_reassignments_total[5m]) by (reason)
```

#### butterfly_barrier_wait_duration_seconds
**Type:** Histogram
**Description:** Time waiting at synchronization barriers
**Labels:**
- `barrier_id`: barrier_1, barrier_2, ...
- `node_id`: compute-node-1, ...

**Buckets:** [0.001, 0.01, 0.1, 1.0, 10.0, +Inf]

```promql
# P99 barrier wait time
histogram_quantile(0.99,
  rate(butterfly_barrier_wait_duration_seconds_bucket[5m])
)
```

#### butterfly_synchronization_skew_seconds
**Type:** Gauge
**Description:** Clock skew between node pair
**Labels:**
- `node_pair`: node1_node2

```promql
# Max clock skew
max(butterfly_synchronization_skew_seconds)
```

#### butterfly_dependency_wait_duration_seconds
**Type:** Histogram
**Description:** Time waiting for dependencies
**Labels:**
- `task_id`: task_001, task_002, ...
- `node_id`: compute-node-1, ...

**Buckets:** [0.001, 0.01, 0.1, 1.0, 10.0, +Inf]

### 8. Byzantine Fault Detection Metrics

#### butterfly_byzantine_faults_detected_total
**Type:** Counter
**Description:** Byzantine faults detected
**Labels:**
- `node_id`: compute-node-1, ...
- `fault_type`: incorrect_result, invalid_signature, merkle_mismatch

```promql
# Fault detection rate
rate(butterfly_byzantine_faults_detected_total[1h]) by (fault_type)
```

#### butterfly_verification_checks_total
**Type:** Counter
**Description:** Verification checks performed
**Labels:**
- `node_id`: compute-node-1, ...
- `check_type`: signature, merkle_tree, result_comparison

```promql
# Verification rate
rate(butterfly_verification_checks_total[5m]) by (check_type)
```

#### butterfly_verification_failures_total
**Type:** Counter
**Description:** Failed verification checks
**Labels:**
- `node_id`: compute-node-1, ...
- `failure_type`: signature_invalid, merkle_mismatch, result_divergence

```promql
# Verification failure rate
rate(butterfly_verification_failures_total[5m])
  / rate(butterfly_verification_checks_total[5m])
```

#### butterfly_reputation_score
**Type:** Gauge
**Description:** Reputation score of node (0.0 - 1.0)
**Labels:**
- `node_id`: compute-node-1, ...

```promql
# Low reputation nodes
butterfly_reputation_score < 0.5
```

#### butterfly_cryptographic_failures_total
**Type:** Counter
**Description:** Cryptographic operation failures
**Labels:**
- `node_id`: compute-node-1, ...
- `operation`: sign, verify, encrypt, decrypt

```promql
# Crypto failure rate
rate(butterfly_cryptographic_failures_total[5m])
```

#### butterfly_merkle_root_mismatches_total
**Type:** Counter
**Description:** Merkle root mismatches between nodes
**Labels:**
- `task_id`: task_001, ...

```promql
# Mismatch rate (should be near 0)
rate(butterfly_merkle_root_mismatches_total[5m]) > 0
```

## Recording Rules

Pre-compute common aggregations to reduce query load:

```yaml
# /docs/prometheus_rules.yml
groups:
  - name: butterfly_aggregations
    interval: 1m
    rules:
      # Request rate
      - record: butterfly:request_rate:1m
        expr: rate(butterfly_requests_total[1m])

      # Error rate
      - record: butterfly:error_rate:1m
        expr: rate(butterfly_request_errors_total[1m])

      # P99 latency
      - record: butterfly:p99_latency:1m
        expr: |
          histogram_quantile(0.99,
            rate(butterfly_request_duration_seconds_bucket[1m])
          )

      # P50 latency
      - record: butterfly:p50_latency:1m
        expr: |
          histogram_quantile(0.50,
            rate(butterfly_request_duration_seconds_bucket[1m])
          )

      # Cluster throughput (tokens/sec)
      - record: butterfly:cluster_throughput:1m
        expr: sum(rate(butterfly_tokens_processed_total[1m]))

      # Network bandwidth (bytes/sec)
      - record: butterfly:network_bandwidth:1m
        expr: |
          sum(rate(butterfly_network_bytes_sent[1m]))
          + sum(rate(butterfly_network_bytes_received[1m]))

      # Average CPU usage
      - record: butterfly:cpu_usage_avg:1m
        expr: avg(butterfly_cpu_usage_ratio)

      # Average GPU utilization
      - record: butterfly:gpu_utilization_avg:1m
        expr: avg(butterfly_gpu_utilization_ratio)

      # Memory usage per node
      - record: butterfly:memory_used_bytes:1m
        expr: sum(butterfly_memory_allocated_bytes) by (node_id)

      # Task completion rate
      - record: butterfly:task_completion_rate:1m
        expr: rate(butterfly_tasks_completed_total{status="success"}[1m])

      # Byzantine fault rate
      - record: butterfly:byzantine_fault_rate:1m
        expr: rate(butterfly_byzantine_faults_detected_total[1m])
```

## Alerting Rules

```yaml
# /docs/prometheus_alerts.yml
groups:
  - name: butterfly_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: butterfly:error_rate:1m > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: butterfly:p99_latency:1m > 5.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 5s)"

      # Cluster size below quorum
      - alert: ClusterBelowQuorum
        expr: butterfly_cluster_size{state="active"} < butterfly_quorum_size
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Cluster size below quorum"
          description: "Only {{ $value }} active nodes (need {{ $labels.butterfly_quorum_size }})"

      # Frequent leader changes
      - alert: FrequentLeaderChanges
        expr: rate(butterfly_consensus_leader_changes_total[1h]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Frequent leader changes"
          description: "{{ $value }} leader changes per hour"

      # Byzantine fault detected
      - alert: ByzantineFaultDetected
        expr: butterfly:byzantine_fault_rate:1m > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Byzantine fault detected"
          description: "Fault rate: {{ $value }}/sec"

      # High memory usage
      - alert: HighMemoryUsage
        expr: |
          butterfly:memory_used_bytes:1m
          / on(node_id) (butterfly_memory_allocated_bytes + butterfly_memory_peak_bytes)
          > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.node_id }}"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      # GPU overheating
      - alert: GPUOverheating
        expr: butterfly_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU overheating on {{ $labels.node_id }}"
          description: "GPU {{ $labels.gpu_id }} temperature is {{ $value }}°C"

      # Network partition
      - alert: NetworkPartition
        expr: butterfly_connection_active == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Network partition detected"
          description: "Connection from {{ $labels.from_node }} to {{ $labels.to_node }} is down"

      # Queue buildup
      - alert: QueueBuildup
        expr: butterfly_tasks_queued > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Task queue buildup on {{ $labels.node_id }}"
          description: "{{ $value }} tasks queued"
```

## Prometheus Configuration

```yaml
# /docs/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'butterfly-prod'
    environment: 'production'

scrape_configs:
  # Coordinators
  - job_name: 'butterfly-coordinators'
    static_configs:
      - targets:
          - 'coordinator-1:9090'
          - 'coordinator-2:9090'
          - 'coordinator-3:9090'
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - target_label: node_role
        replacement: 'coordinator'

  # Compute nodes (DNS service discovery)
  - job_name: 'butterfly-compute-nodes'
    dns_sd_configs:
      - names:
          - '_butterfly-compute._tcp.butterfly.local'
    relabel_configs:
      - source_labels: [__meta_dns_name]
        target_label: node_id
      - target_label: node_role
        replacement: 'compute'

  # Observers (Kubernetes service discovery)
  - job_name: 'butterfly-observers'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['butterfly']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_role]
        action: keep
        regex: observer
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: node_id
      - target_label: node_role
        replacement: 'observer'

rule_files:
  - 'prometheus_rules.yml'
  - 'prometheus_alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'

# Remote write for long-term storage
remote_write:
  - url: 'http://thanos-receive:19291/api/v1/receive'
    queue_config:
      max_samples_per_send: 10000
      max_shards: 30
      capacity: 50000
```

## Usage Examples

### Finding Bottlenecks

```promql
# Which layer is slowest?
topk(5,
  histogram_quantile(0.99,
    rate(butterfly_layer_compute_duration_seconds_bucket[5m])
  ) by (layer_id)
)

# Which node has highest latency to coordinator?
topk(5,
  histogram_quantile(0.99,
    rate(butterfly_connection_latency_seconds_bucket{to_node=~"coordinator.*"}[5m])
  ) by (from_node)
)

# What's consuming most memory?
topk(5,
  sum(butterfly_memory_allocated_bytes) by (pool)
)
```

### Capacity Planning

```promql
# Current cluster utilization
avg(butterfly_cpu_usage_ratio)
avg(butterfly_gpu_utilization_ratio)
avg(butterfly_memory_allocated_bytes) / avg(butterfly_memory_peak_bytes)

# Requests per second capacity
sum(rate(butterfly_requests_total[1h]))

# Tokens per second throughput
sum(rate(butterfly_tokens_processed_total[1h]))
```

### SLO Monitoring

```promql
# Availability: % of successful requests
sum(rate(butterfly_requests_total{status="success"}[7d]))
  / sum(rate(butterfly_requests_total[7d]))
> 0.999  # 99.9% availability SLO

# Latency: P99 < 5s
histogram_quantile(0.99,
  rate(butterfly_request_duration_seconds_bucket[7d])
) < 5.0

# Throughput: > 1000 tokens/sec
sum(rate(butterfly_tokens_processed_total[7d])) > 1000
```

## Next Steps

1. Implement metrics in butterfly-metrics crate
2. Add Prometheus client to each component
3. Create Grafana dashboards (see /docs/dashboards/)
4. Set up alerting rules
5. Load test to validate cardinality limits
