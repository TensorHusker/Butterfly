# Butterfly Debugging Guide

## Overview

This guide provides comprehensive debugging procedures for common issues in the Butterfly distributed inference system. Use the debug console, observability stack, and specialized tools to diagnose and resolve problems.

## Debug Console

### Starting the Debug Console

```bash
# Connect to coordinator
butterfly-cli debug --coordinator coordinator-1:8080

# Connect with authentication
butterfly-cli debug --coordinator coordinator-1:8080 \
  --cert /path/to/client.crt \
  --key /path/to/client.key
```

### Command Reference

#### Cluster Commands

```bash
# Show cluster status
butterfly> cluster status

Output:
Cluster State: HEALTHY
Active Nodes: 10
Quorum Size: 3
Current Term: 42
Leader: coordinator-1

# Show cluster topology
butterfly> cluster topology

Output:
Coordinator Nodes:
  coordinator-1 (leader)
  coordinator-2
  coordinator-3

Compute Nodes:
  compute-node-1 [layers: 0-7]
  compute-node-2 [layers: 8-15]
  ...

Observer Nodes:
  observer-1
  observer-2

# Show cluster health
butterfly> cluster health

Output:
Overall: HEALTHY

Components:
  ✓ Consensus: HEALTHY
  ✓ Network: HEALTHY
  ⚠ Resources: DEGRADED (node-7 high memory)
  ✓ Byzantine: HEALTHY
```

#### Node Commands

```bash
# Inspect specific node
butterfly> node inspect compute-node-7

Output:
Node ID: compute-node-7
Role: compute
State: COMPUTING
Uptime: 2h 34m 12s

Resources:
  CPU: 78%
  Memory: 12.4 GB / 16 GB (78%)
  GPU 0: 92% util, 82°C

Assigned Layers: 12-19
Active Tasks: 3
Completed Tasks: 1,247
Error Rate: 0.02%

Last Heartbeat: 234ms ago

# List all nodes
butterfly> node list

# Show node metrics
butterfly> node metrics compute-node-7 --metric cpu_usage --duration 1h
```

#### Request Commands

```bash
# Trace a request
butterfly> request trace req_abc123

Output:
Request ID: req_abc123
Trace ID: 4bf92f3577b34da6a3ce929d0e0e4736
Model: llama-70b
Status: COMPLETED
Duration: 2.345s

Timeline:
  0.000s - Request received (coordinator-1)
  0.012s - Partitioning completed
  0.025s - Layer 0 started (compute-node-1)
  0.234s - Layer 0 completed
  0.245s - Layer 1 started (compute-node-2)
  ...
  2.345s - Response sent

# Replay a request
butterfly> request replay req_abc123

Output:
Replaying request req_abc123...
[0.000s] Injecting request to coordinator
[0.012s] Observing partition assignment
[0.025s] Observing layer 0 execution
...
Replay completed. Duration: 2.356s (original: 2.345s)

# Show request timeline
butterfly> request timeline req_abc123

Output:
(ASCII timeline visualization)

# Find slow requests
butterfly> request find --slow --min-duration 5s --limit 10
```

#### State Commands

```bash
# Dump node state
butterfly> state dump compute-node-7

Output:
State dump saved to: /tmp/butterfly-state-compute-node-7-20251011.json

# Compare two nodes
butterfly> state diff compute-node-7 compute-node-8

Output:
Consensus State:
  ✓ current_term: 42 (identical)
  ✓ commit_index: 1234 (identical)
  ✗ applied_index: 1234 vs 1230 (diverged)

Memory State:
  ⚠ allocated_bytes: 12.4GB vs 11.8GB (differs)

# Show consensus log
butterfly> consensus log --term 42

Output:
Term 42 Log Entries:
  [1230] CHECKPOINT ckpt_20251011_210000
  [1231] TASK_ASSIGN task_789 -> compute-node-7
  [1232] TASK_COMPLETE task_789 SUCCESS
  [1233] HEARTBEAT coordinator-1
  [1234] CHECKPOINT ckpt_20251011_213000
```

#### Performance Profiling Commands

```bash
# Start CPU profiling
butterfly> profile start compute-node-7

Output:
Profiling started on compute-node-7
Session ID: prof_abc123
Frequency: 100 Hz

# Stop profiling
butterfly> profile stop prof_abc123

Output:
Profiling stopped.
Data collected: 10,234 samples over 30 seconds
Output file: /tmp/butterfly-profile-prof_abc123.pb.gz

# Generate flamegraph
butterfly> flamegraph generate prof_abc123

Output:
Flamegraph generated: /tmp/butterfly-flamegraph-prof_abc123.svg
Open with: firefox /tmp/butterfly-flamegraph-prof_abc123.svg

# Memory profiling
butterfly> profile memory compute-node-7

Output:
Memory Profile (Top 10 allocations):
  4.2 GB - tensor_buffer_pool
  2.1 GB - model_weights
  1.5 GB - activation_cache
  ...

# GPU profiling
butterfly> profile gpu compute-node-7 --duration 30s

Output:
Starting GPU profiling for 30 seconds...
Use NVIDIA Nsight Systems to view: /tmp/butterfly-gpu-compute-node-7.nsys-rep
```

#### Network Debugging Commands

```bash
# List connections
butterfly> connections list

Output:
Active Connections:
  coordinator-1 <-> coordinator-2 (RTT: 1.2ms, bandwidth: 10 Gbps)
  coordinator-1 <-> compute-node-1 (RTT: 0.8ms, bandwidth: 10 Gbps)
  compute-node-1 <-> compute-node-2 (RTT: 1.5ms, bandwidth: 10 Gbps)
  ...

# Show latency matrix
butterfly> latency matrix

Output:
           coord-1  coord-2  comp-1  comp-2
coord-1       -      1.2ms   0.8ms   2.1ms
coord-2     1.2ms      -     1.5ms   1.8ms
comp-1      0.8ms    1.5ms     -     1.5ms
comp-2      2.1ms    1.8ms   1.5ms     -

# Show bandwidth usage
butterfly> bandwidth usage

Output:
Total Cluster Bandwidth: 42.3 Gbps

Top Connections by Bandwidth:
  compute-node-1 -> compute-node-2: 8.4 Gbps (tensor transfers)
  compute-node-2 -> compute-node-3: 7.9 Gbps (tensor transfers)
  ...

# Capture packets
butterfly> packet capture start compute-node-1 compute-node-2

Output:
Packet capture started.
Session ID: pcap_xyz789
Filter: host compute-node-1 and host compute-node-2

# Stop packet capture
butterfly> packet capture stop pcap_xyz789

Output:
Packet capture stopped.
Captured 12,458 packets (142 MB)
Output file: /tmp/butterfly-capture-pcap_xyz789.pcap
Analyze with: wireshark /tmp/butterfly-capture-pcap_xyz789.pcap
```

#### Byzantine Debugging Commands

```bash
# Verify checkpoint
butterfly> verify checkpoint ckpt_20251011_213000

Output:
Checkpoint: ckpt_20251011_213000
Merkle Root: a1b2c3d4e5f6...

Verification Results:
  ✓ coordinator-1: VALID (merkle root matches)
  ✓ coordinator-2: VALID (merkle root matches)
  ✓ coordinator-3: VALID (merkle root matches)
  ✓ compute-node-1: VALID (merkle root matches)
  ✗ compute-node-7: INVALID (merkle root mismatch)
    Expected: a1b2c3d4e5f6...
    Actual:   f6e5d4c3b2a1...

# Verify node
butterfly> verify node compute-node-7

Output:
Node: compute-node-7

Verification Checks:
  ✓ Signature verification: PASSED
  ✓ State hash: PASSED
  ✗ Result comparison: FAILED
    Layer 12 output differs from majority
    Expected shape: [32, 128, 4096]
    Actual shape: [32, 128, 4096]
    L2 distance: 0.0423 (threshold: 0.001)

Recommendation: Node compute-node-7 may be faulty. Consider:
  1. Checking hardware (GPU memory errors?)
  2. Reviewing recent logs
  3. Restarting node with checkpoint restore
  4. If issue persists, remove from cluster

# Show reputation scores
butterfly> reputation list

Output:
Node Reputation Scores:
  coordinator-1: 1.000
  coordinator-2: 1.000
  coordinator-3: 1.000
  compute-node-1: 1.000
  ...
  compute-node-7: 0.823 (degraded due to verification failures)
  ...
```

## Common Debugging Scenarios

### Scenario 1: High Latency

**Symptoms:**
- P99 latency > 5 seconds
- User reports slow inference

**Debugging Steps:**

1. **Check Grafana Dashboard**
   - Open "Request Latency" dashboard
   - Identify which percentile is elevated (p50, p95, p99)
   - Check if cluster-wide or specific nodes

2. **Query Prometheus**
   ```promql
   # Identify slow layers
   topk(10,
     histogram_quantile(0.99,
       rate(butterfly_layer_compute_duration_seconds_bucket[5m])
     ) by (layer_id)
   )
   ```

3. **Find Slow Traces in Jaeger**
   - Filter: Service=butterfly, Operation=inference.request, Min Duration=5s
   - Examine span waterfall
   - Identify bottleneck span

4. **Check Logs for Trace**
   ```logql
   {job="butterfly"} | json | trace_id="<trace_id>" | level="WARN" or level="ERROR"
   ```

5. **Profile Slow Node**
   ```bash
   butterfly> profile start <slow_node_id>
   # Wait 30 seconds
   butterfly> flamegraph generate <session_id>
   ```

6. **Check Resource Utilization**
   ```promql
   butterfly_cpu_usage_ratio{node_id="<node_id>"}
   butterfly_memory_allocated_bytes{node_id="<node_id>"}
   butterfly_gpu_utilization_ratio{node_id="<node_id>"}
   ```

7. **Analyze Network**
   ```bash
   butterfly> latency matrix
   butterfly> bandwidth usage
   ```

**Common Causes:**
- Overloaded node (high CPU/memory/GPU)
- Network congestion
- Disk I/O bottleneck (checkpoint writes)
- Large tensor sizes
- Inefficient partitioning

### Scenario 2: Byzantine Fault Detected

**Symptoms:**
- Alert: `ByzantineFaultDetected`
- Logs show "fault.detected"

**Debugging Steps:**

1. **Identify Faulty Node**
   ```logql
   {job="butterfly"} | json | message=~".*fault.detected.*"
   ```

   Output:
   ```
   2025-10-11T21:45:32Z ERROR Byzantine fault detected
     node_id=compute-node-7
     fault_type=result_divergence
     layer_id=12
   ```

2. **Verify Node**
   ```bash
   butterfly> verify node compute-node-7
   ```

3. **Check Node Logs**
   ```logql
   {job="butterfly", node_id="compute-node-7"} | json | level="ERROR"
   ```

4. **Review Reputation Score**
   ```bash
   butterfly> reputation list
   ```

5. **Check Hardware**
   ```bash
   # GPU memory errors?
   butterfly> node metrics compute-node-7 --metric gpu_memory_errors

   # Temperature?
   butterfly> node metrics compute-node-7 --metric gpu_temperature
   ```

6. **Inspect State Divergence**
   ```bash
   butterfly> state diff compute-node-7 <healthy_node_id>
   ```

7. **Review Recent Checkpoints**
   ```bash
   butterfly> verify checkpoint <latest_checkpoint>
   ```

**Remediation:**
```bash
# If hardware issue, remove node from cluster
butterfly> cluster remove compute-node-7

# If software issue, restart node with checkpoint restore
butterfly> node restart compute-node-7 --restore-checkpoint ckpt_latest

# If persistent issue, mark node as observer (read-only)
butterfly> node demote compute-node-7 --role observer
```

### Scenario 3: Network Partition

**Symptoms:**
- Alert: `NetworkPartition`
- Cluster enters DEGRADED state
- Some nodes unreachable

**Debugging Steps:**

1. **Check Connection Status**
   ```bash
   butterfly> connections list
   ```

   Look for connections with status != ACTIVE

2. **View Connection Metrics**
   ```promql
   butterfly_connection_active == 0
   ```

3. **Check Latency Matrix**
   ```bash
   butterfly> latency matrix
   ```

   Look for missing data or high latency

4. **Review Consensus Logs**
   ```bash
   butterfly> consensus log --term <current_term>
   ```

   Look for heartbeat failures

5. **Check Network Errors**
   ```promql
   rate(butterfly_connection_errors_total[5m]) by (error_type)
   ```

6. **Packet Capture**
   ```bash
   butterfly> packet capture start <node1> <node2>
   # Wait for issue to reproduce
   butterfly> packet capture stop <session_id>
   ```

   Analyze with Wireshark

7. **Check Kubernetes Network Policy (if applicable)**
   ```bash
   kubectl get networkpolicies -n butterfly
   kubectl describe networkpolicy <policy_name>
   ```

**Common Causes:**
- Network switch failure
- Kubernetes network policy blocking traffic
- Firewall rules
- DNS resolution issues
- MTU mismatch causing packet fragmentation

### Scenario 4: Memory Leak

**Symptoms:**
- Memory usage continuously increasing
- OOM kills
- Alert: `HighMemoryUsage`

**Debugging Steps:**

1. **Check Memory Metrics**
   ```promql
   butterfly_memory_allocated_bytes{node_id="<node_id>"}
   ```

   Graph over time to confirm leak

2. **Profile Memory**
   ```bash
   butterfly> profile memory <node_id>
   ```

3. **Generate Heap Dump**
   ```bash
   # SSH to node
   ssh <node>

   # If using jemalloc with profiling
   killall -USR1 butterfly-compute-node

   # Analyze heap dump
   jeprof --show_bytes --pdf butterfly-compute-node jeprof.* > heap.pdf
   ```

4. **Check for Tensor Leaks**
   ```bash
   butterfly> state dump <node_id>
   ```

   Look for:
   - Growing tensor_buffer_pool
   - Uncleaned activation_cache
   - Orphaned intermediate tensors

5. **Review Recent Code Changes**
   - Check git log for memory management changes
   - Look for missing `drop()` calls
   - Check for reference cycles (Arc<Mutex<...>>)

6. **Enable Memory Debugging**
   ```bash
   # Restart node with memory debugging
   RUST_LOG=debug,butterfly::memory=trace butterfly-compute-node
   ```

**Remediation:**
- Restart node as temporary fix
- Deploy fix with proper cleanup
- Add memory usage tests to CI

### Scenario 5: Consensus Stuck

**Symptoms:**
- Alert: `ConsensusStalled`
- Commit index not advancing
- Requests queued but not processed

**Debugging Steps:**

1. **Check Consensus Metrics**
   ```promql
   # Commit lag
   butterfly_consensus_log_entries_total - butterfly_consensus_commit_index
   ```

2. **View Cluster Status**
   ```bash
   butterfly> cluster status
   ```

   Check:
   - Is there a leader?
   - Is leader responsive?
   - Cluster size >= quorum?

3. **Review Consensus Logs**
   ```bash
   butterfly> consensus log --term <current_term> --limit 50
   ```

4. **Check Heartbeat Failures**
   ```promql
   rate(butterfly_consensus_heartbeat_failures_total[5m])
   ```

5. **Examine Leader Node**
   ```bash
   butterfly> node inspect <leader_node_id>
   ```

   Check resource usage - is leader overloaded?

6. **Review Distributed Traces**
   - Find traces for consensus operations
   - Look for timeout spans

**Common Causes:**
- Leader node overloaded
- Network partition
- Clock skew between nodes
- Bug in consensus implementation

**Remediation:**
```bash
# Force leader election
butterfly> consensus trigger-election

# If stuck persistently, restart coordinator nodes
kubectl rollout restart deployment/butterfly-coordinator
```

### Scenario 6: Checkpoint Corruption

**Symptoms:**
- Node fails to restore from checkpoint
- Verification failures
- Log: "checkpoint.verification_failed"

**Debugging Steps:**

1. **Verify Checkpoint**
   ```bash
   butterfly> verify checkpoint <checkpoint_id>
   ```

2. **Check Checkpoint Metadata**
   ```bash
   butterfly> state dump <node_id> --checkpoint <checkpoint_id>
   ```

3. **Compare Merkle Roots**
   ```bash
   # All nodes should have same merkle root for checkpoint
   butterfly> verify checkpoint <checkpoint_id>
   ```

4. **Check Disk Errors**
   ```promql
   butterfly_disk_io_errors_total{operation="write"}
   ```

5. **Review Checkpoint Creation Logs**
   ```logql
   {job="butterfly"} | json | checkpoint_id="<checkpoint_id>" | phase="write"
   ```

**Remediation:**
```bash
# Restore from previous checkpoint
butterfly> node restore <node_id> --checkpoint <previous_checkpoint_id>

# If all checkpoints corrupted, restore from backup
butterfly> cluster restore --backup s3://butterfly-backups/cluster-backup-20251010.tar.gz
```

## Profiling Tools

### CPU Profiling

**Using pprof:**

```bash
# Start profiling
butterfly> profile start <node_id> --type cpu --duration 30s

# Download profile
butterfly> profile download <session_id> -o profile.pb.gz

# Generate flamegraph
pprof -http=:8080 profile.pb.gz
```

**Using perf (Linux):**

```bash
# SSH to node
ssh <node>

# Record CPU profile
sudo perf record -F 99 -p $(pgrep butterfly) -g -- sleep 30

# Generate flamegraph
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Memory Profiling

**Using jemalloc:**

```bash
# Enable profiling at startup
MALLOC_CONF="prof:true,prof_prefix:jeprof,lg_prof_interval:30" butterfly-compute-node

# Dump heap profile
killall -USR1 butterfly-compute-node

# Analyze
jeprof --show_bytes --pdf butterfly-compute-node jeprof.*.heap > heap.pdf
```

**Using Valgrind (slow, for dev only):**

```bash
valgrind --tool=massif --massif-out-file=massif.out butterfly-compute-node

# Visualize
ms_print massif.out
```

### GPU Profiling

**Using NVIDIA Nsight Systems:**

```bash
# Profile GPU workload
nsys profile -o butterfly-gpu-profile butterfly-compute-node

# Open in Nsight Systems UI
nsys-ui butterfly-gpu-profile.nsys-rep
```

**Using NVIDIA Nsight Compute:**

```bash
# Profile specific kernel
ncu --set full -o butterfly-kernel-profile butterfly-compute-node

# Open in Nsight Compute UI
ncu-ui butterfly-kernel-profile.ncu-rep
```

## Packet Analysis

### Capturing Traffic

```bash
# Using butterfly debug console
butterfly> packet capture start <node1> <node2> --filter "port 8080"

# Using tcpdump directly
sudo tcpdump -i any -w capture.pcap host <node1> and host <node2>

# Decrypt QUIC traffic (requires SSLKEYLOGFILE)
SSLKEYLOGFILE=/tmp/quic-keys.log butterfly-compute-node
```

### Analyzing with Wireshark

```bash
# Open capture
wireshark capture.pcap

# Filters:
# - QUIC traffic: quic
# - Large packets: frame.len > 1500
# - Retransmissions: tcp.analysis.retransmission
```

### Analyzing QUIC Streams

```bash
# Using butterfly QUIC analyzer
butterfly-cli analyze-quic capture.pcap

Output:
QUIC Analysis Report:

Connections:
  1. compute-node-1 <-> compute-node-2
     Duration: 30.5s
     Streams: 1,234
     Bytes transferred: 4.2 GB
     Packet loss: 0.02%
     Avg RTT: 1.5ms

Stream Analysis:
  - Stream 0: Control stream (42 KB)
  - Stream 4: Tensor transfer (2.1 GB)
  - Stream 8: Tensor transfer (2.1 GB)

Congestion Control:
  - Algorithm: BBR
  - Avg cwnd: 1.2 MB
  - Pacing rate: 8.4 Gbps
```

## Log Analysis

### Finding Errors

```logql
# All errors in last hour
{job="butterfly"} | json | level="ERROR" | __timestamp__ > 1h

# Errors from specific node
{job="butterfly", node_id="compute-node-7"} | json | level="ERROR"

# Error rate by component
rate({job="butterfly"} | json | level="ERROR" [5m]) by (component)
```

### Correlating with Traces

```logql
# Get all logs for a trace
{job="butterfly"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"

# Get error logs for a trace
{job="butterfly"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736" | level="ERROR"
```

### Pattern Matching

```logql
# Find OOM errors
{job="butterfly"} | json | message=~".*out of memory.*"

# Find timeout errors
{job="butterfly"} | json | message=~".*timeout.*"

# Find Byzantine faults
{job="butterfly"} | json | message=~".*fault.*detected.*"
```

## Request Replay

### Recording Requests

Recording happens automatically when `BUTTERFLY_RECORD_REQUESTS=true`

```bash
# Enable recording
export BUTTERFLY_RECORD_REQUESTS=true
export BUTTERFLY_RECORDING_DIR=/var/lib/butterfly/recordings

# Recordings saved as: /var/lib/butterfly/recordings/req_<request_id>.json
```

### Replaying Requests

```bash
# Replay specific request
butterfly> request replay req_abc123

# Replay with slowdown
butterfly> request replay req_abc123 --speed 0.1  # 10x slower

# Replay step-by-step
butterfly> request replay req_abc123 --step

Output:
Press Enter to advance to next event...
[0.000s] Request received
> <Enter>
[0.012s] Partitioning completed
> <Enter>
...

# Replay with divergence detection
butterfly> request replay req_abc123 --detect-divergence

Output:
Divergence detected at layer 12!
Original: tensor shape [32, 128, 4096], L2 norm 234.5
Replay:   tensor shape [32, 128, 4096], L2 norm 234.7
Diff: 0.0008 (within tolerance)
```

## Best Practices

### Debugging Workflow

1. **Start with dashboards**: Get high-level view in Grafana
2. **Query metrics**: Use Prometheus to identify affected components
3. **Find traces**: Use Jaeger to see request flow
4. **Check logs**: Use Loki to get detailed error messages
5. **Deep dive**: Use debug console for node inspection
6. **Profile**: Use profiling tools to identify bottlenecks
7. **Reproduce**: Use request replay to reproduce issues

### When to Escalate

Escalate to development team if:
- Byzantine fault rate > 0.1%
- Consensus stuck for > 5 minutes
- Memory leak confirmed
- Checkpoint corruption widespread
- Unknown error patterns

### Collecting Debug Bundle

```bash
# Collect comprehensive debug bundle
butterfly-cli collect-debug-bundle --output debug-bundle.tar.gz

Bundle contains:
- Cluster status
- All node states
- Recent logs (last 1 hour)
- Metrics snapshot
- Recent traces (last 100)
- Network topology
- Checkpoint metadata

# Share with development team
```

## Appendix: Useful Queries

### Prometheus

```promql
# Top 10 slowest layers
topk(10, histogram_quantile(0.99, rate(butterfly_layer_compute_duration_seconds_bucket[5m])) by (layer_id))

# Error rate trend
rate(butterfly_request_errors_total[5m])

# Memory usage per node
sum(butterfly_memory_allocated_bytes) by (node_id)

# Network bandwidth by connection
rate(butterfly_network_bytes_sent[5m]) by (from_node, to_node)

# Byzantine fault rate
rate(butterfly_byzantine_faults_detected_total[5m])
```

### LogQL

```logql
# All errors with trace ID
{job="butterfly"} | json | level="ERROR" | line_format "{{.trace_id}} {{.message}}"

# Slow layer executions
{job="butterfly", component="execution_engine"} | json | unwrap duration_ms | duration_ms > 1000

# Consensus failures
{job="butterfly", component="consensus"} | json | message=~".*failed.*"
```

### Jaeger

- Slow requests: Min Duration > 5s
- Errors: Tags error=true
- Byzantine faults: Tags fault.detected=true
- Specific node: Tags node.id=compute-node-7

## Support Resources

- Documentation: https://docs.butterfly.ai
- GitHub Issues: https://github.com/tensorhusker/butterfly/issues
- Slack: #butterfly-support
- On-call: PagerDuty rotation
