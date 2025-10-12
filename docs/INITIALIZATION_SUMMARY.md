# Butterfly Initialization System - Executive Summary

## Document Purpose

This document provides a high-level overview of the Butterfly cluster initialization and bootstrap system. It ties together the detailed specifications and serves as the entry point for understanding how a Butterfly cluster starts up.

**Last Updated**: 2025-10-11
**Status**: Design Specification

## Quick Links

- **Detailed Design**: [initialization_design.md](initialization_design.md)
- **Protocol Specification**: [BOOTSTRAP_PROTOCOL.md](BOOTSTRAP_PROTOCOL.md)
- **Crate Specification**: [../crates/butterfly-cluster/SPECIFICATION.md](../crates/butterfly-cluster/SPECIFICATION.md)
- **Architecture**: [architecture.md](architecture.md)
- **Coordination Protocol**: [coordination_protocol.md](coordination_protocol.md)

---

## System Overview

The Butterfly initialization system transforms a collection of independent processes into a coordinated, fault-tolerant cluster ready to serve distributed inference requests. The design prioritizes:

1. **Correctness**: Formal guarantees about cluster state consistency
2. **Fault Tolerance**: Graceful handling of partial failures
3. **Observable**: Rich telemetry for debugging and monitoring
4. **Performance**: Minimize time-to-ready while maintaining safety

### High-Level Flow

```
Individual Nodes                Cluster Formation              Ready Cluster
┌──────────┐                    ┌──────────┐                  ┌──────────┐
│  Node A  │                    │ Discover │                  │ Coord: A │
│  (cold)  │────discover───────>│ Peers    │───elect────────>│ Worker:B │
└──────────┘                    └──────────┘                  │ Worker:C │
┌──────────┐                         │                        └──────────┘
│  Node B  │────────────────────────┤                              │
│  (cold)  │                         │                         Assign
└──────────┘                         │                         Partitions
┌──────────┐                         │                              │
│  Node C  │────────────────────────┘                          Load &
│  (cold)  │                                                   Validate
└──────────┘                                                        │
                                                                    ▼
                                                             ┌──────────┐
                                                             │  READY   │
                                                             │ (serving)│
                                                             └──────────┘
```

**Typical Timeline** (70B model, 8 nodes, S3 weights):
- **Discovery**: 10-30 seconds
- **Election**: 1-5 seconds
- **Assignment**: 5-10 seconds
- **Loading**: 90-180 seconds
- **Validation**: 10-20 seconds
- **Total**: 2-4 minutes

---

## Architecture Components

### 1. Node State Machine

Each node progresses through deterministic states:

```
COLD → BOOTSTRAP → JOINING → LOADING → VALIDATING → READY → OPERATIONAL
                      ↓          ↓          ↓
                   FAILED ← RECOVERING ← DEGRADED
```

**Key States**:
- **COLD**: Process just started
- **BOOTSTRAP**: Loading config, detecting hardware
- **JOINING**: Connecting to cluster
- **LOADING**: Downloading and loading model weights
- **VALIDATING**: Cross-checking weights with peers
- **READY**: Waiting for inference requests
- **OPERATIONAL**: Actively serving requests

### 2. Cluster State Machine

The cluster as a whole transitions through:

```
UNFORMED → FORMING → ELECTING → DISTRIBUTING → VERIFYING → READY → ACTIVE
```

**Key States**:
- **FORMING**: Collecting nodes, waiting for quorum
- **ELECTING**: Running Raft leader election
- **DISTRIBUTING**: Coordinator assigning partitions
- **VERIFYING**: Cross-validating model weights
- **READY**: Cluster ready for first request

### 3. Core Protocols

#### Discovery Protocol
Nodes find each other via:
- **Static**: Predefined seed node list
- **DNS**: SRV record lookups
- **Consul/etcd**: Service discovery integration
- **Multicast**: Local network broadcast (optional)

#### Join Protocol
```
Worker                          Coordinator
  │──────── JoinRequest ──────────>│
  │  {node_id, capabilities}       │
  │                                 │
  │<─────── JoinResponse ───────────│
  │  {cluster_config, manifest}    │
  │                                 │
  │──────── JoinAck ────────────────>│
```

#### Election Protocol (Raft-based)
```
Follower A     Follower B     Follower C
    │              │              │
    │ [timeout]    │              │
    │──RequestVote>│              │
    │──────────────┼─RequestVote─>│
    │              │              │
    │<─Vote────────│              │
    │<─────────────┼─Vote─────────│
    │              │              │
    │ [Won]        │              │
    │──Heartbeat──>│              │
    │──────────────┼─Heartbeat───>│
```

#### Model Distribution Protocol
```
Coordinator              Worker
    │                      │
    │──ModelManifest──────>│
    │──PartitionAssign────>│
    │                      │
    │                 [fetch weights]
    │                 [verify checksums]
    │                      │
    │<─LoadProgress────────│
    │<─LoadProgress────────│  (periodic)
    │<─LoadComplete────────│
    │                      │
    │──ChecksumRequest────>│
    │<─ChecksumReport──────│
    │                      │
    │ [compute Merkle root]│
    │                      │
    │──ValidationResult───>│
```

---

## Key Design Decisions

### 1. Raft for Coordinator Election

**Decision**: Use Raft consensus for leader election rather than simpler approaches.

**Rationale**:
- Proven correctness properties (single leader per term)
- Handles network partitions correctly
- Naturally extends to replicated state machine for future HA
- Well-understood implementation patterns

**Trade-off**: Slightly more complex than single-leader static assignment, but provides critical fault tolerance.

### 2. Centralized Partition Assignment

**Decision**: Coordinator computes and distributes partition assignments.

**Rationale**:
- Optimal partitioning requires global view of cluster resources
- Simplifies reasoning about partition consistency
- Enables sophisticated assignment algorithms (DP-based optimization)

**Trade-off**: Coordinator is temporary bottleneck during initialization, but this is one-time cost.

### 3. Cross-Validation with Merkle Trees

**Decision**: Validate weight consistency using Merkle tree of partition checksums.

**Rationale**:
- Detects corrupted downloads or Byzantine nodes
- O(N) communication (each node reports once)
- Provides verifiable proof of global consistency
- Enables efficient identification of mismatches

**Trade-off**: Additional 10-20 seconds to initialization, but prevents silent corruption.

### 4. Pull-Based Weight Distribution

**Decision**: Workers fetch weights from source (S3, filesystem) rather than coordinator streaming.

**Rationale**:
- Avoids coordinator bandwidth bottleneck
- Leverages object storage parallelism and CDN
- Workers can fetch at their own pace
- Natural retry behavior

**Trade-off**: Requires shared storage infrastructure, but this is standard for ML systems.

### 5. Adaptive Timeouts

**Decision**: Timeouts adjust based on observed performance.

**Rationale**:
- Accommodates varying model sizes and network speeds
- Reduces false failures on slow-but-functioning nodes
- Improves user experience across diverse deployments

**Trade-off**: Slightly more complex timeout logic, but significantly better robustness.

---

## Failure Scenarios and Recovery

### Scenario 1: Node Crashes During Init

**Detection**: Coordinator notices missing progress reports
**Recovery**: Repartition model excluding failed node
**Impact**: 30-60s delay, no data loss
**Guarantee**: Cluster proceeds if quorum remains

### Scenario 2: Coordinator Crashes

**Detection**: Workers miss 3 consecutive heartbeats (300ms)
**Recovery**: Raft election (~200-500ms)
**Impact**: <1s downtime
**Guarantee**: New leader has all committed state

### Scenario 3: Network Partition

**Detection**: Bidirectional heartbeat failures
**Behavior**: Majority partition continues, minority waits
**Recovery**: Automatic after partition heals
**Guarantee**: No split-brain, minority safely halts

### Scenario 4: Corrupted Weights

**Detection**: Checksum mismatch during load or validation
**Recovery**: Retry download up to 3 times, then fetch from peer
**Impact**: Delays that node's loading
**Guarantee**: Invalid weights never enter cluster

### Scenario 5: Insufficient Memory

**Detection**: Pre-flight feasibility check or OOM during load
**Recovery**: Reject startup with clear error message
**Fallback**: Suggest quantization or adding nodes
**Guarantee**: Fail-safe rather than undefined behavior

---

## Formal Correctness Properties

### Safety Properties

**Theorem 1: Single Coordinator Per Term**

For any term T, at most one node is coordinator.

*Proof*: Raft election guarantees single leader per term. Each node votes once per term. Majority (quorum) required to win. Two majorities must overlap, preventing split votes.

**Theorem 2: Consistent Model Weights**

All operational nodes have identical model weights for their assigned layers.

*Proof*: Weights validated via cryptographic checksums. Merkle tree aggregation ensures global consistency. Byzantine agreement requires 2f+1 matching checksums. With ≤f Byzantine nodes, honest majority guarantees correct weights.

**Theorem 3: No Inference Before Ready**

Cluster never serves requests until all nodes validate as ready.

*Proof*: Coordinator gates request acceptance on ValidationResult with cluster_ready=true. ValidationResult only set after receiving ChecksumReports from quorum. ChecksumReports only sent after successful warmup. Therefore, all nodes proven ready before first request.

### Liveness Properties

**Theorem 4: Initialization Terminates**

If ≥quorum_size nodes are operational and network is eventually synchronous, initialization completes in bounded time.

*Proof*: Each phase has deterministic timeout. Discovery terminates (finite peers to contact). Election terminates (Raft guarantee). Loading terminates (finite bytes to transfer). Validation terminates (finite checksums to verify). Sum of phase timeouts is finite bound.

**Theorem 5: Progress Under Failures**

Cluster makes progress as long as operational_nodes ≥ quorum_size.

*Proof*: Quorum-based protocols never require unanimity. Failed nodes excluded from quorum calculations. Repartitioning algorithm guarantees feasible assignment exists if quorum present. Therefore, cluster proceeds.

---

## Performance Characteristics

### Time Complexity

| Phase | Algorithm Complexity | Typical Latency |
|-------|---------------------|-----------------|
| Discovery | O(N) peers to contact | 5-15s |
| Election | O(N²) messages in worst case | 0.5-2s |
| Assignment | O(L log N) dynamic programming | 1-5s |
| Loading | O(W) bytes to transfer | 60-300s |
| Validation | O(N) checksum reports | 5-15s |

Where:
- N = number of nodes
- L = number of model layers
- W = total weight size in bytes

### Communication Complexity

| Operation | Messages | Total Bytes |
|-----------|----------|-------------|
| Discovery | O(N) | ~1KB per node |
| Election | O(N²) worst case | ~1KB per message |
| Assignment | O(N) | ~10KB per node |
| Progress Reports | O(N × P) | ~1KB per report |
| Validation | O(N) | ~1KB per checksum |

Where P = number of progress reports per node (typically 10-20).

**Total Network Overhead**: Typically <1MB for cluster formation, <10MB for progress tracking, negligible compared to model transfer.

### Scalability

**Horizontal Scalability**:
- Discovery: O(N) - linear in node count
- Election: O(N²) messages but <1s in practice up to 100 nodes
- Assignment: O(L log N) - near-linear with smart algorithms
- Loading: Embarrassingly parallel - perfect scaling
- Validation: O(N) - linear in node count

**Practical Limits**:
- Tested up to 32 nodes (design)
- Expected to scale to 100+ nodes with tuning
- Bottleneck shifts from protocol to network bandwidth at large scale

---

## Security Model

### Phase 1: Trusted Network (Current)

**Assumptions**:
- All nodes are trusted (no Byzantine behavior)
- Network is private (no eavesdropping)
- Physical security protects nodes

**Mitigations**:
- Checksum validation (detects corruption, not malice)
- Protocol-level sanity checks
- Comprehensive logging

### Phase 2: Authentication (Planned)

**Features**:
- Ed25519 public key authentication
- Message signature verification
- Node identity verification
- Replay attack prevention

### Phase 3: Encryption (Future)

**Features**:
- TLS 1.3 for all connections
- Encrypted weight transfer
- Perfect forward secrecy

### Phase 4: Byzantine Tolerance (Long-term)

**Features**:
- Full PBFT-style agreement on results
- Computational proofs (ZK-SNARKs?)
- Reputation system for node behavior
- Automatic Byzantine node isolation

---

## Observability

### Metrics

```rust
// Initialization metrics
init_duration_seconds{phase="discovery"}
init_duration_seconds{phase="election"}
init_duration_seconds{phase="loading"}
init_duration_seconds{phase="validation"}

// Model loading metrics
model_bytes_downloaded_total
model_download_speed_mbps
model_load_errors_total{reason}

// Cluster health
cluster_size
cluster_operational_nodes
cluster_failed_nodes
election_count_total
election_duration_seconds
```

### Tracing

Distributed traces span the entire initialization:

```
Trace: cluster-initialization
├─ Span: discovery [5.2s]
│  ├─ Span: dns-lookup [0.3s]
│  └─ Span: peer-handshake [0.1s per peer]
├─ Span: election [0.8s]
│  ├─ Span: request-votes [0.2s]
│  └─ Span: await-quorum [0.6s]
├─ Span: assignment [2.1s]
│  ├─ Span: compute-partition [1.9s]
│  └─ Span: distribute-assignments [0.2s]
├─ Span: loading [124.3s]
│  ├─ Span: fetch-weights [110s]
│  ├─ Span: verify-checksums [8s]
│  └─ Span: gpu-upload [6.3s]
└─ Span: validation [12.5s]
   ├─ Span: collect-checksums [1.2s]
   └─ Span: merkle-validation [11.3s]
```

### Logging

Structured logs at each phase:

```json
{
  "timestamp": "2025-10-11T10:23:45Z",
  "level": "INFO",
  "node_id": "node-a-7f3d",
  "phase": "loading",
  "message": "Weight loading progress",
  "progress_percent": 45.2,
  "bytes_loaded": 38654705664,
  "total_bytes": 85899345920,
  "current_file": "layer_20-29.safetensors",
  "eta_seconds": 68
}
```

---

## Implementation Checklist

### Phase 1: Core Framework ✅ Specified

- [x] State machine design
- [x] Protocol message definitions
- [x] bootstrap_protocol.md specification
- [ ] `butterfly-cluster` crate scaffolding
- [ ] State machine implementation
- [ ] Unit tests for state transitions

### Phase 2: Discovery & Election 📋 Next

- [ ] Static discovery implementation
- [ ] DNS discovery implementation
- [ ] Raft election implementation
- [ ] Join/leave protocol
- [ ] Integration tests

### Phase 3: Model Loading ⏳ Blocked on Phase 2

- [ ] Model manifest parsing
- [ ] Filesystem model source
- [ ] S3 model source
- [ ] Partition assignment algorithm
- [ ] Checksum validation

### Phase 4: Validation ⏳ Blocked on Phase 3

- [ ] Merkle tree construction
- [ ] Cross-validation protocol
- [ ] Byzantine detection (basic)
- [ ] Ready state determination

### Phase 5: Production Hardening ⏳ Future

- [ ] Adaptive timeouts
- [ ] Failure recovery
- [ ] Comprehensive error handling
- [ ] Performance optimization
- [ ] Security features (TLS, auth)

---

## Usage Example

### Starting a Cluster

**Node A (Coordinator)**:
```bash
butterfly-node \
  --config coordinator.toml \
  --role coordinator \
  --bind 0.0.0.0:7000 \
  --cluster-name production \
  --model-path s3://ml-models/llama-70b-v2
```

**Node B, C, D (Workers)**:
```bash
butterfly-node \
  --config worker.toml \
  --role worker \
  --bind 0.0.0.0:7000 \
  --seed-nodes coordinator.example.com:7000 \
  --cluster-name production
```

**Expected Output**:
```
[INFO ] Node starting: id=node-b-3a2f
[INFO ] Detecting hardware: 64 CPU cores, 2x H100 GPUs
[INFO ] Discovering peers via DNS...
[INFO ] Discovered 3 peers
[INFO ] Joining cluster 'production'...
[INFO ] Joined cluster: coordinator=node-a-7f3d, members=4
[INFO ] Waiting for partition assignment...
[INFO ] Assigned layers 20-39 (18.2 GB)
[INFO ] Loading weights from s3://ml-models/llama-70b-v2
[INFO ] Loading progress: 25% (4.6 GB / 18.2 GB) - ETA 90s
[INFO ] Loading progress: 50% (9.1 GB / 18.2 GB) - ETA 60s
[INFO ] Loading progress: 75% (13.7 GB / 18.2 GB) - ETA 30s
[INFO ] Loading complete: 115.3s
[INFO ] Running warmup inference...
[INFO ] Warmup latency: 42.3ms
[INFO ] Validating weights with cluster...
[INFO ] Validation passed: merkle_root=0xabcd1234
[INFO ] Node ready: state=OPERATIONAL
[INFO ] Cluster ready for inference requests
```

### Monitoring Initialization

**Prometheus Metrics**:
```promql
# Time to ready
init_duration_seconds{phase="total"}

# Current cluster state
cluster_operational_nodes / cluster_size

# Model loading progress
rate(model_bytes_downloaded_total[1m])
```

**Grafana Dashboard**: Shows real-time progress through initialization phases, estimated time to completion, and early warning of stuck nodes.

---

## Troubleshooting Guide

### Problem: Nodes can't discover each other

**Symptoms**: `Discovered 0 peers` after 30s
**Causes**:
- Incorrect seed node addresses
- Firewall blocking ports
- DNS resolution failure

**Solutions**:
```bash
# Test connectivity
nc -zv coordinator.example.com 7000

# Check DNS
dig +short _butterfly._tcp.example.com SRV

# Verify firewall
iptables -L -n | grep 7000
```

### Problem: Election never completes

**Symptoms**: Repeated `Starting election` messages, no coordinator
**Causes**:
- Network partitions
- Clock skew
- Quorum not reachable

**Solutions**:
```bash
# Check clock sync
timedatectl status

# Verify network connectivity
ping -c 3 other-node.example.com

# Check logs for split vote patterns
grep "election" butterfly.log | tail -20
```

### Problem: Weight loading fails

**Symptoms**: `LoadError::ChecksumMismatch` or timeouts
**Causes**:
- Corrupted download
- Wrong model version
- Insufficient storage space

**Solutions**:
```bash
# Verify checksum manually
sha256sum layer_00-09.safetensors

# Check disk space
df -h /var/butterfly/models

# Test S3 access
aws s3 ls s3://ml-models/llama-70b-v2/
```

### Problem: Validation fails

**Symptoms**: `ValidationError::MerkleRootMismatch`
**Causes**:
- Inconsistent model versions across nodes
- Corrupted weights on one node
- Byzantine node (rare)

**Solutions**:
```bash
# Check which node has mismatch
grep "checksum_report" butterfly.log | jq '.weight_hash'

# Force re-download on mismatched node
butterfly-admin clear-cache --node node-b-3a2f

# Exclude Byzantine node
butterfly-admin quarantine --node suspicious-node-id
```

---

## Future Enhancements

### Near-Term (3-6 months)

1. **Incremental Model Updates**: Only download changed layers on model update
2. **P2P Weight Distribution**: BitTorrent-style collaborative downloading
3. **Parallel Validation**: Validate while loading (overlap phases)
4. **Hot Reload**: Update models without full restart

### Medium-Term (6-12 months)

1. **Hierarchical Clusters**: Support multi-datacenter deployment
2. **Dynamic Repartitioning**: Adjust partitions without full restart
3. **Speculative Loading**: Predictively load models before assignment
4. **Differential Checkpointing**: Only transfer changed weights

### Long-Term (12+ months)

1. **Zero-Downtime Updates**: Rolling upgrades with traffic migration
2. **Federated Clusters**: Coordinate across organizational boundaries
3. **Quantum-Resistant Crypto**: Post-quantum signature schemes
4. **Hardware Attestation**: TPM-based secure boot verification

---

## References

### Internal Documents

- [initialization_design.md](initialization_design.md): Detailed design specification
- [BOOTSTRAP_PROTOCOL.md](BOOTSTRAP_PROTOCOL.md): Wire protocol specification
- [coordination_protocol.md](coordination_protocol.md): Runtime coordination protocol
- [architecture.md](architecture.md): Overall system architecture

### External Research

1. **Raft Consensus**
   - Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm"
   - https://raft.github.io/

2. **PBFT**
   - Castro, M. & Liskov, B. (1999). "Practical Byzantine Fault Tolerance"
   - http://pmg.csail.mit.edu/papers/osdi99.pdf

3. **Phi Accrual Failure Detector**
   - Hayashibara, N. et al. (2004). "The φ Accrual Failure Detector"
   - https://ieeexplore.ieee.org/document/1353004

4. **Distributed Model Serving**
   - Crankshaw, D. et al. (2017). "Clipper: A Low-Latency Online Prediction Serving System"
   - https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw

5. **Large Model Inference**
   - Pope, R. et al. (2022). "Efficiently Scaling Transformer Inference"
   - https://arxiv.org/abs/2211.05102

---

## Glossary

**Bootstrap**: The process of initializing a cluster from cold start to operational state.

**Coordinator**: The elected leader node responsible for cluster-wide decisions (partition assignment, work distribution).

**Epoch**: A monotonically increasing counter representing the current operational phase. Incremented on each new inference request or configuration change.

**Partition**: A contiguous range of model layers assigned to a specific worker node.

**Quorum**: The minimum number of nodes (2f+1 for f Byzantine failures) required to make progress.

**Merkle Tree**: A tree of cryptographic hashes allowing efficient verification of data consistency across nodes.

**Raft**: A consensus algorithm guaranteeing single leader election and log replication.

**Warmup**: Running a test inference to compile kernels and verify correctness before accepting real traffic.

---

## Contact & Support

**Design Questions**: See [CLAUDE.md](../CLAUDE.md) for project guidance

**Implementation Issues**: File issues on GitHub with `[initialization]` tag

**Performance Discussion**: Coordinate on `#butterfly-perf` channel

---

**Document Version**: 1.0
**Specification Status**: Design Complete, Implementation Pending
**Target Implementation**: Milestone 2 (Q1 2025)
