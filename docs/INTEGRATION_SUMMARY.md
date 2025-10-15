# Butterfly Integration Architecture - Executive Summary

## Overview

This document provides a high-level summary of the complete integration architecture for Butterfly's distributed inference system.

## Problem Statement

**Current State:**
- 7 high-quality components implemented in isolation
- 0% integration between components
- Components cannot communicate or coordinate
- No end-to-end inference capability

**Target State:**
- Fully integrated distributed inference system
- Components communicate through well-defined interfaces
- Complete request lifecycle from API to result
- Production-ready architecture with observability and fault tolerance

## Architecture at a Glance

```
Client Request
      │
      ▼
┌──────────────┐
│ butterfly-api│  (HTTP Server)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│butterfly-    │  (Task Scheduling)
│schedule      │
└──┬───────┬───┘
   │       │
   │       ▼
   │  ┌──────────────┐
   │  │butterfly-    │  (Model Partitioning)
   │  │partition     │
   │  └──────────────┘
   │
   ▼
┌──────────────┐
│butterfly-    │  (Byzantine Coordination)
│coordination  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│butterfly-comm│  (Network Layer)
└──────────────┘
       │
       │ (All components emit events)
       ▼
┌──────────────┐
│butterfly-    │  (Observability)
│metrics       │
└──────────────┘
```

## Key Integration Patterns

### 1. Service-Oriented Architecture

Each component is wrapped in a service layer:

```rust
// Common pattern across all components
pub struct XyzService {
    // Dependencies (injected)
    dependency1: Arc<dyn Dependency1Interface>,
    dependency2: Arc<dyn Dependency2Interface>,

    // Metrics
    metrics: Arc<MetricsService>,

    // Internal state
    state: Arc<RwLock<InternalState>>,

    // Config
    config: XyzConfig,
}
```

**Benefits:**
- Clear dependency boundaries
- Easy to mock for testing
- Enables runtime composition
- Supports graceful degradation

### 2. Event-Driven Metrics

All components emit events to a shared metrics bus:

```rust
// Any component can emit metrics
metrics.record_event(MetricsEvent::TaskCompleted {
    task_id: 123,
    latency_ms: 45.2,
    nodes_involved: vec![NodeId(0), NodeId(1)],
});
```

**Benefits:**
- Non-blocking observability
- Centralized metrics aggregation
- No direct coupling for monitoring
- Easy to add new metric types

### 3. Dependency Injection

Components receive dependencies through constructors:

```rust
// Initialization order respects dependencies
let metrics = Arc::new(MetricsService::new(config));
let network = Arc::new(NetworkService::new(backend, metrics.clone()));
let partition = Arc::new(PartitionService::new(config, metrics.clone()));
let coordination = Arc::new(CoordinationService::new(
    config,
    network.clone(),
    metrics.clone()
));
```

**Benefits:**
- Explicit dependency graph
- Easy to validate initialization order
- Supports testing with mocks
- Clear component boundaries

### 4. Async Interfaces

All service interfaces are async traits:

```rust
#[async_trait]
pub trait SchedulingInterface: Send + Sync {
    async fn submit_task(&self, task: InferenceTask)
        -> Result<u64, SchedulingError>;

    async fn query_status(&self, task_id: u64)
        -> Result<TaskStatus, SchedulingError>;
}
```

**Benefits:**
- Non-blocking operations
- Efficient resource utilization
- Natural concurrency model
- Tokio runtime integration

## Complete Request Flow

### Step-by-Step Lifecycle

```
1. Client → API Service
   POST /inference
   { "input": [1.0, 2.0, 3.0] }

2. API → Scheduling Service
   submit_task(InferenceTask { ... })

3. Scheduling → Partition Service
   compute_partitions(layers, nodes, strategy)

4. Partition Service
   - Check cache
   - Compute optimal partition
   - Record metrics
   - Return partitions

5. Scheduling → Coordination Service
   assign_work(task_id, partitions)

6. Coordination Service
   - Create work assignments
   - Broadcast via Network Service
   - Wait for node readiness (barrier)

7. Network Service
   - Send messages to compute nodes
   - Record transfer metrics
   - Handle retries

8. Computation Phase
   - Nodes execute their partitions
   - Send results back
   - Coordination runs Byzantine consensus

9. Aggregation & Commitment
   - Verify result correctness
   - Commit to checkpoint
   - Mark task complete

10. Metrics Service (throughout)
    - Task latency
    - Network throughput
    - Node utilization
    - Error rates
```

### Metrics Collected

- **Task Metrics**: Latency, throughput, success rate
- **Partition Metrics**: Computation time, cache hit rate, quality scores
- **Coordination Metrics**: Consensus rounds, phase transitions, failures
- **Network Metrics**: Bytes transferred, message counts, retries
- **System Metrics**: Node health, resource utilization, queue depths

## Error Handling Strategy

### Error Type Hierarchy

```
ButterflyError
├── ApiError
│   ├── InvalidRequest
│   ├── TaskNotFound
│   └── ServiceUnavailable
├── SchedulingError
│   ├── QueueFull
│   ├── TaskAlreadyExists
│   └── CoordinationUnavailable
├── PartitionError
│   ├── NoNodesAvailable
│   ├── InvalidConfiguration
│   └── InsufficientResources
├── CoordinationError
│   ├── ConsensusTimeout
│   ├── InsufficientNodes
│   └── ByzantineViolation
└── NetworkError
    ├── ConnectionFailed
    ├── SendTimeout
    └── Unreachable
```

### Recovery Strategies

1. **Transient Failures** → Retry with exponential backoff
2. **Network Failures** → Circuit breaker pattern
3. **Byzantine Failures** → Consensus-based recovery
4. **Resource Failures** → Fallback to simpler strategies
5. **Fatal Errors** → Graceful degradation and logging

## Component Dependencies

### Dependency Graph

```
butterfly-core (foundation)
    ↓
butterfly-metrics (no other deps)
    ↓
butterfly-comm (depends on: core, metrics)
    ↓
butterfly-partition (depends on: core, metrics)
    ↓
butterfly-coordination (depends on: core, comm, metrics)
    ↓
butterfly-schedule (depends on: core, coordination, partition, metrics)
    ↓
butterfly-api (depends on: core, schedule, metrics)
```

### Initialization Order

Must follow dependency graph bottom-up:
1. Metrics (no dependencies)
2. Network (needs metrics)
3. Partition (needs metrics)
4. Coordination (needs network, metrics)
5. Scheduling (needs coordination, partition, metrics)
6. API (needs scheduling, metrics)

## Configuration Management

### Unified Configuration

```toml
# config.toml

[api]
bind_address = "0.0.0.0:8080"
max_request_size_mb = 100
request_timeout_secs = 30

[scheduling]
scheduler_type = "priority"
max_queue_size = 1000

[partition]
default_strategy = "load_balanced"
cache_size = 100

[coordination]
node_id = 0
cluster_size = 3
max_byzantine = 1

[network]
backend_type = "tcp"
listen_address = "0.0.0.0:9000"
peer_addresses = ["node1:9000", "node2:9000"]

[metrics]
collector_type = "prometheus"
export_interval_secs = 60
```

### Environment-Specific Configs

- `config.dev.toml` - Development (local backend, verbose logging)
- `config.test.toml` - Testing (mock backends, no network)
- `config.prod.toml` - Production (libp2p, distributed coordination)

## Testing Strategy

### Unit Tests

Each component has isolated unit tests with mocks:

```rust
#[tokio::test]
async fn test_scheduling_service() {
    let mock_coordination = Arc::new(MockCoordinationService::new());
    let mock_partition = Arc::new(MockPartitionService::new());
    let metrics = Arc::new(MetricsService::new(test_config()));

    let service = SchedulingService::new(
        Box::new(FifoScheduler::new()),
        mock_coordination,
        mock_partition,
        metrics,
        SchedulingConfig::default(),
    );

    // Test scheduling logic...
}
```

### Integration Tests

End-to-end tests with real components:

```rust
#[tokio::test]
async fn test_complete_inference_pipeline() {
    let system = ButterflySystem::new(SystemConfig::test_defaults())
        .await
        .unwrap();

    // Submit request
    let task = InferenceTask { ... };
    let task_id = system.scheduling.submit_task(task).await.unwrap();

    // Wait for completion
    loop {
        let status = system.scheduling.query_status(task_id).await.unwrap();
        if status == TaskStatus::Completed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Verify metrics
    let metrics = system.metrics.export_all().await;
    assert!(metrics.tasks_completed >= 1);
}
```

### Mock Implementations

Each interface has a mock for testing:

```rust
pub struct MockCoordinationService {
    assignments: Arc<RwLock<Vec<WorkAssignment>>>,
}

#[async_trait]
impl CoordinationInterface for MockCoordinationService {
    async fn assign_work(&self, task_id: u64, partitions: Vec<ModelPartition>)
        -> Result<WorkAssignment, CoordinationError>
    {
        let assignment = WorkAssignment { task_id, partitions, ... };
        self.assignments.write().await.push(assignment.clone());
        Ok(assignment)
    }
}
```

## Fault Tolerance Mechanisms

### 1. Circuit Breaker (Network)

Prevents cascading failures from unreachable nodes:

```
Closed (normal) → Open (failing) → HalfOpen (testing) → Closed
```

### 2. Retry with Backoff (Network)

Handles transient failures:

```
Attempt 1: immediate
Attempt 2: wait 100ms
Attempt 3: wait 200ms
Give up: propagate error
```

### 3. Byzantine Consensus (Coordination)

Tolerates up to `f` malicious nodes in a `2f+1` cluster:

```
PrePrepare → Prepare → Commit → Checkpoint
```

### 4. Checkpointing (Coordination)

Enables recovery after failures:

```
Every N tasks → Create checkpoint → Store state
On failure → Restore from last checkpoint
```

### 5. Graceful Degradation (Partition)

Falls back to simpler strategies on failure:

```
TopologyAware (fails) → LoadBalanced (fails) → Uniform (always works)
```

## Performance Considerations

### Caching Strategy

- **Partition Cache**: LRU cache for computed partitions (hit rate >90%)
- **Model Cache**: Cache loaded model layers
- **Result Cache**: Cache recent inference results

### Zero-Copy Optimization

- **Tensor Transfer**: Use shared memory when possible
- **Message Passing**: Minimize serialization overhead
- **Buffer Pooling**: Reuse allocated buffers

### Async Concurrency

- **Task Parallelism**: Multiple requests processed concurrently
- **Pipeline Parallelism**: Overlapping computation and communication
- **Event Processing**: Non-blocking metrics collection

### Network Optimization

- **Message Batching**: Combine small messages
- **Compression**: Optional compression for large tensors
- **Protocol Selection**: TCP for reliability, QUIC for performance

## Observability

### Metrics Export

- **Prometheus**: Standard /metrics endpoint
- **Console**: Human-readable logs
- **JSON**: Structured logging for analysis

### Key Metrics

```
# Task Metrics
task_latency_seconds{quantile="0.5"}
task_latency_seconds{quantile="0.95"}
task_latency_seconds{quantile="0.99"}
tasks_total{status="completed"}
tasks_total{status="failed"}

# Network Metrics
bytes_transferred_total
messages_sent_total
messages_received_total

# Partition Metrics
partition_computation_duration_seconds
partition_cache_hit_ratio

# Coordination Metrics
consensus_rounds_total
phase_transitions_total
node_failures_total
```

### Health Checks

Each service exposes health status:

```
GET /health
{
  "healthy": true,
  "services": {
    "scheduling": "ok",
    "coordination": "ok",
    "network": "ok",
    "partition": "ok"
  },
  "nodes": [
    {"id": 0, "status": "active"},
    {"id": 1, "status": "active"}
  ]
}
```

## Security Considerations

### Byzantine Fault Tolerance

- Tolerates malicious nodes up to configured threshold
- Cryptographic signatures on coordination messages
- Merkle trees for result verification

### Network Security

- TLS for inter-node communication (optional)
- Authentication tokens for API access
- Rate limiting on API endpoints

### Resource Limits

- Max request size enforcement
- Queue size limits to prevent DoS
- Timeouts on all blocking operations

## Deployment Model

### Single Node (Development)

```rust
let config = SystemConfig {
    coordination: CoordinationConfig {
        cluster_size: 1,
        max_byzantine: 0,
        ...
    },
    network: NetworkConfig {
        backend_type: "local",
        ...
    },
    ...
};
```

### Multi-Node Cluster (Production)

```rust
// Node 0 (primary)
let config = SystemConfig {
    coordination: CoordinationConfig {
        node_id: 0,
        cluster_size: 5,
        max_byzantine: 2,
        ...
    },
    network: NetworkConfig {
        backend_type: "libp2p",
        peer_addresses: vec![
            "node1:9000",
            "node2:9000",
            "node3:9000",
            "node4:9000",
        ],
        ...
    },
    ...
};

// Nodes 1-4 (workers)
// Similar config with different node_id
```

## Migration Path

### Phase 1: Foundation (Weeks 1-2)

- Implement butterfly-core service traits
- Create unified error types
- Add configuration management
- Build metrics infrastructure

### Phase 2: Service Wrappers (Weeks 3-5)

- Wrap each component in service layer
- Add dependency injection
- Implement interfaces
- Create mock implementations

### Phase 3: Integration (Weeks 6-7)

- Connect components through interfaces
- Implement ButterflySystem orchestrator
- Add integration tests
- Verify complete request flows

### Phase 4: Production Hardening (Week 8)

- Performance profiling
- Stress testing
- Security audit
- Documentation

## Success Metrics

### Integration Success

- ✅ All 7 components communicate correctly
- ✅ Complete end-to-end inference pipeline
- ✅ Metrics collected from all components
- ✅ Error propagation works correctly
- ✅ System can initialize and shutdown cleanly

### Performance Targets

- Task latency p95 < 500ms (for small models)
- Network overhead < 10% of computation time
- Partition computation < 50ms (cached)
- System throughput > 100 requests/second

### Quality Targets

- Integration test coverage > 80%
- Zero memory leaks under load
- Graceful degradation under failures
- Clean shutdown within 30 seconds

## Next Steps

1. **Review this architecture** with the team
2. **Start with Phase 1** (butterfly-core foundation)
3. **Implement incrementally** following the dependency graph
4. **Test continuously** with unit and integration tests
5. **Iterate based on feedback** from actual usage

## Related Documents

- **integration_architecture.md** - Detailed component wiring and interfaces
- **INTEGRATION_PATTERNS.md** - Code examples for each pattern
- **INTEGRATION_IMPLEMENTATION_PLAN.md** - Step-by-step implementation guide

---

**Document Version**: 1.0
**Date**: 2025-10-11
**Status**: Design Specification
