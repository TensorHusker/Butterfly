# Butterfly Architecture

## System Overview

Butterfly is a distributed inference system designed to partition large AI models across multiple compute nodes, enabling collaborative processing of inference requests beyond the capacity of individual machines.

### Core Philosophy

The system embodies three fundamental principles:

1. **Transparent Distribution**: Model partitioning and distributed execution should be invisible to the end user
2. **Fault Tolerance**: Individual node failures should not bring down the entire system
3. **Efficient Communication**: Minimize data transfer between nodes while maximizing parallelism

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator Node                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Request    │  │  Partition   │  │    Scheduling      │   │
│  │   Handler    │──│  Manager     │──│    Engine          │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Worker    │ │   Worker    │ │   Worker    │
    │   Node 1    │ │   Node 2    │ │   Node N    │
    │             │ │             │ │             │
    │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
    │ │Model    │ │ │ │Model    │ │ │ │Model    │ │
    │ │Partition│ │ │ │Partition│ │ │ │Partition│ │
    │ │  1      │ │ │ │  2      │ │ │ │  N      │ │
    │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## Component Architecture

### Orchestrator Node

The orchestrator serves as the central coordination point:

```
┌───────────────────────────────────────────────────────────┐
│                    Orchestrator Crate                      │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │             Request Management Layer                 │ │
│  │  • Request validation and parsing                   │ │
│  │  • Response aggregation                             │ │
│  │  • Client connection management                     │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Partition Management Layer                │ │
│  │  • Model graph analysis                             │ │
│  │  • Partition strategy selection                     │ │
│  │  • Topology optimization                            │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │            Scheduling & Coordination                │ │
│  │  • Worker health monitoring                         │ │
│  │  • Load balancing                                   │ │
│  │  • Task assignment                                  │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

**Key Responsibilities**:
- Accept inference requests from clients
- Partition models across available workers
- Coordinate distributed computation
- Aggregate results and return to clients
- Monitor system health and adapt to failures

### Worker Nodes

Workers execute model partitions and communicate intermediate results:

```
┌───────────────────────────────────────────────────────────┐
│                       Worker Crate                         │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Execution Engine                        │ │
│  │  • Tensor operations                                │ │
│  │  • Model inference                                  │ │
│  │  • Memory management                                │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Communication Manager                      │ │
│  │  • Inter-worker messaging                           │ │
│  │  • Orchestrator coordination                        │ │
│  │  • Data serialization/deserialization               │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Resource Monitor                        │ │
│  │  • Memory usage tracking                            │ │
│  │  • Compute utilization                              │ │
│  │  • Health reporting                                 │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

**Key Responsibilities**:
- Load assigned model partitions into memory
- Execute inference on local partition
- Exchange intermediate tensors with peers
- Report health and resource metrics
- Handle graceful degradation on failures

### Common Library

Shared functionality across components:

```
┌───────────────────────────────────────────────────────────┐
│                      Common Crate                          │
│                                                            │
│  • Protocol definitions (message formats)                 │
│  • Tensor representation and serialization                │
│  • Error types and handling                               │
│  • Configuration structures                               │
│  • Network utilities                                      │
│  • Logging and telemetry                                  │
└───────────────────────────────────────────────────────────┘
```

## Data Flow

### Inference Request Lifecycle

1. **Request Submission**
   - Client sends inference request to orchestrator
   - Request includes: model identifier, input tensors, configuration

2. **Partitioning Phase**
   - Orchestrator analyzes model computational graph
   - Determines optimal partition strategy based on:
     - Model architecture (layer dependencies)
     - Available worker resources
     - Network topology
     - Historical performance data

3. **Distribution Phase**
   - Orchestrator assigns partitions to workers
   - Workers load model weights for their partition
   - Workers signal readiness to orchestrator

4. **Execution Phase**
   - Input tensors distributed to first-layer workers
   - Workers execute their partition sequentially:
     - Receive input from previous layer
     - Perform local computation
     - Send output to next layer workers
   - Pipeline parallelism where possible

5. **Aggregation Phase**
   - Final layer workers send outputs to orchestrator
   - Orchestrator assembles complete inference result
   - Result returned to client

### Inter-Worker Communication Patterns

```
Layer 1 Workers          Layer 2 Workers          Layer 3 Workers
┌─────────┐             ┌─────────┐              ┌─────────┐
│ Worker  │────────────▶│ Worker  │─────────────▶│ Worker  │
│   A     │             │   D     │              │   G     │
└─────────┘             └─────────┘              └─────────┘
     │                       ▲                        ▲
     │                       │                        │
     └──────────────────┐    │                        │
                        │    │                        │
┌─────────┐            │    │                        │
│ Worker  │────────────┘    │                        │
│   B     │                 │                        │
└─────────┘                 │                        │
     │                      │                        │
     └──────────────────────┼────────────────────────┘
                            │
┌─────────┐                 │
│ Worker  │─────────────────┘
│   C     │
└─────────┘
```

## Key Design Decisions

### 1. Rust as Implementation Language

**Rationale**:
- Memory safety without garbage collection overhead
- Zero-cost abstractions for performance-critical paths
- Excellent concurrency primitives (async/await, channels)
- Strong type system catches errors at compile time
- Growing ecosystem for ML/AI infrastructure

See [ADR-001](decisions/001-rust-language-choice.md) for detailed analysis.

### 2. Workspace Structure

**Rationale**:
- Modular design enables independent development and testing
- Clear separation of concerns between components
- Shared common library eliminates code duplication
- Facilitates future extensibility (new worker types, partition strategies)

See [ADR-002](decisions/002-workspace-structure.md) for detailed analysis.

### 3. Centralized Orchestration

**Decision**: Use a centralized orchestrator rather than peer-to-peer coordination.

**Rationale**:
- Simpler reasoning about global state
- Easier to implement sophisticated partitioning algorithms
- Clear point of authority for conflict resolution
- Future path to orchestrator replication for HA

**Trade-offs**:
- Orchestrator is a potential bottleneck (mitigated by async design)
- Single point of failure (addressed in future roadmap)

### 4. Synchronous Execution Model

**Decision**: Workers execute layers synchronously with barrier synchronization.

**Rationale**:
- Simpler correctness guarantees
- Predictable memory usage
- Easier debugging and profiling
- Matches most model architectures

**Future Evolution**: Asynchronous pipeline parallelism for transformer models.

### 5. Tensor-Level Communication

**Decision**: Workers exchange complete tensors rather than fine-grained operations.

**Rationale**:
- Reduces network round trips
- Leverages efficient tensor serialization
- Aligns with natural model layer boundaries
- Simplifies worker implementation

## Scalability Considerations

### Horizontal Scaling

- **Worker Pool**: Linear scaling by adding workers
- **Partition Granularity**: More workers = finer-grained partitions
- **Network Bandwidth**: Consider topology-aware placement

### Vertical Scaling

- **Memory per Worker**: Larger partitions require more RAM
- **Compute per Worker**: GPU/TPU acceleration at worker level
- **Orchestrator Resources**: May need beefier orchestrator for large clusters

## Security Model

### Phase 1 (Current)
- Trusted network environment
- No authentication/authorization
- Focus on correctness and performance

### Phase 2 (Planned)
- Mutual TLS between components
- Worker authentication to orchestrator
- Input validation and sanitization
- Resource usage limits

## Observability

### Metrics
- Request latency (end-to-end and per-stage)
- Worker resource utilization
- Network bandwidth consumption
- Partition execution time
- Queue depths and backpressure

### Logging
- Structured logs at each component
- Distributed tracing for request flows
- Error and warning aggregation

### Debugging
- State inspection endpoints
- Partition visualization tools
- Replay capability for failed requests

## Future Architecture Evolution

### Near Term
1. Fault tolerance: Worker failure detection and recovery
2. Dynamic repartitioning: Adapt to changing workloads
3. Multi-tenancy: Isolate different models/users

### Long Term
1. Hierarchical orchestration: Multi-datacenter deployment
2. Edge integration: Hybrid cloud-edge inference
3. Federated learning: Distributed training support
4. Heterogeneous hardware: Mixed GPU/CPU/NPU workers

## References

- [Algorithms Documentation](algorithms.md)
- [Architecture Decision Records](decisions/)
- [API Documentation](../README.md)
