# Butterfly Integration Architecture

## Executive Summary

This document specifies the complete integration architecture for Butterfly's distributed inference system. The current implementation has 7 high-quality components that are completely isolated (0% integration). This architecture defines how these components will communicate, coordinate, and form a cohesive distributed system.

## Current Component Status

| Component | Purpose | Status | Integration Level |
|-----------|---------|--------|------------------|
| butterfly-core | Shared types and traits | Complete | Foundation |
| butterfly-api | HTTP API server | Isolated | 0% |
| butterfly-schedule | Task scheduling | Isolated | 0% |
| butterfly-partition | Model partitioning | Isolated | 0% |
| butterfly-comm | Network communication | Isolated | 0% |
| butterfly-coordination | Byzantine consensus | Isolated | 0% |
| butterfly-metrics | Performance monitoring | Isolated | 0% |

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                            │
│                     (HTTP/gRPC Requests)                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       butterfly-api                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ InferenceService (HTTP Server)                          │    │
│  │  - POST /inference (submit requests)                    │    │
│  │  - GET /status/{task_id} (query status)                │    │
│  │  - GET /health (system health)                          │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ (1) Submit InferenceTask
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    butterfly-schedule                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ SchedulingService                                       │    │
│  │  - Enqueue tasks (priority queue)                      │    │
│  │  - Track task lifecycle                                 │    │
│  │  - Dispatch to coordination layer                       │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────┬────────────────────┬───────────────────────────┘
               │                    │
               │ (2) Trigger        │ (3) Query partitions
               │ Work Assignment    │
               ▼                    ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│  butterfly-coordination  │   │    butterfly-partition            │
│  ┌────────────────────┐  │   │  ┌──────────────────────────┐   │
│  │ DistributedCoord   │  │   │  │ PartitionService         │   │
│  │  - Work assignment │  │   │  │  - Compute partitions    │   │
│  │  - Phase sync      │◄─┼───┼──┤  - Optimize placement    │   │
│  │  - BFT consensus   │  │   │  │  - Quality metrics       │   │
│  └────────────────────┘  │   │  └──────────────────────────┘   │
└────────┬─────────────────┘   └──────────────────────────────────┘
         │
         │ (4) Send coordination messages
         │     & tensor data
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      butterfly-comm                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ NetworkService                                          │    │
│  │  - Peer-to-peer messaging (libp2p)                     │    │
│  │  - Tensor transfer (zero-copy where possible)          │    │
│  │  - Reliable delivery with retries                      │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ (5) Publish metrics
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     butterfly-metrics                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ MetricsService (Observer)                               │    │
│  │  - Collect from all components                         │    │
│  │  - Aggregate system-wide stats                         │    │
│  │  - Export to Prometheus/console                        │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Wiring Diagram

### Detailed Component Interactions

```
butterfly-api
    │
    ├──[owns]─────────> SchedulingService (Arc<RwLock<>>)
    │                      │
    │                      ├──[uses]──> Scheduler trait impl
    │                      └──[sends]──> CoordinationService
    │
    ├──[owns]─────────> MetricsService (Arc<>)
    │                      └──[observes]─> All components
    │
    └──[queries]──────> StatusRegistry (shared state)


butterfly-schedule (SchedulingService)
    │
    ├──[owns]─────────> Scheduler (Box<dyn Scheduler>)
    │                      └─ FifoScheduler / PriorityScheduler
    │
    ├──[calls]────────> CoordinationService::assign_work()
    │
    ├──[queries]──────> PartitionService::get_partitions()
    │
    ├──[emits]────────> MetricsEvent::TaskEnqueued
    │                   MetricsEvent::TaskDispatched
    │
    └──[publishes]────> StatusUpdate to registry


butterfly-partition (PartitionService)
    │
    ├──[owns]─────────> PartitionStrategy (Box<dyn PartitionStrategyTrait>)
    │                      ├─ UniformPartitioner
    │                      ├─ LoadBalancedPartitioner
    │                      └─ TopologyAwarePartitioner
    │
    ├──[receives]─────> NodeCapability updates from coordination
    │
    ├──[caches]───────> Computed partitions (LRU cache)
    │
    └──[emits]────────> MetricsEvent::PartitionComputed


butterfly-coordination (CoordinationService)
    │
    ├──[owns]─────────> DistributedCoordinator
    │                      ├─ StateMachine (phase management)
    │                      ├─ ByzantineAgreement (consensus)
    │                      ├─ BarrierCoordinator (synchronization)
    │                      ├─ CheckpointManager (recovery)
    │                      └─ FailureDetector (liveness)
    │
    ├──[uses]─────────> NetworkService::send/broadcast
    │
    ├──[receives]─────> CoordinationMessage via NetworkService
    │
    ├──[calls]────────> PartitionService::partition()
    │
    ├──[emits]────────> MetricsEvent::PhaseTransition
    │                   MetricsEvent::ConsensusReached
    │                   MetricsEvent::NodeFailure
    │
    └──[publishes]────> WorkAssignment to compute nodes


butterfly-comm (NetworkService)
    │
    ├──[owns]─────────> CommunicationBackend (Box<dyn>)
    │                      ├─ Libp2pBackend (production)
    │                      ├─ TcpBackend (simple)
    │                      └─ LocalBackend (testing)
    │
    ├──[manages]──────> MessageRouter
    │                      ├─ Routes messages to handlers
    │                      └─ Maintains connection pool
    │
    ├──[implements]───> Reliable delivery
    │                      ├─ Retry logic with exponential backoff
    │                      ├─ Message deduplication
    │                      └─ Flow control
    │
    ├──[handles]──────> TensorTransfer (specialized)
    │                      ├─ Chunking for large tensors
    │                      ├─ Compression (optional)
    │                      └─ Zero-copy where possible
    │
    └──[emits]────────> MetricsEvent::MessageSent
                        MetricsEvent::MessageReceived
                        MetricsEvent::BytesTransferred


butterfly-metrics (MetricsService)
    │
    ├──[owns]─────────> MetricsCollector (Arc<RwLock<>>)
    │                      └─ InMemoryCollector / PrometheusCollector
    │
    ├──[subscribes]───> MetricsEvent channel (tokio::sync::broadcast)
    │
    ├──[aggregates]───> System-wide statistics
    │                      ├─ Task latency histograms
    │                      ├─ Network throughput
    │                      ├─ Node utilization
    │                      └─ Error rates
    │
    └──[exports]──────> Prometheus metrics / Console output
```

## Integration Patterns

### 1. Dependency Injection Pattern

All components use constructor injection with trait abstractions:

```rust
// Component initialization with DI
struct ButterflySystem {
    // Services (main components)
    scheduling_service: Arc<SchedulingService>,
    partition_service: Arc<PartitionService>,
    coordination_service: Arc<CoordinationService>,
    network_service: Arc<NetworkService>,
    metrics_service: Arc<MetricsService>,

    // Shared state
    status_registry: Arc<RwLock<StatusRegistry>>,

    // Configuration
    config: SystemConfig,
}

impl ButterflySystem {
    pub fn new(config: SystemConfig) -> Self {
        // 1. Create metrics service first (no dependencies)
        let metrics_service = Arc::new(MetricsService::new(
            config.metrics.clone()
        ));

        // 2. Create network service (depends on metrics)
        let network_service = Arc::new(NetworkService::new(
            config.network.clone(),
            Arc::clone(&metrics_service),
        ));

        // 3. Create partition service (depends on metrics)
        let partition_service = Arc::new(PartitionService::new(
            config.partition.clone(),
            Arc::clone(&metrics_service),
        ));

        // 4. Create coordination service (depends on network, partition, metrics)
        let coordination_service = Arc::new(CoordinationService::new(
            config.coordination.clone(),
            Arc::clone(&network_service),
            Arc::clone(&partition_service),
            Arc::clone(&metrics_service),
        ));

        // 5. Create scheduling service (depends on coordination, partition, metrics)
        let scheduling_service = Arc::new(SchedulingService::new(
            config.scheduling.clone(),
            Arc::clone(&coordination_service),
            Arc::clone(&partition_service),
            Arc::clone(&metrics_service),
        ));

        // 6. Create shared state
        let status_registry = Arc::new(RwLock::new(StatusRegistry::new()));

        Self {
            scheduling_service,
            partition_service,
            coordination_service,
            network_service,
            metrics_service,
            status_registry,
            config,
        }
    }
}
```

### 2. Message-Passing Pattern

Components communicate via typed message channels:

```rust
// Event bus for cross-component communication
pub enum SystemEvent {
    // From API to Scheduler
    InferenceRequested {
        task: InferenceTask,
        response_tx: oneshot::Sender<InferenceResponse>,
    },

    // From Scheduler to Coordination
    WorkAssignmentNeeded {
        task_id: u64,
        layers: Vec<LayerInfo>,
        callback: Box<dyn FnOnce(WorkAssignment) + Send>,
    },

    // From Coordination to Network
    SendMessage {
        target: NodeId,
        message: CoordinationMessage,
    },

    // From Network to Coordination
    MessageReceived {
        from: NodeId,
        message: CoordinationMessage,
    },

    // To Metrics (from any component)
    MetricEvent(MetricsEvent),

    // Status updates
    TaskStatusChanged {
        task_id: u64,
        status: TaskStatus,
    },
}
```

### 3. Service Interface Pattern

Each component exposes a clean async interface:

```rust
// butterfly-schedule/src/service.rs
#[async_trait]
pub trait SchedulingInterface: Send + Sync {
    async fn submit_task(&self, task: InferenceTask)
        -> Result<TaskId, SchedulingError>;

    async fn query_status(&self, task_id: TaskId)
        -> Result<TaskStatus, SchedulingError>;

    async fn cancel_task(&self, task_id: TaskId)
        -> Result<(), SchedulingError>;
}

// butterfly-partition/src/service.rs
#[async_trait]
pub trait PartitionInterface: Send + Sync {
    async fn compute_partitions(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
        strategy: PartitionStrategy,
    ) -> Result<Vec<ModelPartition>, PartitionError>;

    async fn get_partition_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<PartitionQuality, PartitionError>;
}

// butterfly-coordination/src/service.rs
#[async_trait]
pub trait CoordinationInterface: Send + Sync {
    async fn assign_work(
        &self,
        task_id: u64,
        partitions: Vec<ModelPartition>,
    ) -> Result<WorkAssignment, CoordinationError>;

    async fn synchronize_phase(
        &self,
        phase: Phase,
    ) -> Result<(), CoordinationError>;

    async fn handle_message(
        &self,
        from: NodeId,
        message: CoordinationMessage,
    ) -> Result<(), CoordinationError>;
}

// butterfly-comm/src/service.rs
#[async_trait]
pub trait NetworkInterface: Send + Sync {
    async fn send(
        &self,
        target: NodeId,
        message: Message,
    ) -> Result<(), NetworkError>;

    async fn broadcast(
        &self,
        message: Message,
    ) -> Result<(), NetworkError>;

    async fn transfer_tensor(
        &self,
        target: NodeId,
        tensor: TensorRef,
    ) -> Result<(), NetworkError>;
}

// butterfly-metrics/src/service.rs
pub trait MetricsInterface: Send + Sync {
    fn record_event(&self, event: MetricsEvent);

    fn query_metrics(&self, query: MetricsQuery) -> MetricsSnapshot;

    fn export_prometheus(&self) -> String;
}
```

## Complete Inference Request Lifecycle

### Sequence Diagram

```
Client          API             Scheduler       Partition       Coordination    Network         Metrics
  │              │                  │               │                │              │              │
  │─POST────────>│                  │               │                │              │              │
  │ /inference   │                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │              │──submit_task────>│               │                │              │              │
  │              │                  │               │                │              │              │
  │              │                  │─record_event─────────────────────────────────>│              │
  │              │                  │               │                │              │              │
  │              │                  │─get_partitions─>               │              │              │
  │              │                  │               │                │              │              │
  │              │                  │               │─partition()───>│              │              │
  │              │                  │               │                │              │              │
  │              │                  │               │<──partitions───│              │              │
  │              │                  │               │                │              │              │
  │              │                  │<─partitions───│                │              │              │
  │              │                  │               │                │              │              │
  │              │                  │─assign_work─────────────────>│              │              │
  │              │                  │               │                │              │              │
  │              │                  │               │                │─broadcast───>│              │
  │              │                  │               │                │  WorkAssign  │              │
  │              │                  │               │                │              │              │
  │              │                  │               │                │<─────────────│              │
  │              │                  │               │                │  ack         │              │
  │              │                  │               │                │              │              │
  │              │                  │               │                │◄─────────────┤              │
  │              │                  │               │                │  heartbeats  │              │
  │              │                  │               │                │              │              │
  │              │<─task_id────────│               │                │              │              │
  │              │                  │               │                │              │              │
  │<─202 taskId──│                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │─GET─────────>│                  │               │                │              │              │
  │ /status/123  │                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │              │──query_status───>│               │                │              │              │
  │              │                  │               │                │              │              │
  │              │<────status───────│               │                │              │              │
  │              │                  │               │                │              │              │
  │<─200 status──│                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │              │                  │   [Computation Phase]          │              │              │
  │              │                  │               │                │◄─────────────┤              │
  │              │                  │               │                │  results     │              │
  │              │                  │               │                │              │              │
  │              │                  │               │                │─────────────>│              │
  │              │                  │               │                │  consensus   │              │
  │              │                  │               │                │              │              │
  │              │                  │<──result──────────────────────│              │              │
  │              │                  │               │                │              │              │
  │              │                  │─record_event─────────────────────────────────>│              │
  │              │                  │               │                │              │              │
  │─GET─────────>│                  │               │                │              │              │
  │ /status/123  │                  │               │                │              │              │
  │              │                  │               │                │              │              │
  │<─200─────────│                  │               │                │              │              │
  │  completed   │                  │               │                │              │              │
```

### State Transitions

```
Task Lifecycle States:

  [Submitted]
      │
      │ API receives request
      │
      ▼
  [Queued] ──────────────────────> [Failed]
      │                              (validation error)
      │ Scheduler accepts
      │
      ▼
  [Partitioning]
      │
      │ Partition service computes optimal distribution
      │
      ▼
  [WorkAssigned] ─────────────────> [Failed]
      │                              (no nodes available)
      │ Coordination broadcasts assignment
      │
      ▼
  [Computing] ───────────────────> [Failed]
      │                              (node failure, consensus failure)
      │ All nodes compute their partitions
      │
      ▼
  [Aggregating]
      │
      │ Coordination runs Byzantine consensus
      │
      ▼
  [Committing] ──────────────────> [Failed]
      │                              (checkpoint failure)
      │ Results committed to checkpoints
      │
      ▼
  [Completed]
```

## Data Flow Specifications

### 1. Tensor Movement Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tensor Data Pipeline                            │
└─────────────────────────────────────────────────────────────────┘

Input Tensor (from client)
    │
    ▼
[API Layer]
    │ Deserialize from JSON/protobuf
    │ Validate shape/dtype
    │ Allocate in TensorPool (butterfly-core)
    │
    ▼
TensorRef { id, pool, shape, dtype }
    │
    ▼
[Scheduling Layer]
    │ Attach to InferenceTask
    │ Pass reference (no copy)
    │
    ▼
[Coordination Layer]
    │ Include TensorRef in WorkAssignment
    │ Serialize metadata only
    │
    ▼
[Network Layer]
    │ Transfer actual tensor data
    │ ┌─────────────────────────────┐
    │ │ For local node: shared mem  │
    │ │ For remote: serialize       │
    │ │  - Chunk into 64KB blocks   │
    │ │  - Optional compression     │
    │ │  - Send via QUIC/TCP        │
    │ └─────────────────────────────┘
    │
    ▼
[Compute Node]
    │ Deserialize into local TensorPool
    │ Execute computation
    │ Produce output tensor
    │
    ▼
[Return Path]
    │ Serialize output tensor
    │ Send via Network Layer
    │ Arrive at coordination node
    │
    ▼
[Aggregation]
    │ Multiple partial results from different nodes
    │ Byzantine consensus on correct values
    │ Combine if tensor-parallel
    │
    ▼
[Final Result]
    │ Return TensorRef to client
    │ Serialize to JSON/protobuf
    │
    ▼
Client receives result
```

### 2. Control Message Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              Control Plane Message Routing                       │
└─────────────────────────────────────────────────────────────────┘

CoordinationMessage Types:

1. WorkAssignment
   Source: Coordination (primary)
   Destination: All compute nodes (broadcast)
   Payload: { task_id, partition_assignments: Map<NodeId, PartitionInfo> }
   Reliability: Must be delivered (retry with timeout)

2. BarrierReady
   Source: Compute nodes
   Destination: Coordination (primary)
   Payload: { node_id, computation_hash }
   Purpose: Signal computation completion

3. PrePrepare / Prepare / Commit (Byzantine consensus)
   Source: Coordination (primary for PrePrepare, all for others)
   Destination: All nodes
   Payload: { result_hash, proof, signatures }
   Purpose: Agree on correct result despite failures

4. Heartbeat
   Source: All nodes (periodic)
   Destination: All nodes (broadcast)
   Payload: { node_id, phi_value, timestamp }
   Reliability: Best-effort (loss is acceptable)

5. Suspicion
   Source: Any node detecting failure
   Destination: Coordination (primary) + broadcast
   Payload: { suspected_node_id, evidence }
   Purpose: Trigger failure recovery

6. Checkpoint
   Source: Coordination (primary)
   Destination: All nodes
   Payload: { checkpoint_id, state_snapshot }
   Purpose: Enable recovery after failures

Message Router Logic:
    NetworkService::receive()
        │
        ▼
    Parse message type
        │
        ├──> WorkAssignment ──────> route_to_coordination()
        ├──> BarrierReady ────────> route_to_coordination()
        ├──> PrePrepare ──────────> route_to_coordination()
        ├──> Prepare ─────────────> route_to_coordination()
        ├──> Commit ──────────────> route_to_coordination()
        ├──> Heartbeat ───────────> route_to_failure_detector()
        ├──> Suspicion ───────────> route_to_coordination()
        └──> Checkpoint ──────────> route_to_checkpoint_manager()
```

### 3. Metrics Collection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics Aggregation Flow                      │
└─────────────────────────────────────────────────────────────────┘

Event Source          MetricsEvent              MetricsService
    │                      │                          │
    │                      │                          │
[API]                     │                          │
  │─record_event(         │                          │
  │   TaskSubmitted)─────>│                          │
    │                      │─broadcast channel───────>│
    │                      │                          │──[collect]
    │                      │                          │
[Scheduler]               │                          │
  │─record_event(         │                          │
  │   TaskEnqueued)──────>│                          │
    │                      │─broadcast channel───────>│
    │                      │                          │──[collect]
    │                      │                          │
[Partition]               │                          │
  │─record_event(         │                          │
  │   PartitionComputed)─>│                          │
    │                      │─broadcast channel───────>│
    │                      │                          │──[collect]
    │                      │                          │
[Coordination]            │                          │
  │─record_event(         │                          │
  │   PhaseTransition)───>│                          │
    │                      │─broadcast channel───────>│
    │                      │                          │──[collect]
    │                      │                          │
[Network]                 │                          │
  │─record_event(         │                          │
  │   BytesTransferred)──>│                          │
    │                      │─broadcast channel───────>│
    │                      │                          │──[collect]
    │                      │                          │
    │                      │                          ▼
    │                      │                    [Aggregator]
    │                      │                          │
    │                      │                          ├──[Histogram]
    │                      │                          │   Latencies
    │                      │                          │
    │                      │                          ├──[Counter]
    │                      │                          │   Events
    │                      │                          │
    │                      │                          ├──[Gauge]
    │                      │                          │   Utilization
    │                      │                          │
    │                      │                          └──[Summary]
    │                      │                              Percentiles
    │                      │                          │
    │                      │                          ▼
    │                      │                    [Exporter]
    │                      │                          │
    │                      │                          ├──> Prometheus
    │                      │                          │    /metrics
    │                      │                          │
    │                      │                          └──> Console
    │                      │                               logs

MetricsEvent Types:

- TaskSubmitted { task_id, timestamp }
- TaskEnqueued { task_id, priority, timestamp }
- TaskDispatched { task_id, node_ids, timestamp }
- PartitionComputed { num_partitions, quality, duration }
- PhaseTransition { from, to, timestamp }
- ConsensusReached { task_id, rounds, duration }
- MessageSent { to, size_bytes, message_type }
- MessageReceived { from, size_bytes, message_type }
- BytesTransferred { from, to, bytes, duration }
- NodeFailure { node_id, evidence, timestamp }
- TaskCompleted { task_id, latency_ms, nodes_involved }
```

## Error Propagation Paths

### Error Types Hierarchy

```rust
pub enum ButterflyError {
    // API Layer errors
    Api(ApiError),

    // Scheduling errors
    Scheduling(SchedulingError),

    // Partitioning errors
    Partition(PartitionError),

    // Coordination errors
    Coordination(CoordinationError),

    // Network errors
    Network(NetworkError),

    // System-wide errors
    SystemError(String),
}

pub enum ApiError {
    InvalidRequest(String),
    TaskNotFound(u64),
    ServiceUnavailable,
}

pub enum SchedulingError {
    QueueFull,
    TaskAlreadyExists(u64),
    CoordinationUnavailable,
}

pub enum PartitionError {
    NoNodesAvailable,
    InvalidConfiguration(String),
    InsufficientResources(String),
    OptimizationFailed(String),
}

pub enum CoordinationError {
    ConsensusTimeout,
    InsufficientNodes,
    ByzantineViolation(String),
    PhaseTransitionFailed(String),
    Internal(String),
}

pub enum NetworkError {
    ConnectionFailed(NodeId),
    SendTimeout,
    SerializationFailed(String),
    Unreachable(NodeId),
}
```

### Error Handling Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                  Error Recovery Strategies                       │
└─────────────────────────────────────────────────────────────────┘

1. Network Errors (Transient)
   Strategy: Retry with exponential backoff

   NetworkService::send()
       │
       ├─> [Attempt 1] ─X─> NetworkError::SendTimeout
       │                         │
       │                         └─> wait 100ms
       │
       ├─> [Attempt 2] ─X─> NetworkError::SendTimeout
       │                         │
       │                         └─> wait 200ms
       │
       ├─> [Attempt 3] ─X─> NetworkError::SendTimeout
       │                         │
       │                         └─> wait 400ms
       │
       └─> [Give up] ────────> Propagate error to caller
                                   │
                                   └─> CoordinationError::InsufficientNodes

2. Coordination Errors (Byzantine)
   Strategy: Consensus-based recovery

   Node failure detected
       │
       ├─> Broadcast Suspicion message
       │
       ├─> Wait for 2f+1 confirmations
       │
       ├─> Remove node from active set
       │
       ├─> Recompute partitions
       │
       └─> Resume from last checkpoint

3. Partition Errors (Configuration)
   Strategy: Fallback to simpler strategy

   TopologyAwarePartitioner::partition()
       │
       ├─> Optimization failed (timeout)
       │
       └─> Fallback to LoadBalancedPartitioner
               │
               ├─> Success ─> Return partitions
               │
               └─> Failure ─> Fallback to UniformPartitioner
                       │
                       └─> Last resort, should always succeed

4. Scheduling Errors (Resource)
   Strategy: Queue management

   Scheduler::enqueue()
       │
       ├─> Queue full
       │     │
       │     ├─> Check priority
       │     │     │
       │     │     ├─> High priority: Evict lowest priority
       │     │     └─> Low priority: Reject with SchedulingError::QueueFull
       │     │
       │     └─> Return ApiError::ServiceUnavailable to client

5. API Errors (Client)
   Strategy: Graceful HTTP responses

   submit_inference()
       │
       ├─> ValidationError ────────> 400 Bad Request
       ├─> SchedulingError::QueueFull ─> 503 Service Unavailable
       ├─> TaskAlreadyExists ──────> 409 Conflict
       └─> SystemError ────────────> 500 Internal Server Error
```

## Interface Contracts

### butterfly-api ↔ butterfly-schedule

```rust
// File: crates/butterfly-api/src/lib.rs

use butterfly_schedule::{SchedulingInterface, InferenceTask, TaskStatus};

pub struct ApiService {
    scheduler: Arc<dyn SchedulingInterface>,
}

impl ApiService {
    pub async fn submit_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, ApiError> {
        let task = InferenceTask {
            task_id: generate_task_id(),
            input: request.input,
            priority: request.priority.unwrap_or(0),
        };

        let task_id = self.scheduler
            .submit_task(task)
            .await
            .map_err(|e| ApiError::Scheduling(e))?;

        Ok(InferenceResponse {
            task_id,
            status: "submitted".to_string(),
        })
    }

    pub async fn query_status(
        &self,
        task_id: u64,
    ) -> Result<TaskStatusResponse, ApiError> {
        let status = self.scheduler
            .query_status(task_id)
            .await
            .map_err(|e| ApiError::Scheduling(e))?;

        Ok(TaskStatusResponse {
            task_id,
            status: status.to_string(),
        })
    }
}
```

### butterfly-schedule ↔ butterfly-coordination

```rust
// File: crates/butterfly-schedule/src/service.rs

use butterfly_coordination::{CoordinationInterface, WorkAssignment};
use butterfly_partition::{PartitionInterface, PartitionStrategy};

pub struct SchedulingService {
    coordinator: Arc<dyn CoordinationInterface>,
    partitioner: Arc<dyn PartitionInterface>,
    scheduler: Box<dyn Scheduler>,
}

impl SchedulingService {
    async fn dispatch_task(
        &self,
        task: InferenceTask,
    ) -> Result<(), SchedulingError> {
        // 1. Get model layer information
        let layers = self.get_model_layers(&task)?;

        // 2. Get available nodes
        let nodes = self.get_available_nodes().await?;

        // 3. Compute partitions
        let partitions = self.partitioner
            .compute_partitions(&layers, &nodes, PartitionStrategy::LoadBalanced)
            .await
            .map_err(|e| SchedulingError::PartitionFailed(e))?;

        // 4. Assign work through coordination
        let assignment = self.coordinator
            .assign_work(task.task_id, partitions)
            .await
            .map_err(|e| SchedulingError::CoordinationFailed(e))?;

        Ok(())
    }
}
```

### butterfly-coordination ↔ butterfly-comm

```rust
// File: crates/butterfly-coordination/src/service.rs

use butterfly_comm::{NetworkInterface, Message};
use butterfly_core::NodeId;

pub struct CoordinationService {
    network: Arc<dyn NetworkInterface>,
    coordinator: Arc<RwLock<DistributedCoordinator>>,
}

impl CoordinationService {
    pub async fn broadcast_work_assignment(
        &self,
        assignment: WorkAssignment,
    ) -> Result<(), CoordinationError> {
        let message = Message::Coordination(
            CoordinationMessage::WorkAssignment(assignment)
        );

        self.network
            .broadcast(message)
            .await
            .map_err(|e| CoordinationError::NetworkFailed(e))?;

        Ok(())
    }

    pub async fn handle_incoming_messages(&self) {
        loop {
            match self.network.receive().await {
                Ok(Message::Coordination(coord_msg)) => {
                    let coord = self.coordinator.write().await;
                    if let Err(e) = coord.handle_message(coord_msg).await {
                        error!("Failed to handle coordination message: {}", e);
                    }
                }
                Ok(_) => {} // Ignore non-coordination messages
                Err(e) => {
                    error!("Network receive error: {}", e);
                }
            }
        }
    }
}
```

### All Components ↔ butterfly-metrics

```rust
// File: crates/butterfly-metrics/src/service.rs

use tokio::sync::broadcast;

pub struct MetricsService {
    event_tx: broadcast::Sender<MetricsEvent>,
    collector: Arc<RwLock<Box<dyn MetricsCollector>>>,
}

impl MetricsService {
    pub fn new(config: MetricsConfig) -> Self {
        let (event_tx, mut event_rx) = broadcast::channel(10000);
        let collector = Arc::new(RwLock::new(
            Box::new(InMemoryCollector::new()) as Box<dyn MetricsCollector>
        ));

        // Spawn aggregator task
        let collector_clone = Arc::clone(&collector);
        tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                let mut c = collector_clone.write().await;
                c.process_event(event);
            }
        });

        Self { event_tx, collector }
    }

    pub fn subscriber(&self) -> broadcast::Receiver<MetricsEvent> {
        self.event_tx.subscribe()
    }
}

// Each component holds a MetricsService reference
pub trait Component {
    fn record_metric(&self, event: MetricsEvent) {
        // Non-blocking send
        let _ = self.metrics_service.event_tx.send(event);
    }
}
```

## Initialization Order and Lifetime Management

### System Startup Sequence

```rust
// File: crates/butterfly-core/src/system.rs

pub async fn initialize_butterfly_system(
    config: SystemConfig,
) -> Result<ButterflySystem, SystemError> {
    // Phase 1: Metrics (no dependencies)
    info!("Initializing metrics service...");
    let metrics_service = Arc::new(MetricsService::new(config.metrics.clone()));

    // Phase 2: Network (depends on metrics)
    info!("Initializing network service...");
    let network_backend = create_network_backend(&config.network)?;
    let network_service = Arc::new(NetworkService::new(
        network_backend,
        Arc::clone(&metrics_service),
    ));

    // Start network message receiver loop
    let network_clone = Arc::clone(&network_service);
    tokio::spawn(async move {
        network_clone.run_receiver_loop().await;
    });

    // Phase 3: Partition (depends on metrics)
    info!("Initializing partition service...");
    let partition_strategy = create_partition_strategy(&config.partition)?;
    let partition_service = Arc::new(PartitionService::new(
        partition_strategy,
        Arc::clone(&metrics_service),
    ));

    // Phase 4: Coordination (depends on network, partition, metrics)
    info!("Initializing coordination service...");
    let coordination_service = Arc::new(CoordinationService::new(
        config.coordination.clone(),
        Arc::clone(&network_service),
        Arc::clone(&partition_service),
        Arc::clone(&metrics_service),
    ));

    // Start coordination message handler
    let coord_clone = Arc::clone(&coordination_service);
    tokio::spawn(async move {
        coord_clone.run_message_handler().await;
    });

    // Phase 5: Scheduler (depends on coordination, partition, metrics)
    info!("Initializing scheduling service...");
    let scheduler_impl = create_scheduler(&config.scheduling)?;
    let scheduling_service = Arc::new(SchedulingService::new(
        scheduler_impl,
        Arc::clone(&coordination_service),
        Arc::clone(&partition_service),
        Arc::clone(&metrics_service),
    ));

    // Start scheduler dispatch loop
    let sched_clone = Arc::clone(&scheduling_service);
    tokio::spawn(async move {
        sched_clone.run_dispatch_loop().await;
    });

    // Phase 6: API (depends on scheduling, metrics)
    info!("Initializing API service...");
    let api_service = Arc::new(ApiService::new(
        Arc::clone(&scheduling_service),
        Arc::clone(&metrics_service),
    ));

    // Phase 7: Start HTTP server
    info!("Starting HTTP server on {}...", config.api.bind_address);
    let server_handle = tokio::spawn(async move {
        api_service.serve(config.api.bind_address).await
    });

    info!("Butterfly system initialized successfully");

    Ok(ButterflySystem {
        scheduling_service,
        partition_service,
        coordination_service,
        network_service,
        metrics_service,
        api_service,
        server_handle: Some(server_handle),
        config,
    })
}
```

### Graceful Shutdown

```rust
impl ButterflySystem {
    pub async fn shutdown(mut self) -> Result<(), SystemError> {
        info!("Shutting down Butterfly system...");

        // Phase 1: Stop accepting new requests
        info!("Stopping API server...");
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            // Wait for existing requests to complete (timeout: 30s)
            tokio::time::timeout(
                Duration::from_secs(30),
                self.api_service.drain_requests()
            ).await?;
        }

        // Phase 2: Drain scheduler queue
        info!("Draining scheduler queue...");
        self.scheduling_service.stop_accepting().await;
        tokio::time::timeout(
            Duration::from_secs(60),
            self.scheduling_service.wait_for_completion()
        ).await?;

        // Phase 3: Complete coordination phase
        info!("Completing coordination phase...");
        self.coordination_service.finish_current_phase().await?;

        // Phase 4: Checkpoint state
        info!("Creating final checkpoint...");
        self.coordination_service.create_checkpoint().await?;

        // Phase 5: Close network connections
        info!("Closing network connections...");
        self.network_service.shutdown().await?;

        // Phase 6: Export final metrics
        info!("Exporting final metrics...");
        let final_metrics = self.metrics_service.export_all().await;
        info!("Final metrics: {:?}", final_metrics);

        info!("Butterfly system shutdown complete");
        Ok(())
    }
}
```

## Testing Strategy

### Integration Test Architecture

```rust
// File: tests/integration/test_full_pipeline.rs

#[tokio::test]
async fn test_complete_inference_pipeline() {
    // Setup: Create in-memory test system
    let config = SystemConfig::test_defaults();
    let system = initialize_butterfly_system(config).await.unwrap();

    // Create test client
    let client = TestClient::new(system.api_address());

    // Submit inference request
    let response = client.post("/inference")
        .json(&InferenceRequest {
            input: vec![1.0, 2.0, 3.0],
            priority: Some(1),
        })
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 202);
    let task_response: InferenceResponse = response.json().await.unwrap();

    // Poll for completion
    let mut attempts = 0;
    loop {
        let status = client.get(&format!("/status/{}", task_response.task_id))
            .send()
            .await
            .unwrap();

        let status_response: TaskStatusResponse = status.json().await.unwrap();

        if status_response.status == "completed" {
            break;
        }

        attempts += 1;
        assert!(attempts < 100, "Task did not complete in time");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Verify metrics were collected
    let metrics = system.metrics_service.export_all().await;
    assert!(metrics.total_tasks_completed >= 1);

    // Cleanup
    system.shutdown().await.unwrap();
}
```

### Mock Implementations

```rust
// File: crates/butterfly-comm/src/mock.rs

pub struct MockNetworkService {
    sent_messages: Arc<RwLock<Vec<(NodeId, Message)>>>,
    incoming_queue: Arc<RwLock<VecDeque<Message>>>,
}

#[async_trait]
impl NetworkInterface for MockNetworkService {
    async fn send(&self, target: NodeId, message: Message)
        -> Result<(), NetworkError>
    {
        let mut sent = self.sent_messages.write().await;
        sent.push((target, message));
        Ok(())
    }

    async fn receive(&self) -> Result<Message, NetworkError> {
        let mut queue = self.incoming_queue.write().await;
        queue.pop_front()
            .ok_or(NetworkError::NoMessages)
    }

    async fn broadcast(&self, message: Message)
        -> Result<(), NetworkError>
    {
        // Simulate broadcast by sending to all nodes
        for node_id in 0..5 {
            self.send(NodeId(node_id), message.clone()).await?;
        }
        Ok(())
    }
}

impl MockNetworkService {
    pub fn inject_message(&self, message: Message) {
        let mut queue = self.incoming_queue.blocking_write();
        queue.push_back(message);
    }

    pub fn get_sent_messages(&self) -> Vec<(NodeId, Message)> {
        self.sent_messages.blocking_read().clone()
    }
}
```

## Configuration Management

```rust
// File: crates/butterfly-core/src/config.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub api: ApiConfig,
    pub scheduling: SchedulingConfig,
    pub partition: PartitionConfig,
    pub coordination: CoordinationConfig,
    pub network: NetworkConfig,
    pub metrics: MetricsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub bind_address: String,
    pub max_request_size_mb: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    pub scheduler_type: String, // "fifo", "priority", "fair"
    pub max_queue_size: usize,
    pub dispatch_batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub default_strategy: String, // "uniform", "load_balanced", "topology_aware"
    pub cache_size: usize,
    pub recompute_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    pub node_id: u64,
    pub cluster_size: usize,
    pub max_byzantine: usize,
    pub phase_timeout_ms: u64,
    pub checkpoint_interval: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub backend_type: String, // "libp2p", "tcp", "local"
    pub listen_address: String,
    pub peer_addresses: Vec<String>,
    pub max_connections: usize,
    pub send_timeout_ms: u64,
    pub retry_attempts: usize,
    pub retry_backoff_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub collector_type: String, // "in_memory", "prometheus"
    pub export_interval_secs: u64,
    pub retention_duration_secs: u64,
}

impl SystemConfig {
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: SystemConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn test_defaults() -> Self {
        Self {
            api: ApiConfig {
                bind_address: "127.0.0.1:8080".to_string(),
                max_request_size_mb: 100,
                request_timeout_secs: 30,
            },
            scheduling: SchedulingConfig {
                scheduler_type: "fifo".to_string(),
                max_queue_size: 1000,
                dispatch_batch_size: 10,
            },
            partition: PartitionConfig {
                default_strategy: "load_balanced".to_string(),
                cache_size: 100,
                recompute_threshold: 0.1,
            },
            coordination: CoordinationConfig {
                node_id: 0,
                cluster_size: 1,
                max_byzantine: 0,
                phase_timeout_ms: 5000,
                checkpoint_interval: 10,
            },
            network: NetworkConfig {
                backend_type: "local".to_string(),
                listen_address: "127.0.0.1:9000".to_string(),
                peer_addresses: vec![],
                max_connections: 100,
                send_timeout_ms: 1000,
                retry_attempts: 3,
                retry_backoff_ms: 100,
            },
            metrics: MetricsConfig {
                collector_type: "in_memory".to_string(),
                export_interval_secs: 60,
                retention_duration_secs: 3600,
            },
        }
    }
}
```

## Summary of Required Code Changes

### butterfly-api
- Add `SchedulingInterface` dependency injection
- Implement error mapping from `SchedulingError` to HTTP status codes
- Add metrics event recording
- Create status registry for tracking task states

### butterfly-schedule
- Add `CoordinationInterface` and `PartitionInterface` dependencies
- Implement dispatch loop that coordinates with partition and coordination services
- Add async task management
- Integrate metrics recording

### butterfly-partition
- Wrap partitioning logic in `PartitionService` with async interface
- Add caching layer for computed partitions
- Integrate metrics for partition computation
- Add dynamic strategy selection

### butterfly-coordination
- Wrap `DistributedCoordinator` in `CoordinationService`
- Add `NetworkInterface` dependency for message passing
- Add `PartitionInterface` dependency for partition queries
- Implement message routing and phase management
- Integrate metrics for consensus and phase transitions

### butterfly-comm
- Create `NetworkService` wrapper around `CommunicationBackend`
- Implement message router with handler registration
- Add reliable delivery with retry logic
- Implement specialized tensor transfer protocol
- Integrate metrics for network events

### butterfly-metrics
- Create broadcast channel based event bus
- Implement background aggregator task
- Add Prometheus exporter
- Create query interface for other components

### butterfly-core
- Add `SystemConfig` types
- Add error type hierarchy
- Add service trait definitions
- Add system initialization and shutdown logic
- Add `StatusRegistry` for shared task state

## Next Steps for Implementation

1. Define service traits in butterfly-core
2. Implement metric event infrastructure
3. Create NetworkService wrapper
4. Create PartitionService wrapper
5. Create CoordinationService wrapper
6. Create SchedulingService wrapper
7. Update ApiService with dependencies
8. Implement ButterflySystem initialization
9. Write integration tests
10. Performance profiling and optimization
