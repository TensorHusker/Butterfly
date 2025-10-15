# Butterfly State Machine Design

## Executive Summary

This document defines the complete state machine architecture for the Butterfly distributed inference system. It establishes formal state models that ensure **mechanical fitness** (correct operation) and **narrative coherence** (understandable system behavior).

**Core Principle**: Every component's state tells a story about what the system is doing, where it's been, and where it can go next. State transitions are not just technical operations—they are chapters in the system's narrative of transforming inputs into outputs while maintaining safety and liveness.

---

## 1. System-Level State Machine

The **System** represents the entire Butterfly cluster as a single entity with observable behavior.

### 1.1 System States

```
┌──────────────┐
│ UNINITIALIZED│ ──────┐
└──────────────┘       │
                       │ bootstrap()
                       ▼
              ┌─────────────────┐
              │  BOOTSTRAPPING  │
              └─────────────────┘
                       │
                       │ quorum_established()
                       ▼
              ┌─────────────────┐      ┌──────────────┐
         ┌───│      READY       │◄─────│   DEGRADED   │
         │   └─────────────────┘      └──────────────┘
         │            │                        ▲
         │            │ begin_inference()      │
         │            ▼                        │
         │   ┌─────────────────┐              │
         │   │   COMPUTING     │──────────────┘
         │   └─────────────────┘   failure_detected()
         │            │
         │            │ all_phases_complete()
         │            ▼
         │   ┌─────────────────┐
         │   │   COMMITTING    │
         │   └─────────────────┘
         │            │
         │            │ result_committed()
         └────────────┘
                      │
                      │ shutdown_initiated()
                      ▼
              ┌─────────────────┐
              │    SHUTDOWN     │
              └─────────────────┘
                      │
                      │ cleanup_complete()
                      ▼
              ┌─────────────────┐
              │   TERMINATED    │
              └─────────────────┘
```

### 1.2 State Descriptions

**UNINITIALIZED**
- **Meaning**: System exists but has not started initialization
- **Entry Conditions**: System creation
- **Activities**: None (dormant state)
- **Exit Trigger**: Explicit bootstrap command
- **Invariants**:
  - No nodes registered
  - No coordinator elected
  - No network connections active

**BOOTSTRAPPING**
- **Meaning**: System is discovering nodes and establishing initial cluster
- **Entry Conditions**: `bootstrap()` called from UNINITIALIZED
- **Activities**:
  - Node discovery via service registry
  - Coordinator election (Raft-based)
  - Initial health checks
  - Configuration distribution
- **Exit Trigger**: Quorum (2f+1 nodes) operational and coordinator elected
- **Invariants**:
  - 0 < operational_nodes < quorum_size (incomplete cluster)
  - At most one coordinator being elected
  - All nodes in INITIALIZING state

**READY**
- **Meaning**: System is operational and waiting for inference requests
- **Entry Conditions**:
  - From BOOTSTRAPPING: Quorum established
  - From COMMITTING: Previous inference completed
  - From DEGRADED: Recovery completed, quorum restored
- **Activities**:
  - Monitoring node health
  - Accepting new inference requests
  - Load balancing and resource tracking
- **Exit Trigger**: Inference request received
- **Invariants**:
  - operational_nodes >= quorum_size
  - Exactly one coordinator
  - All operational nodes in READY state
  - No in-flight computations

**COMPUTING**
- **Meaning**: System is executing distributed inference across nodes
- **Entry Conditions**: `begin_inference()` from READY
- **Activities**:
  - Work assignment distribution
  - Pipelined computation across nodes
  - Intermediate result forwarding
  - Checkpoint creation (every K tokens)
  - Progress monitoring
- **Exit Trigger**: All phases complete or failure detected
- **Invariants**:
  - operational_nodes >= quorum_size
  - Valid work assignment exists
  - All nodes in {COMPUTING, AGGREGATING} states
  - Computation progressing (heartbeat validation)

**DEGRADED**
- **Meaning**: System detected failures but maintains quorum—operating with reduced capacity
- **Entry Conditions**: Failure detected during any operational state
- **Activities**:
  - Isolating failed nodes
  - Triggering recovery protocols
  - Redistributing work from failed nodes
  - Continuing computation on healthy nodes (if possible)
- **Exit Trigger**:
  - Recovery complete → READY
  - Quorum lost → SHUTDOWN
- **Invariants**:
  - quorum_size <= operational_nodes < total_nodes
  - 0 < confirmed_failures <= max_byzantine
  - Recovery protocol active
  - System making progress (possibly slower)

**COMMITTING**
- **Meaning**: System performing Byzantine agreement on final result
- **Entry Conditions**: All computation phases complete
- **Activities**:
  - Result validation (PRE-PREPARE phase)
  - Voting on correctness (PREPARE phase)
  - Finalizing commitment (COMMIT phase)
  - Persisting result to storage
- **Exit Trigger**: 2f+1 nodes commit result
- **Invariants**:
  - Result hash agreed by quorum
  - All proofs validated
  - Result persisted before state exit

**SHUTDOWN**
- **Meaning**: System is gracefully terminating
- **Entry Conditions**:
  - Explicit shutdown command
  - Quorum permanently lost
  - Unrecoverable system error
- **Activities**:
  - Completing in-flight inference (if possible)
  - Persisting final checkpoints
  - Closing network connections
  - Releasing resources
- **Exit Trigger**: All cleanup tasks complete
- **Invariants**:
  - No new work accepted
  - Existing work completing or cancelled
  - Checkpoints being persisted

**TERMINATED**
- **Meaning**: System has shut down completely
- **Entry Conditions**: Cleanup complete from SHUTDOWN
- **Activities**: None (final state)
- **Exit Trigger**: None (absorbing state)
- **Invariants**:
  - All nodes disconnected
  - All resources released
  - Final state persisted

### 1.3 System State Transition Table

| From State      | To State      | Trigger                    | Guard Conditions                     | Actions                                      |
|-----------------|---------------|----------------------------|--------------------------------------|----------------------------------------------|
| UNINITIALIZED   | BOOTSTRAPPING | `bootstrap()`              | None                                 | Start node discovery, initiate election      |
| BOOTSTRAPPING   | READY         | `quorum_established()`     | nodes >= quorum_size                 | Finalize cluster config, notify all nodes    |
| READY           | COMPUTING     | `begin_inference()`        | Valid request, no in-flight work     | Create work assignment, distribute to nodes  |
| COMPUTING       | DEGRADED      | `failure_detected()`       | failure_count > 0                    | Isolate failed nodes, start recovery         |
| COMPUTING       | COMMITTING    | `all_phases_complete()`    | All nodes report completion          | Collect results, start Byzantine agreement   |
| COMMITTING      | READY         | `result_committed()`       | 2f+1 votes received                  | Persist result, increment epoch, clear state |
| DEGRADED        | READY         | `recovery_complete()`      | quorum restored                      | Resume normal operation                      |
| DEGRADED        | SHUTDOWN      | `quorum_lost()`            | nodes < quorum_size                  | Initiate graceful shutdown                   |
| COMPUTING       | SHUTDOWN      | `shutdown_initiated()`     | Explicit command                     | Cancel work, save checkpoints                |
| READY           | SHUTDOWN      | `shutdown_initiated()`     | Explicit command                     | Disconnect nodes cleanly                     |
| SHUTDOWN        | TERMINATED    | `cleanup_complete()`       | All resources released               | Final state persistence                      |

### 1.4 Forbidden Transitions

These transitions violate safety or liveness properties:

| From            | To            | Why Forbidden                                                      |
|-----------------|---------------|--------------------------------------------------------------------|
| UNINITIALIZED   | READY         | Skips cluster formation—no coordinator, no quorum                  |
| COMPUTING       | READY         | Skips result commitment—would lose computation results             |
| DEGRADED        | COMPUTING     | Cannot start new work during recovery—violates safety              |
| COMMITTING      | DEGRADED      | Agreement phase must complete atomically—partial commit forbidden  |
| TERMINATED      | Any           | Terminal state—resurrection impossible without reboot              |
| Any             | BOOTSTRAPPING | Cannot re-bootstrap running system—must shutdown first             |

---

## 2. Component State Machines

### 2.1 Coordinator State Machine

The coordinator orchestrates the cluster and maintains global system state.

#### States

```
┌─────────────┐
│  ELECTED    │ ───────┐
└─────────────┘        │ initialize_cluster()
                       ▼
              ┌────────────────┐
              │   MONITORING   │◄──────┐
              └────────────────┘       │
                       │               │
                       │ request_received()
                       ▼               │
              ┌────────────────┐       │
              │  ASSIGNING     │       │
              └────────────────┘       │
                       │               │
                       │ assignments_acked()
                       ▼               │
              ┌────────────────┐       │
              │ COORDINATING   │       │
              └────────────────┘       │
                       │               │
                       │ computation_complete()
                       ▼               │
              ┌────────────────┐       │
              │  AGGREGATING   │       │
              └────────────────┘       │
                       │               │
                       │ results_collected()
                       ▼               │
              ┌────────────────┐       │
              │   COMMITTING   │       │
              └────────────────┘       │
                       │               │
                       │ commit_finalized()
                       └───────────────┘

                       │ failure_detected()
                       ▼
              ┌────────────────┐
              │   RECOVERING   │
              └────────────────┘
                       │
                       │ recovery_complete()
                       └──────────► MONITORING
```

#### Coordinator State Invariants

**ELECTED**
- Leader term established
- No other active coordinators
- Not yet handling requests

**MONITORING**
- Actively tracking node health (heartbeats every 100ms)
- No in-flight inference
- Ready to accept requests
- worker_count >= quorum_size

**ASSIGNING**
- Valid work assignment created
- Assignments being distributed
- Awaiting acknowledgments from 2f+1 nodes
- Timeout: 5 seconds

**COORDINATING**
- All nodes have acknowledged assignments
- Computation in progress
- Monitoring progress via intermediate checkpoints
- No global synchronization (pipelined execution)

**AGGREGATING**
- Computation phases complete
- Collecting results from final-layer nodes
- Awaiting quorum (2f+1) of matching results
- Timeout: computed_expected_time + 3σ

**COMMITTING**
- Quorum of results received
- Byzantine agreement protocol active
- Voting on result correctness
- Persisting committed result

**RECOVERING**
- One or more node failures detected
- Reassigning work to healthy nodes
- Restoring from latest checkpoint
- Coordinating recovery synchronization

#### Coordinator Responsibilities by State

| State         | Primary Activities                                                           |
|---------------|------------------------------------------------------------------------------|
| ELECTED       | Establish leadership, load cluster configuration                             |
| MONITORING    | Health checks, metric collection, request queuing                            |
| ASSIGNING     | Partition model, compute assignments, broadcast work, collect acks           |
| COORDINATING  | Monitor progress, handle checkpoints, forward intermediate results           |
| AGGREGATING   | Collect final results, validate consistency, prepare for agreement           |
| COMMITTING    | Byzantine agreement (PRE-PREPARE, PREPARE, COMMIT), result persistence       |
| RECOVERING    | Detect failures, isolate nodes, redistribute work, restore from checkpoints  |

### 2.2 Worker Node State Machine

Workers execute assigned computation and report status.

#### States

```
┌──────────────┐
│ INITIALIZED  │
└──────────────┘
       │
       │ coordinator_connected()
       ▼
┌──────────────┐
│   JOINED     │◄────────────────┐
└──────────────┘                 │
       │                         │
       │ model_partition_loaded()│
       ▼                         │
┌──────────────┐                 │
│    READY     │                 │
└──────────────┘                 │
       │                         │
       │ work_assigned()         │
       ▼                         │
┌──────────────┐                 │
│  COMPUTING   │                 │
└──────────────┘                 │
       │                         │
       │ local_computation_done()│
       ▼                         │
┌──────────────┐                 │
│ AGGREGATING  │                 │
└──────────────┘                 │
       │                         │
       │ barrier_released()      │
       ▼                         │
┌──────────────┐                 │
│  COMMITTING  │                 │
└──────────────┘                 │
       │                         │
       │ commit_finalized()      │
       └─────────────────────────┘

       │ failure_detected()
       ▼
┌──────────────┐
│   SUSPECT    │
└──────────────┘
       │
       │ confirm_failure()
       ▼
┌──────────────┐
│    FAILED    │
└──────────────┘
```

#### Worker State Invariants

**INITIALIZED**
- Process started, resources allocated
- No coordinator connection
- Model not loaded

**JOINED**
- Connected to coordinator
- Registered in cluster
- Awaiting model partition assignment
- Sending heartbeats every 100ms

**READY**
- Model partition loaded and validated
- Memory allocated for computation
- Waiting for work assignment
- Advertising available capacity

**COMPUTING**
- Active computation on assigned layers
- Receiving input from dependencies
- Forwarding output to consumers
- Creating checkpoints every K tokens
- Reporting progress to coordinator

**AGGREGATING**
- Local computation complete
- Waiting at barrier for peers
- Final result ready
- Awaiting coordinator signal to proceed

**COMMITTING**
- Participating in Byzantine agreement
- Voting on result correctness
- Persisting local state
- Preparing for next inference

**SUSPECT**
- Other nodes suspect this node has failed
- Still attempting to complete work
- Increased heartbeat frequency
- Attempting to prove liveness

**FAILED**
- Confirmed failure by quorum
- No longer part of active computation
- May be recovering or permanently removed
- Not accepting new work

#### Worker Responsibilities by State

| State        | Primary Activities                                                    |
|--------------|-----------------------------------------------------------------------|
| INITIALIZED  | Load configuration, allocate resources, discover coordinator          |
| JOINED       | Download model partition, perform validation, report capabilities     |
| READY        | Heartbeat transmission, resource monitoring, awaiting assignment      |
| COMPUTING    | Execute inference, forward tensors, create checkpoints, report status |
| AGGREGATING  | Barrier synchronization, hold result, validate against peers          |
| COMMITTING   | Vote on result, persist state, cleanup computation resources          |
| SUSPECT      | Aggressive heartbeats, prove liveness, attempt recovery               |
| FAILED       | Attempt reconnection, or graceful shutdown                            |

### 2.3 Task State Machine

Individual inference requests have their own lifecycle.

#### States

```
┌──────────────┐
│   QUEUED     │
└──────────────┘
       │
       │ coordinator_accepts()
       ▼
┌──────────────┐
│  SCHEDULED   │
└──────────────┘
       │
       │ assignment_created()
       ▼
┌──────────────┐
│  DISPATCHED  │
└──────────────┘
       │
       │ workers_acknowledged()
       ▼
┌──────────────┐
│  EXECUTING   │
└──────────────┘
       │
       ├───────► ┌──────────────┐
       │         │   FAILED     │ (unrecoverable error)
       │         └──────────────┘
       │
       │ all_phases_complete()
       ▼
┌──────────────┐
│  VALIDATING  │
└──────────────┘
       │
       │ quorum_agrees()
       ▼
┌──────────────┐
│  COMPLETED   │
└──────────────┘
```

#### Task State Invariants

**QUEUED**
- Request validated and accepted
- Awaiting scheduling
- Position in priority queue
- Timeout: configurable (default 30s)

**SCHEDULED**
- Assigned to specific epoch
- Resources reserved
- Work assignment being created
- Timeout: 5s

**DISPATCHED**
- Assignment sent to workers
- Awaiting acknowledgments
- Partial acks acceptable if quorum reached
- Timeout: 5s

**EXECUTING**
- Computation in progress across cluster
- Progress tracked via checkpoints
- Intermediate results flowing
- Timeout: model_specific (e.g., layers * time_per_layer + 3σ)

**VALIDATING**
- Byzantine agreement in progress
- Results being cross-validated
- Votes being collected
- Timeout: 10s (3 RTT for PBFT)

**COMPLETED**
- Result committed by quorum
- Response returned to client
- Metrics recorded
- Resources released

**FAILED**
- Unrecoverable error occurred
- Retries exhausted
- Error reported to client
- Cleanup initiated

### 2.4 Connection State Machine

Network connections between nodes have managed lifecycles.

#### States

```
┌──────────────┐
│  CONNECTING  │
└──────────────┘
       │
       │ handshake_complete()
       ▼
┌──────────────┐
│ ESTABLISHED  │◄────────┐
└──────────────┘         │
       │                 │
       │ data_flowing()  │ keepalive_ok()
       ▼                 │
┌──────────────┐         │
│    ACTIVE    │─────────┘
└──────────────┘
       │
       │ no_activity_timeout()
       ▼
┌──────────────┐
│    IDLE      │
└──────────────┘
       │
       │ activity_resumed()
       └──────────────────► ACTIVE
       │
       │ keepalive_failed()
       ▼
┌──────────────┐
│  SUSPECTED   │
└──────────────┘
       │
       │ reconnect() / timeout()
       ▼
┌──────────────┐
│  TERMINATED  │
└──────────────┘
```

#### Connection State Properties

**CONNECTING**
- TCP/QUIC connection being established
- TLS handshake (if enabled)
- Protocol version negotiation
- Timeout: 5s

**ESTABLISHED**
- Connection ready for data
- No active message flow yet
- Keepalive starting
- Initial state after handshake

**ACTIVE**
- Messages being exchanged
- Normal operation
- Latency monitoring
- Flow control active

**IDLE**
- No messages for idle_timeout (default: 30s)
- Keepalive maintaining connection
- Resources may be reclaimed
- Quick resume on activity

**SUSPECTED**
- Keepalive failures detected
- Connection possibly broken
- Attempting recovery
- Failure detector notified

**TERMINATED**
- Connection closed (graceful or forced)
- Resources released
- Reconnection possible via new CONNECTING state

### 2.5 Partition State Machine

Model partitions loaded on workers have managed states.

#### States

```
┌──────────────┐
│  UNLOADED    │
└──────────────┘
       │
       │ download_initiated()
       ▼
┌──────────────┐
│ DOWNLOADING  │
└──────────────┘
       │
       │ download_complete()
       ▼
┌──────────────┐
│  VALIDATING  │
└──────────────┘
       │
       │ validation_passed()
       ▼
┌──────────────┐
│   LOADED     │
└──────────────┘
       │
       │ warmup_initiated()
       ▼
┌──────────────┐
│   WARMING    │ (compile kernels, allocate memory)
└──────────────┘
       │
       │ warmup_complete()
       ▼
┌──────────────┐
│    READY     │◄────────┐
└──────────────┘         │
       │                 │
       │ assigned()      │
       ▼                 │
┌──────────────┐         │
│   IN_USE     │         │
└──────────────┘         │
       │                 │
       │ released()      │
       └─────────────────┘
       │
       │ evict()
       ▼
┌──────────────┐
│   EVICTED    │
└──────────────┘
```

#### Partition State Characteristics

**UNLOADED**
- Partition metadata known but weights not loaded
- No memory allocated

**DOWNLOADING**
- Fetching model weights from storage
- Progress tracked (bytes downloaded / total)
- Checksum validation on chunks
- Retries on network errors

**VALIDATING**
- Verifying checksum of complete partition
- Checking format compatibility
- Ensuring completeness

**LOADED**
- Weights in memory
- Not yet optimized for execution
- Occupying RAM but not GPU memory

**WARMING**
- Compiling CUDA/Metal kernels
- Allocating GPU memory
- Running test inference
- Measuring baseline performance

**READY**
- Fully optimized and ready for inference
- Can be assigned to tasks immediately
- Performance characteristics known

**IN_USE**
- Actively computing on assigned task
- Cannot be evicted
- Performance being monitored

**EVICTED**
- Removed from memory (LRU policy)
- Metadata retained for fast reload
- Can transition back to DOWNLOADING if needed

---

## 3. State Coherence Across Components

### 3.1 Vertical State Alignment

The system enforces **vertical coherence**: component states must align with system state.

| System State  | Allowed Coordinator States      | Allowed Worker States           | Allowed Task States         |
|---------------|---------------------------------|---------------------------------|-----------------------------|
| UNINITIALIZED | None                            | None                            | None                        |
| BOOTSTRAPPING | ELECTED                         | INITIALIZED, JOINED             | None                        |
| READY         | MONITORING                      | READY                           | QUEUED, SCHEDULED           |
| COMPUTING     | ASSIGNING, COORDINATING         | COMPUTING, AGGREGATING          | DISPATCHED, EXECUTING       |
| DEGRADED      | RECOVERING                      | READY, COMPUTING, AGGREGATING, SUSPECT | EXECUTING, VALIDATING |
| COMMITTING    | COMMITTING                      | COMMITTING                      | VALIDATING                  |
| SHUTDOWN      | Any (draining)                  | Any (draining)                  | FAILED or COMPLETED         |
| TERMINATED    | None                            | None                            | None                        |

**Enforcement**: State transition handlers check vertical alignment and reject invalid transitions.

### 3.2 Horizontal State Synchronization

Components at the same level must maintain **horizontal coherence**.

#### Worker Node Synchronization Rules

1. **Phase Barriers**: All workers must reach barrier before advancing
   - Implemented via `BarrierReady` / `BarrierRelease` messages
   - Coordinator tracks: `phase_completed_count >= quorum_size`

2. **Epoch Consistency**: All workers operate on same epoch
   - Workers reject work assignments from wrong epoch
   - Coordinator increments epoch only after all workers commit

3. **Computation Pipelining**: Workers can be in different phases
   - Early-layer workers may be in COMPUTING
   - Late-layer workers still in READY
   - This is valid: horizontal asynchrony within computation pipeline
   - **Constraint**: No worker can skip phases

#### Task Execution Synchronization

All tasks in same batch must:
- Start in same epoch
- Progress through phases together
- Complete or fail atomically (batch-level)

### 3.3 State Invariants (Global Properties)

These properties MUST hold at all times:

**Safety Invariants** (nothing bad happens):

1. **Single Coordinator**: At most one coordinator in MONITORING/ASSIGNING/COORDINATING state
2. **Quorum Operational**: `operational_workers >= 2f+1` OR system in SHUTDOWN
3. **Epoch Monotonicity**: Epoch numbers only increase, never decrease
4. **Result Determinism**: For same (model, input, epoch), all honest nodes produce same result
5. **No Partial Commits**: Either all 2f+1 nodes commit result, or none do
6. **Checkpoint Causality**: Checkpoint K+1 causally depends on checkpoint K

**Liveness Invariants** (something good eventually happens):

1. **Progress Guarantee**: If system in COMPUTING and no failures, eventually reaches COMMITTING
2. **Failure Detection**: Failed nodes detected within 3 * heartbeat_interval (with high probability)
3. **Recovery Termination**: Recovery completes in bounded time: T_detect + T_checkpoint + K * T_layer
4. **Request Completion**: Accepted requests eventually reach COMPLETED or FAILED state
5. **Barrier Release**: If quorum nodes reach barrier, barrier eventually releases

**Consistency Invariants** (state makes sense):

1. **State Alignment**: Component states align with system state (vertical coherence)
2. **Phase Ordering**: Phases occur in order: Assignment → Computation → Aggregation → Commitment
3. **Work Assignment Validity**: Assigned layers form valid partition of model
4. **Dependency Respect**: Node only computes after dependencies satisfied
5. **Connection Liveness**: ACTIVE connections have heartbeats within timeout

### 3.4 Race Condition Prevention

Critical sections protected by state machine atomicity:

#### Coordinator Leader Election
- **Race**: Multiple nodes become coordinator simultaneously
- **Prevention**: Raft consensus ensures single leader per term
- **State Guard**: Only node in term T can act as coordinator

#### Phase Advancement
- **Race**: Nodes advance phases at different times
- **Prevention**: Barrier synchronization with quorum requirement
- **State Guard**: Coordinator holds `phase_completed` lock during count check

#### Failure Detection vs. Computation Completion
- **Race**: Node marked failed after completing work
- **Prevention**: Timestamps on messages; late completions from suspected nodes still accepted if valid
- **State Guard**: Work completion checked before failure confirmation

#### Checkpoint Read/Write
- **Race**: Reading checkpoint while it's being written
- **Prevention**: Atomic file operations; use temp file + rename
- **State Guard**: Checkpoints immutable after creation

### 3.5 Deadlock Avoidance

Potential deadlock scenarios and solutions:

#### Circular Dependency Deadlock
- **Scenario**: Node A waits for B, B waits for C, C waits for A
- **Avoidance**: Topological ordering of dependencies; dependency graph acyclicity checked during assignment
- **Detection**: Work assignment validation rejects cyclic graphs

#### Barrier Deadlock
- **Scenario**: Some nodes wait at barrier, others failed without detection
- **Avoidance**: Barrier timeout; coordinator releases barrier if quorum reached or timeout expires
- **Detection**: Phi-accrual failure detector marks unresponsive nodes

#### Resource Deadlock
- **Scenario**: Node A holds resource X waiting for Y, Node B holds Y waiting for X
- **Avoidance**: Global ordering of resource acquisition (e.g., acquire locks in node ID order)
- **Detection**: Timeout on resource acquisition with retry

#### Commit Deadlock
- **Scenario**: Byzantine agreement doesn't terminate
- **Avoidance**: PBFT guarantees termination with 2f+1 honest nodes
- **Detection**: View change if no progress after timeout

---

## 4. State Persistence and Recovery

### 4.1 Which State Survives Crashes?

State is categorized by durability requirement:

| State Type              | Durability      | Persistence Mechanism         | Recovery Strategy                |
|-------------------------|-----------------|-------------------------------|----------------------------------|
| System Epoch            | **MUST persist**| Raft log + checkpoint         | Read from coordinator on rejoin  |
| Work Assignments        | **SHOULD persist** (recent) | Checkpoint every 10 tokens | Recompute if lost; fallback to last checkpoint |
| Computation Checkpoints | **MUST persist**| Disk/object storage           | Restore from latest; recompute from there |
| Node Health Status      | Ephemeral       | In-memory only                | Rediscover on restart            |
| Connection State        | Ephemeral       | In-memory only                | Reconnect on restart             |
| Metrics/Logs            | **SHOULD persist** | Async write to telemetry store | Loss acceptable; non-critical   |
| Result Commitments      | **MUST persist**| Replicated storage (3 copies) | Serve from any replica           |

### 4.2 Checkpoint Design

Checkpoints enable recovery without recomputing entire inference.

#### Checkpoint Structure

```rust
struct Checkpoint {
    /// Unique checkpoint ID
    id: CheckpointId,

    /// Epoch this checkpoint belongs to
    epoch: Epoch,

    /// Token position in sequence (for autoregressive models)
    token_position: usize,

    /// Per-node state snapshots
    node_states: HashMap<NodeId, NodeSnapshot>,

    /// Intermediate tensor results
    intermediate_results: HashMap<(NodeId, LayerId), TensorData>,

    /// Metadata for validation
    metadata: CheckpointMetadata,
}

struct CheckpointMetadata {
    /// Creation timestamp
    timestamp: i64,

    /// Checkpoint format version
    version: u32,

    /// SHA-256 checksum of all data
    checksum: [u8; 32],

    /// Vector clock for causality
    vector_clock: VectorClock,

    /// Signatures from 2f+1 nodes
    signatures: Vec<NodeSignature>,
}

struct NodeSnapshot {
    /// Node's state when checkpoint created
    state: NodeState,

    /// Node's local computation state
    computation_state: ComputationState,

    /// Memory allocations
    allocated_memory: MemoryLayout,
}
```

#### Checkpoint Creation Protocol

**Trigger**: Every K tokens (default K=10) OR explicit coordinator request

1. **Initiation** (Coordinator):
   - Broadcast `CreateCheckpoint(epoch, token_position)`
   - Timestamp: T_start

2. **Snapshot** (Workers):
   - Pause computation at safe point (between layers)
   - Serialize local state
   - Compute partial checksum
   - Send snapshot to coordinator
   - Resume computation

3. **Aggregation** (Coordinator):
   - Collect snapshots from 2f+1 nodes
   - Verify consistency (all token_positions match)
   - Compute global checksum
   - Assign checkpoint ID

4. **Persistence** (Coordinator):
   - Write checkpoint to replicated storage (3 replicas)
   - Wait for sync confirmation from storage
   - Broadcast `CheckpointComplete(checkpoint_id, checksum)`

5. **Acknowledgment** (Workers):
   - Validate checkpoint existence
   - Record checkpoint ID
   - Delete older checkpoints (retain last 3)

**Time Budget**: 200ms per checkpoint (target), 500ms timeout

**Frequency**: Every 10 tokens balances overhead vs. recovery time
- Overhead: ~2% of total inference time
- Recovery: Worst case recompute 10 tokens

### 4.3 Recovery from Checkpoints

When a worker fails and is replaced:

#### Recovery Protocol

1. **Detection** (Coordinator):
   - Phi-accrual failure detector marks node as FAILED
   - Current computation may continue if quorum maintained

2. **Replacement Selection** (Coordinator):
   - Identify standby node or spawn new worker
   - Assign same NodeId or update mapping

3. **Checkpoint Transfer** (Storage → Replacement):
   - Fetch latest checkpoint for failed node's partition
   - Verify checksum
   - Load into memory

4. **State Restoration** (Replacement):
   - Deserialize NodeSnapshot
   - Allocate memory according to MemoryLayout
   - Restore computation_state
   - Transition to RECOVERING

5. **Recomputation** (Replacement):
   - Compute from checkpoint position to current
   - Request intermediate results from dependencies (if available)
   - Otherwise recompute dependencies too (cascading recovery)

6. **Rejoin** (Replacement → Coordinator):
   - Send `RecoveryComplete(node_id, current_position)`
   - Coordinator validates position matches peers
   - Replacement transitions to COMPUTING or AGGREGATING

7. **Synchronization** (Coordinator):
   - Replacement joins at next barrier
   - Participates in remaining phases normally

**Recovery Time**:
- T_detect: ~300ms (phi-accrual with 3x heartbeat timeout)
- T_checkpoint_transfer: ~100ms (10 tokens × 10MB ≈ 100MB at 1GB/s)
- T_recompute: K × T_layer ≈ 10 × 50ms = 500ms
- **Total**: ~900ms (< 1 second)

### 4.4 State Reconciliation After Network Partition

If network partition separates cluster into two groups:

#### Partition Handling Strategy

**Scenario**: Cluster of 7 nodes (f=2, quorum=5) splits into:
- Group A: 4 nodes (includes coordinator)
- Group B: 3 nodes

**During Partition**:
- Group A: quorum NOT met (4 < 5) → transitions to DEGRADED → SHUTDOWN
- Group B: quorum NOT met (3 < 5) → transitions to DEGRADED → SHUTDOWN
- Both groups stop processing new requests
- In-flight work may complete if within single group and quorum met
- Results NOT committed (no quorum)

**After Partition Heals**:
1. Nodes detect each other via heartbeat restoration
2. Coordinator re-election occurs (if needed)
3. Nodes compare epochs and checkpoints
4. Reconciliation based on highest valid checkpoint with 2f+1 signatures
5. Any conflicting state discarded (no commits occurred during partition, so no conflict)
6. System transitions to BOOTSTRAPPING → READY

**Key Property**: Split-brain avoided by quorum requirement
- No group can commit results during partition
- No divergent state to reconcile
- Safety preserved at cost of availability (CA in CAP)

#### State Reconciliation Algorithm

```python
def reconcile_after_partition(nodes):
    # 1. Collect state from all nodes
    states = {node: node.get_state() for node in nodes}

    # 2. Find highest epoch seen by any node
    max_epoch = max(s.epoch for s in states.values())

    # 3. Find all checkpoints for max_epoch
    checkpoints = {}
    for node, state in states.items():
        if state.epoch == max_epoch:
            checkpoints[node] = state.latest_checkpoint

    # 4. Identify canonical checkpoint (most signatures)
    canonical = max(checkpoints.values(),
                   key=lambda cp: len(cp.metadata.signatures))

    # 5. Verify canonical checkpoint has 2f+1 signatures
    if len(canonical.metadata.signatures) < quorum_size:
        # No valid checkpoint; rollback to last known good
        canonical = find_last_checkpoint_with_quorum()

    # 6. Restore all nodes from canonical checkpoint
    for node in nodes:
        if node.state != canonical:
            node.restore_from_checkpoint(canonical)

    # 7. Resume from restored state
    coordinator.resume_inference(canonical.epoch, canonical.token_position)
```

---

## 5. State Transitions: Atomic Updates and Distributed Coordination

### 5.1 Atomic State Updates (Single Node)

Within a single node, state updates must be atomic to avoid corrupted state.

#### Implementation Strategy

**Use Rust's Type System for Safety**:

```rust
/// State machine with private state
pub struct WorkerStateMachine {
    state: RwLock<WorkerState>,  // Protected by lock
    epoch: AtomicU64,             // Atomic updates
}

impl WorkerStateMachine {
    /// Atomic state transition with validation
    pub fn transition_to(&self, new_state: WorkerState) -> Result<(), StateError> {
        let mut state = self.state.write().unwrap();

        // Validate transition is allowed
        if !self.is_valid_transition(&*state, &new_state) {
            return Err(StateError::InvalidTransition {
                from: *state,
                to: new_state,
            });
        }

        // Atomic update
        *state = new_state;

        // Log transition
        tracing::info!(
            node_id = ?self.node_id,
            new_state = ?new_state,
            "State transition complete"
        );

        Ok(())
    }

    fn is_valid_transition(&self, from: &WorkerState, to: &WorkerState) -> bool {
        matches!(
            (from, to),
            (WorkerState::Joined, WorkerState::Ready)
            | (WorkerState::Ready, WorkerState::Computing)
            | (WorkerState::Computing, WorkerState::Aggregating)
            | (WorkerState::Aggregating, WorkerState::Committing)
            | (WorkerState::Committing, WorkerState::Ready)
            // ... etc
        )
    }
}
```

**Key Techniques**:
1. **RwLock** for thread-safe state access
2. **Atomic types** for simple counters (epoch)
3. **Validation** before state change
4. **No partial updates**: all-or-nothing transitions

### 5.2 Distributed State Coordination

Coordinating state transitions across multiple nodes requires consensus.

#### Phase Transition Protocol

**Example: COMPUTING → AGGREGATING transition**

```
Timeline:

T0: Worker nodes complete local computation
    Node A: local_computation_complete()
    Node B: local_computation_complete()
    Node C: local_computation_complete()
    ...

T1: Workers notify coordinator
    A → Coordinator: ComputationComplete(epoch=5, node=A, checksum=hash_A)
    B → Coordinator: ComputationComplete(epoch=5, node=B, checksum=hash_B)
    C → Coordinator: ComputationComplete(epoch=5, node=C, checksum=hash_C)

T2: Coordinator waits for quorum
    Coordinator counts: {A, B, C, D, E} = 5 nodes ≥ quorum
    Coordinator validates: hash_A == hash_B == hash_C == hash_D == hash_E

T3: Coordinator broadcasts phase advance
    Coordinator → All: AdvanceToAggregating(epoch=5, result_hash=hash_A)

T4: Workers transition atomically
    All workers: state.transition_to(Aggregating)
    All workers: await barrier for Committing phase

T5: Workers acknowledge
    A → Coordinator: PhaseAck(Aggregating, epoch=5)
    ...

T6: System-level phase transition
    System transitions: COMPUTING → COMMITTING (after aggregation)
```

**Synchronization Primitive**: Two-phase protocol
1. **Phase 1**: Nodes vote to advance (implicit via ComputationComplete)
2. **Phase 2**: Coordinator broadcasts advance after quorum

**Timeout Handling**:
- If quorum not reached within timeout → enter DEGRADED
- If some nodes don't ack → proceed anyway if quorum present
- Failed nodes marked as SUSPECT → FAILED

#### Epoch Advancement Protocol

Epoch changes require strong consistency (all nodes or none).

```
Epoch N completion:

1. Coordinator receives commit votes from 2f+1 nodes
2. Coordinator persists result to storage (blocking write)
3. Coordinator updates local epoch: N → N+1
4. Coordinator broadcasts: EpochComplete(N, result_hash)
5. Workers:
   a. Verify result_hash matches local computation
   b. Persist local state
   c. Update local epoch: N → N+1
   d. Send EpochAck(N+1)
6. Coordinator waits for 2f+1 EpochAck messages
7. Coordinator transitions to MONITORING (ready for epoch N+1 work)
8. Workers transition to READY

If any step fails:
- Coordinator retries broadcast (workers may already be at N+1; idempotent)
- Workers retry persistence (idempotent)
- After 3 retries → enter DEGRADED state, investigate
```

**Guarantee**: Epoch N never abandoned without commit
- Result persistence happens BEFORE epoch increment
- If coordinator crashes after step 2, replacement reads persisted result
- No lost work

---

## 6. Error States and Recovery Procedures

### 6.1 Partial Failure Modes

Butterfly tolerates failures as long as quorum (2f+1) nodes remain healthy.

#### Worker Node Crash

**Detection**:
- Heartbeats stop arriving
- Phi-accrual detector: φ > φ_failed (default: φ > 10)
- Time: 3 × heartbeat_interval ≈ 300ms

**Response**:
1. Coordinator marks node as SUSPECTED
2. If suspected nodes count < f: continue with degraded performance
3. If suspected count ≥ f: enter DEGRADED state
4. Broadcast `NodeFailure(node_id, evidence)` to cluster
5. Initiate recovery protocol (select replacement, transfer checkpoint)

**Recovery**:
- Spawn replacement worker OR promote standby
- Transfer latest checkpoint
- Recompute from checkpoint to current position
- Rejoin at next barrier

**Impact**:
- Latency increase: +T_recovery ≈ 900ms
- Throughput decrease: (N-1)/N during recovery
- No correctness impact (result still deterministic)

#### Coordinator Crash

**Detection**:
- Workers detect missing heartbeats from coordinator
- Raft election timeout triggered
- Time: ~1 second (Raft configuration)

**Response**:
1. Workers notice coordinator unresponsive
2. Raft consensus elects new leader
3. New coordinator promoted from worker pool
4. New coordinator reads persisted state from Raft log
5. New coordinator reconstructs current epoch, work assignments
6. New coordinator broadcasts `CoordinatorElected(new_id, term)`

**Recovery**:
- In-flight computations may complete if workers have assignments
- Workers continue execution independently (pipelined)
- New coordinator resumes coordination at next barrier

**Impact**:
- Latency increase: ~1 second for election + state reconstruction
- No data loss (state in Raft log)
- Transparency to workers (they continue computing)

#### Network Partition

**Detection**:
- Nodes in different partitions stop receiving heartbeats
- Both sides suspect the other failed
- Phi-accrual detector marks entire other partition as failed

**Response (Quorum Side)**:
- If partition contains ≥ 2f+1 nodes: continues operation
- If partition contains < 2f+1 nodes: transitions to DEGRADED → SHUTDOWN
- Majority partition remains operational

**Response (Minority Side)**:
- Minority partition cannot reach quorum
- Transitions to DEGRADED state
- Stops accepting new work
- Attempts to complete in-flight work (may fail at commit phase)
- Eventually transitions to SHUTDOWN

**Recovery (After Partition Heals)**:
- Nodes reconnect
- State reconciliation protocol runs
- Minority nodes restore from majority's checkpoints
- Cluster transitions to BOOTSTRAPPING → READY
- Resume normal operation

**Impact**:
- Availability: Minority partition unavailable during partition
- Consistency: Maintained (no commits without quorum)
- Partition duration: Until network heals

### 6.2 Timeout States

Different operations have different timeout behaviors.

| Operation                | Timeout     | On Timeout Action                                          |
|--------------------------|-------------|------------------------------------------------------------|
| Work assignment ack      | 5s          | Retry 3x, then mark slow nodes as SUSPECT                  |
| Computation checkpoint   | K×T_layer+3σ| Mark slow nodes as SUSPECT; continue with quorum           |
| Barrier synchronization  | 10s         | Release if quorum reached; mark missing nodes as FAILED    |
| Byzantine agreement vote | 10s (3 RTT) | View change; re-propose with new coordinator               |
| Checkpoint write         | 5s          | Retry with backoff; enter DEGRADED if persistent failure   |
| Recovery recomputation   | K×T_layer×2 | Abort recovery; try different checkpoint                   |
| Heartbeat                | 300ms       | Mark as SUSPECTED; increase φ value                        |

**Adaptive Timeouts**:
- Timeouts adjust based on observed latency
- Formula: `timeout = mean_latency + 3 × std_dev`
- Recalculated every 100 operations
- Prevents false positives in slow networks

### 6.3 Circuit Breaker States

To prevent cascading failures, use circuit breakers for external dependencies.

#### Circuit Breaker State Machine

```
┌──────────┐
│  CLOSED  │ (normal operation)
└──────────┘
       │
       │ error_rate > threshold
       ▼
┌──────────┐
│   OPEN   │ (fail fast, don't attempt)
└──────────┘
       │
       │ timeout expires
       ▼
┌──────────┐
│ HALF_OPEN│ (try one request)
└──────────┘
       │
       ├─ success → CLOSED
       └─ failure → OPEN
```

**Applied To**:
- Checkpoint storage writes
- Model weight downloads
- Telemetry uploads

**Configuration**:
- Error threshold: 50% failures in 10 requests
- Open duration: 30 seconds
- Half-open test requests: 1

**Benefit**: Prevents worker from repeatedly trying to write checkpoints to dead storage, blocking computation

---

## 7. State Visualization and Debugging

### 7.1 State Inspection Endpoints

Expose HTTP/gRPC endpoints for runtime state queries:

```rust
// Coordinator endpoints
GET  /api/v1/system/state             // Returns: System state
GET  /api/v1/nodes                    // Returns: All node states
GET  /api/v1/nodes/{id}/state         // Returns: Specific node state
GET  /api/v1/epochs/current           // Returns: Current epoch, phase
GET  /api/v1/tasks                    // Returns: All task states
GET  /api/v1/tasks/{id}/state         // Returns: Specific task state
GET  /api/v1/checkpoints              // Returns: Available checkpoints
GET  /api/v1/health                   // Returns: Overall health summary

// Worker endpoints
GET  /api/v1/worker/state             // Returns: This worker's state
GET  /api/v1/worker/partitions        // Returns: Loaded partitions
GET  /api/v1/worker/connections       // Returns: Connection states
GET  /api/v1/worker/metrics           // Returns: Performance metrics
```

**Response Example**:

```json
{
  "system_state": "COMPUTING",
  "epoch": 42,
  "phase": "COMPUTATION",
  "coordinator": "node-0",
  "nodes": [
    {
      "id": "node-1",
      "state": "COMPUTING",
      "layers": [0, 1, 2, 3],
      "progress": 0.75,
      "last_heartbeat": "2025-10-11T12:34:56Z"
    },
    {
      "id": "node-2",
      "state": "AGGREGATING",
      "layers": [4, 5, 6],
      "progress": 1.0,
      "last_heartbeat": "2025-10-11T12:34:57Z"
    }
  ],
  "quorum": {
    "required": 5,
    "operational": 6,
    "suspected": 1
  }
}
```

### 7.2 State Transition Logging

Every state transition is logged with structured context:

```rust
tracing::info!(
    node_id = ?self.node_id,
    old_state = ?old_state,
    new_state = ?new_state,
    epoch = self.epoch,
    trigger = ?trigger,
    duration_ms = duration.as_millis(),
    "State transition completed"
);
```

**Centralized Logging**:
- All nodes send logs to central collector (e.g., Loki, Elasticsearch)
- Enables distributed tracing across state transitions
- Correlation via trace IDs

**Example Log Query** (finding slow transitions):

```promql
{job="butterfly-worker"}
| json
| duration_ms > 1000
| line_format "{{.node_id}} slow transition: {{.old_state}} → {{.new_state}} ({{.duration_ms}}ms)"
```

### 7.3 State Machine Visualization

Generate real-time state diagrams for debugging:

**Mermaid Diagram Generation**:

```rust
pub fn generate_state_diagram(&self) -> String {
    let mut diagram = String::from("stateDiagram-v2\n");

    // Add current state
    diagram.push_str(&format!("    [*] --> {}\n", self.current_state));

    // Add recent transitions
    for transition in &self.transition_history {
        diagram.push_str(&format!(
            "    {} --> {}: {} ({}ms)\n",
            transition.from,
            transition.to,
            transition.trigger,
            transition.duration_ms
        ));
    }

    // Highlight current state
    diagram.push_str(&format!("    state {} {{\n", self.current_state));
    diagram.push_str("        [*]: Current\n");
    diagram.push_str("    }\n");

    diagram
}
```

**Rendered Visualization** (via web UI):
- Real-time state machine diagram
- Nodes colored by state (green=healthy, yellow=degraded, red=failed)
- Transition arrows with timestamps
- Hover for detailed state information

### 7.4 State Assertion Framework

Programmatic checks for state invariants:

```rust
/// Runtime state invariant checker
pub struct StateInvariantChecker;

impl StateInvariantChecker {
    pub fn check_system_invariants(system: &SystemState) -> Vec<InvariantViolation> {
        let mut violations = Vec::new();

        // Check: Single coordinator
        let coordinator_count = system.nodes.iter()
            .filter(|n| matches!(n.role, NodeRole::Coordinator))
            .count();
        if coordinator_count != 1 {
            violations.push(InvariantViolation::MultipleCoordinators {
                count: coordinator_count,
            });
        }

        // Check: Quorum operational
        let operational_count = system.operational_node_count();
        let quorum_size = system.quorum_size();
        if operational_count < quorum_size && system.state != SystemState::SHUTDOWN {
            violations.push(InvariantViolation::QuorumLost {
                operational: operational_count,
                required: quorum_size,
            });
        }

        // Check: Epoch consistency
        let epochs: HashSet<u64> = system.nodes.iter()
            .map(|n| n.epoch)
            .collect();
        if epochs.len() > 2 {  // Allow 1 epoch skew during transition
            violations.push(InvariantViolation::EpochSkew {
                epochs: epochs.into_iter().collect(),
            });
        }

        // ... more checks

        violations
    }
}
```

**Integration**:
- Run invariant checks every second (background task)
- Log violations at ERROR level
- Optionally trigger alerting
- Can force DEGRADED state if critical invariant violated

---

## 8. Testing Strategy for State Machines

### 8.1 Unit Tests (Single Component)

Test individual state machines in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_state_transitions() {
        let mut worker = WorkerStateMachine::new(NodeId(0));

        // Test valid transition
        assert_eq!(worker.state(), WorkerState::Initialized);
        worker.transition_to(WorkerState::Joined).unwrap();
        assert_eq!(worker.state(), WorkerState::Joined);

        // Test invalid transition
        let result = worker.transition_to(WorkerState::Computing);
        assert!(result.is_err());
        assert_eq!(worker.state(), WorkerState::Joined);  // No change
    }

    #[test]
    fn test_worker_recovery() {
        let mut worker = WorkerStateMachine::new(NodeId(0));

        // Advance to computing state
        worker.transition_to(WorkerState::Joined).unwrap();
        worker.transition_to(WorkerState::Ready).unwrap();
        worker.transition_to(WorkerState::Computing).unwrap();

        // Simulate failure detection
        worker.handle_suspected_failure().unwrap();
        assert_eq!(worker.state(), WorkerState::Suspect);

        // Recovery
        worker.start_recovery().unwrap();
        assert_eq!(worker.state(), WorkerState::Recovering);

        worker.complete_recovery().unwrap();
        assert_eq!(worker.state(), WorkerState::Ready);
    }
}
```

### 8.2 Integration Tests (Multi-Component)

Test interactions between state machines:

```rust
#[tokio::test]
async fn test_coordinator_worker_phase_transition() {
    // Setup
    let coordinator = CoordinatorStateMachine::new(NodeId(0), 5, 2);
    let workers: Vec<_> = (1..=5)
        .map(|i| WorkerStateMachine::new(NodeId(i)))
        .collect();

    // Scenario: Complete one inference cycle

    // 1. Workers join
    for worker in &mut workers {
        worker.transition_to(WorkerState::Joined).unwrap();
        coordinator.register_worker(worker.node_id()).await.unwrap();
    }

    // 2. Coordinator assigns work
    let assignment = coordinator.create_work_assignment(/* ... */);
    coordinator.distribute_assignment(assignment).await.unwrap();

    // 3. Workers receive assignment
    for worker in &mut workers {
        worker.apply_assignment(assignment.clone()).await.unwrap();
        assert_eq!(worker.state(), WorkerState::Computing);
    }

    // 4. Workers complete computation
    for worker in &mut workers {
        worker.complete_computation().await.unwrap();
        coordinator.report_completion(worker.node_id()).await.unwrap();
    }

    // 5. Coordinator advances phase
    assert!(coordinator.is_phase_complete());
    coordinator.advance_phase().await.unwrap();
    assert_eq!(coordinator.phase(), Phase::Aggregation);

    // ... continue through commitment
}
```

### 8.3 Property-Based Tests

Use `proptest` to generate random state sequences and verify invariants:

```rust
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_epoch_monotonicity(
            transitions in prop::collection::vec(any::<ValidTransition>(), 1..100)
        ) {
            let mut system = SystemState::new();
            let mut previous_epoch = system.epoch();

            for transition in transitions {
                system.apply_transition(transition);
                let current_epoch = system.epoch();

                // Invariant: epochs never decrease
                assert!(current_epoch >= previous_epoch);

                previous_epoch = current_epoch;
            }
        }

        #[test]
        fn test_quorum_safety(
            node_count in 5usize..20,
            max_byzantine in 1usize..5,
            failure_sequence in prop::collection::vec(any::<NodeId>(), 0..10)
        ) {
            let quorum_size = 2 * max_byzantine + 1;
            prop_assume!(node_count >= quorum_size);

            let mut system = SystemState::new(node_count, max_byzantine);

            for failed_node in failure_sequence {
                system.mark_failed(failed_node);

                let operational = system.operational_node_count();
                if operational < quorum_size {
                    // Invariant: system must enter shutdown if quorum lost
                    assert_eq!(system.state(), SystemState::SHUTDOWN);
                } else {
                    // Invariant: system can continue if quorum maintained
                    assert_ne!(system.state(), SystemState::SHUTDOWN);
                }
            }
        }
    }
}
```

### 8.4 Chaos Testing

Inject failures and verify recovery:

```rust
#[tokio::test]
async fn chaos_test_random_node_failures() {
    use rand::Rng;

    // Setup 7-node cluster (f=2, quorum=5)
    let mut cluster = TestCluster::new(7, 2).await;
    cluster.bootstrap().await.unwrap();

    let mut rng = rand::thread_rng();

    for iteration in 0..100 {
        // Start inference
        let task_id = cluster.submit_inference(/* ... */).await.unwrap();

        // Randomly kill up to f nodes during computation
        let failures = rng.gen_range(0..=2);
        for _ in 0..failures {
            let victim = NodeId(rng.gen_range(0..7));
            cluster.kill_node(victim).await;
        }

        // Wait for task completion
        let result = cluster.wait_for_task(task_id).await;

        // Invariant: task should complete or fail gracefully
        assert!(result.is_ok() || matches!(result, Err(TaskError::QuorumLost)));

        // Recover killed nodes
        cluster.recover_all_nodes().await;

        // Invariant: cluster should return to READY state
        tokio::time::sleep(Duration::from_secs(2)).await;  // Allow recovery
        assert_eq!(cluster.system_state().await, SystemState::READY);
    }
}
```

### 8.5 Model Checking (TLA+)

Formal verification of state machine properties:

```tla
---- MODULE ButterflyStateMachine ----
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes,         \* Set of node IDs
          MaxByzantine,  \* f parameter
          MaxEpoch       \* Bound for model checking

VARIABLES systemState,   \* UNINITIALIZED | BOOTSTRAPPING | READY | COMPUTING | ...
          nodeStates,    \* Function: Node -> NodeState
          epoch,         \* Current epoch number
          failedNodes    \* Set of failed nodes

QuorumSize == 2 * MaxByzantine + 1
OperationalNodes == Nodes \ failedNodes

\* Type invariant
TypeOK ==
    /\ systemState \in {"UNINITIALIZED", "BOOTSTRAPPING", "READY", "COMPUTING", "DEGRADED", "COMMITTING", "SHUTDOWN", "TERMINATED"}
    /\ epoch \in 0..MaxEpoch
    /\ failedNodes \subseteq Nodes

\* Safety: If less than quorum operational, system must shutdown
QuorumSafety ==
    Cardinality(OperationalNodes) < QuorumSize =>
        systemState \in {"DEGRADED", "SHUTDOWN", "TERMINATED"}

\* Safety: Epochs only increase
EpochMonotonicity ==
    [][epoch' >= epoch]_<<epoch>>

\* Liveness: System eventually completes inference or fails
InferenceProgress ==
    systemState = "COMPUTING" ~> (systemState = "COMMITTING" \/ systemState = "DEGRADED")

====
```

Run with TLC model checker to explore state space and verify properties hold.

---

## 9. Summary: The Narrative of State

### 9.1 The Story Butterfly's State Tells

When you observe Butterfly's state, you're reading a story:

1. **UNINITIALIZED**: The system is asleep, waiting to be awakened
2. **BOOTSTRAPPING**: The system is gathering its components, like a butterfly emerging from a chrysalis
3. **READY**: The system is poised, wings spread, ready to fly
4. **COMPUTING**: The system is in flight—distributed computation flowing like synchronized wing beats
5. **DEGRADED**: The system detected damage but continues flying, compensating for wounded wings
6. **COMMITTING**: The system is landing, ensuring all parts touch down together
7. **SHUTDOWN**: The system is folding its wings, preparing to rest
8. **TERMINATED**: The system is at rest, dormant until reawakened

### 9.2 Mechanical Fitness Achieved

This state machine design ensures:

✅ **Safety**: No invalid states reachable; invariants enforced
✅ **Liveness**: Progress guaranteed under quorum conditions
✅ **Consistency**: Distributed state stays coherent via synchronization
✅ **Recoverability**: Failures detected and handled gracefully
✅ **Debuggability**: State observable and inspectable at runtime
✅ **Testability**: State transitions verifiable via automated tests

### 9.3 Narrative Coherence Achieved

The design tells a coherent story:

- **States have meaning**: Each state represents a meaningful phase of system life
- **Transitions make sense**: You can explain why each transition happens
- **Failures are part of the narrative**: DEGRADED state shows resilience, not just error
- **The system has agency**: Transitions driven by purposeful triggers, not random events
- **Time flows forward**: Epochs increase, checkpoints accumulate, progress is visible

### 9.4 Next Steps

With this state machine design complete:

1. **Implement state management modules** in each crate (see next document)
2. **Add state persistence layer** for checkpoints and recovery
3. **Build monitoring dashboards** to visualize state in production
4. **Write state machine tests** as described in Section 8
5. **Document edge cases** as they're discovered in practice

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Complete Design Specification
**Next Review**: After initial implementation feedback
