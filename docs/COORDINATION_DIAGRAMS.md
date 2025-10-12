# Butterfly Coordination Protocol - Visual Documentation

## State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Node State Machine                              │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ INITIALIZING │  (Node starting up, loading model)
    └──────┬───────┘
           │ Model loaded
           │ Join cluster
           ▼
    ┌──────────────┐
    │    READY     │◄──────────┐ (Ready to receive work)
    └──────┬───────┘           │
           │ Work assigned     │ Result committed
           ▼                   │
    ┌──────────────┐           │
    │  COMPUTING   │           │ (Executing layers)
    └──────┬───────┘           │
           │ Computation done  │
           ▼                   │
    ┌──────────────┐           │
    │ AGGREGATING  │           │ (Waiting at barrier)
    └──────┬───────┘           │
           │ Barrier released  │
           │ Byzantine vote    │
           ▼                   │
    ┌──────────────┐           │
    │  COMMITTING  ├───────────┘ (Finalizing result)
    └──────────────┘

    Failure paths:

    Any state ────failure detected───┐
           │                          ▼
           │                   ┌──────────────┐
           │                   │   DEGRADED   │  (Peer failed)
           │                   └──────┬───────┘
           │                          │ Initiate recovery
           │                          ▼
           │                   ┌──────────────┐
           └──critical error──►│  RECOVERING  │  (Restoring state)
                               └──────┬───────┘
                                      │ Recovery complete
                                      ▼
                               ┌──────────────┐
                               │    READY     │
                               └──────────────┘

                               ┌──────────────┐
                               │    FAILED    │  (Terminal state)
                               └──────────────┘
```

## Execution Phases

```
┌────────────────────────────────────────────────────────────────────┐
│                    Multi-Phase Execution Flow                       │
└────────────────────────────────────────────────────────────────────┘

Phase 1: ASSIGNMENT (Coordinated)
═══════════════════════════════════════════
    Coordinator                    Nodes
        │                            │
        │── WorkAssignment ──────────┤
        │                            │
        │◄──── ACK ──────────────────┤
        │                            │
    [Barrier: All nodes must ACK]
        │                            │
        └────────────────────────────┘
            Timeout: 5 seconds


Phase 2: COMPUTATION (Pipelined - No Sync!)
═══════════════════════════════════════════
    Node 0         Node 1         Node 2
    ┌────┐         ┌────┐         ┌────┐
    │ L0 │────────►│ L1 │────────►│ L2 │
    │ L3 │         │ L4 │         │ L5 │
    └────┘         └────┘         └────┘
      │              │              │
      │ (parallel)   │ (parallel)   │
      ▼              ▼              ▼
    Checkpoint   Checkpoint   Checkpoint
    (every K=10 tokens)

    ⚡ Optimization: Results stream immediately
    ⚡ No synchronization between layers
    ⚡ Checkpoints happen asynchronously


Phase 3: AGGREGATION (Barrier Sync)
═══════════════════════════════════════════
    Coordinator              Final Nodes
        │                         │
        │◄── Computation Done ────┤
        │    + Checkpoint Hash    │
        │                         │
        │◄────────────────────────┤
        │                         │
    [Barrier: Quorum must reach]
    [Verify: All hashes match]
        │                         │
        │─── Barrier Release ─────┤
        │                         │
        └─────────────────────────┘
            Timeout: Compute + 3σ


Phase 4: COMMITMENT (Byzantine Agreement)
═══════════════════════════════════════════

    Standard Path (3 RTT):
    ────────────────────────────────────

    Coordinator         Nodes
        │                 │
        │─ PRE-PREPARE ───┤  (Propose result + proof)
        │   (Result R)    │
        │                 │
        │◄── PREPARE ─────┤  (Validate & vote)
        │    (Hash R)     │
        │                 │
    [Wait for 2f+1 PREPARE votes]
        │                 │
        │◄── COMMIT ──────┤  (Finalize)
        │                 │
    [Wait for 2f+1 COMMIT votes]
        │                 │
        └─────────────────┘


    Optimistic Fast Path (1 RTT):
    ────────────────────────────────────

    Coordinator         Nodes
        │                 │
        │─ PRE-PREPARE ───┤
        │                 │
        │◄── IMMEDIATE ───┤  (All agree instantly)
        │    COMMIT       │
        │                 │
    [Skip PREPARE phase]
        │                 │
        └─────────────────┘

    ⚡ Used when no disagreement detected
```

## Work Assignment Algorithm

```
┌────────────────────────────────────────────────────────────────────┐
│            Hybrid Load-Balanced Assignment Strategy                 │
└────────────────────────────────────────────────────────────────────┘

Input:
  • L layers: [L₀, L₁, ..., Lₙ]
  • N nodes: [N₀, N₁, ..., Nₘ]
  • Cost function: cost(Lᵢ, Nⱼ)

Step 1: Initial Topology-Aware Assignment
═══════════════════════════════════════════════

    Layers:  L₀ L₁ L₂ │ L₃ L₄ L₅ │ L₆ L₇ L₈
             ─────────┬──────────┬──────────
    Nodes:      N₀    │    N₁    │    N₂

    • Group consecutive layers (minimize communication)
    • Create pipeline: N₀ → N₁ → N₂


Step 2: Compute Load Imbalance
═══════════════════════════════════════════════

    Load(N₀) = Σ cost(Lᵢ) for i ∈ {0,1,2}
    Load(N₁) = Σ cost(Lᵢ) for i ∈ {3,4,5}
    Load(N₂) = Σ cost(Lᵢ) for i ∈ {6,7,8}

    Avg = (Load(N₀) + Load(N₁) + Load(N₂)) / 3
    Max = max(Load(Nᵢ))

    Imbalance = (Max - Avg) / Avg


Step 3: Rebalance if Needed
═══════════════════════════════════════════════

    If Imbalance > 20%:

        Greedy reassignment:

        1. Sort layers by cost (descending)
        2. For each layer L:
           Assign to node N with minimum current load
        3. Rebuild dependency graph

    Else:
        Keep topology-aware assignment


Result: WorkAssignment
═══════════════════════════════════════════════

    {
      Node₀: {layers: [L₀, L₃, L₆], deps: [], consumers: [N₁]},
      Node₁: {layers: [L₁, L₄, L₇], deps: [N₀], consumers: [N₂]},
      Node₂: {layers: [L₂, L₅, L₈], deps: [N₁], consumers: []},
    }

    Dependency Graph:
        N₀ ──► N₁ ──► N₂

    Execution Order: [N₀, N₁, N₂]
```

## Byzantine Agreement Protocol

```
┌────────────────────────────────────────────────────────────────────┐
│         Modified PBFT for Distributed Inference Results             │
└────────────────────────────────────────────────────────────────────┘

Participants:
  • 1 Coordinator (C)
  • N-1 Replica Nodes (R₁, R₂, ..., Rₙ₋₁)
  • f Byzantine nodes tolerated
  • Quorum: 2f + 1 votes needed


Phase 1: PRE-PREPARE
═══════════════════════════════════════════════

    C                    R₁        R₂    ...   Rₙ
    │                    │         │           │
    │ Propose result R   │         │           │
    │ with proof π       │         │           │
    ├───────────────────►│         │           │
    ├────────────────────┼────────►│           │
    ├────────────────────┼─────────┼──────────►│
    │                    │         │           │

    Proof π includes:
    • Input hash
    • Output hash
    • Intermediate checksums
    • Cryptographic signature


Phase 2: PREPARE
═══════════════════════════════════════════════

    C         R₁                R₂              Rₙ
    │         │                 │               │
    │         │ Validate π      │               │
    │         │ Check hash(R)   │               │
    │         │                 │               │
    │         ├────────────────►│               │
    │         │   PREPARE(R)    │               │
    │         ├────────────────────────────────►│
    │         │                 │               │
    │         │◄────────────────┤               │
    │         │◄────────────────────────────────┤
    │         │                 │               │

    Wait for 2f+1 matching PREPARE messages


Phase 3: COMMIT
═══════════════════════════════════════════════

    C         R₁                R₂              Rₙ
    │         │                 │               │
    │         │ COMMIT(R)       │               │
    │         ├────────────────►│               │
    │         ├────────────────────────────────►│
    │         │                 │               │
    │         │◄────────────────┤               │
    │         │◄────────────────────────────────┤
    │         │                 │               │

    Wait for 2f+1 COMMIT messages
    → Result R is finalized


Byzantine Behavior Detection:
═══════════════════════════════════════════════

    Scenario 1: Conflicting Results
    ────────────────────────────────

    C proposes R₁
    Byzantine node B proposes R₂ (R₁ ≠ R₂)

    Honest nodes (≥ f+1) vote for R₁
    Byzantine nodes (≤ f) vote for R₂

    R₁ gets ≥ 2f+1 votes → R₁ committed
    R₂ gets ≤ f votes → R₂ rejected


    Scenario 2: Invalid Proof
    ────────────────────────────────

    C proposes R with invalid proof π

    Honest nodes detect: π.signature invalid
    → Reject PRE-PREPARE
    → C marked as Byzantine
    → Trigger leader election


    Scenario 3: Equivocation
    ────────────────────────────────

    Byzantine node B sends:
    • PREPARE(R₁) to some nodes
    • PREPARE(R₂) to other nodes

    Honest nodes exchange messages
    → Detect conflicting votes from B
    → Mark B as Byzantine
    → Exclude from quorum
```

## Failure Detection and Recovery

```
┌────────────────────────────────────────────────────────────────────┐
│              φ-Accrual Failure Detector                             │
└────────────────────────────────────────────────────────────────────┘

Heartbeat Protocol:
═══════════════════════════════════════════════

    Node A                    Node B
      │                         │
      ├──── HEARTBEAT ─────────►│ (t₁)
      │                         │ Record arrival time
      │                         │
      ├──── HEARTBEAT ─────────►│ (t₂)
      │                         │ Compute interval: Δt = t₂ - t₁
      │                         │
      ├──── HEARTBEAT ─────────►│ (t₃)
      │                         │ Update statistics
      │                         │


φ Computation:
═══════════════════════════════════════════════

    Given heartbeat history: [t₁, t₂, t₃, ..., tₙ]

    1. Compute inter-arrival times:
       Δt = [t₂-t₁, t₃-t₂, ..., tₙ-tₙ₋₁]

    2. Statistical parameters:
       μ = mean(Δt)
       σ = stddev(Δt)

    3. Time since last heartbeat:
       T_elapsed = now() - tₙ

    4. Suspicion level (φ):
       z = (T_elapsed - μ) / σ
       φ = -log₁₀(P(X > T_elapsed))

    5. Decision:
       if φ > φ_suspect (8.0):  Mark as SUSPECTED
       if φ > φ_failed (12.0):  Mark as FAILED


Adaptive Interval:
═══════════════════════════════════════════════

    Base interval: 100ms

    Congestion factor: C = μ / base_interval

    Adaptive interval = base_interval × (1 + C)

    ┌──────────────────────────────────────┐
    │ Network Condition    │ Interval      │
    ├──────────────────────┼───────────────┤
    │ Low latency          │ 100ms         │
    │ Moderate congestion  │ 150ms         │
    │ High latency         │ 300ms         │
    └──────────────────────────────────────┘


Recovery Protocol:
═══════════════════════════════════════════════

    Step 1: Detect Failure
    ────────────────────────────────────

    Coordinator                Failed Node (F)
        │                           │
        │ ◄─ no heartbeat ─────────X
        │ ◄─ no heartbeat ─────────X
        │ ◄─ no heartbeat ─────────X
        │                           │
        φ > 12.0 → FAILED
        │
        │ Broadcast: SUSPECT(F)


    Step 2: Select Replacement
    ────────────────────────────────────

    Coordinator       Replacement (R)
        │                  │
        │─ ASSIGN_WORK ───►│
        │   (F's partition)│
        │                  │


    Step 3: Transfer Checkpoint
    ────────────────────────────────────

    Coordinator          R          Observer Nodes
        │                │               │
        │◄─ REQUEST_CP ──┤               │
        │                │               │
        ├─ CHECKPOINT ───┼──────────────►│
        │   @ token K    │               │
        │                │               │
        │                │ Validate CP   │
        │                │ Load state    │


    Step 4: Recompute
    ────────────────────────────────────

    R computes layers from position K to current

    Current position: 47
    Checkpoint position: 40
    Recompute: 40 → 47 (7 tokens)

    Time = 7 × T_layer ≈ 70ms


    Step 5: Rejoin
    ────────────────────────────────────

    R                  Coordinator
    │                       │
    ├── READY ─────────────►│
    │                       │
    │◄── BARRIER_WAIT ──────┤
    │                       │
    Rejoin at next barrier
```

## Checkpoint Management

```
┌────────────────────────────────────────────────────────────────────┐
│               Checkpoint Storage and Retrieval                      │
└────────────────────────────────────────────────────────────────────┘

Checkpoint Structure:
═══════════════════════════════════════════════

    Checkpoint {
      epoch: u64,
      token_position: usize,
      node_states: HashMap<NodeId, Vec<u8>>,
      intermediate_results: HashMap<NodeId, Vec<u8>>,
      metadata: {
        timestamp: i64,
        version: u32,
        checksum: [u8; 32],  // SHA-256
      }
    }


Checkpoint Frequency:
═══════════════════════════════════════════════

    Default: Every K = 10 tokens

    ┌──────────────────────────────────────────┐
    │ Token  │ 0   10   20   30   40   50   60 │
    │ Action │ CP  CP   CP   CP   CP   CP   CP │
    └──────────────────────────────────────────┘

    Configurable based on:
    • Model size (larger → less frequent)
    • Network speed (slower → less frequent)
    • Recovery time requirements (faster → more frequent)


Checkpoint Storage (FIFO):
═══════════════════════════════════════════════

    Max checkpoints: 10 (default)

    ┌──────┬──────┬──────┬─────┬──────┐
    │ CP₀  │ CP₁  │ CP₂  │ ... │ CP₉  │
    └──────┴──────┴──────┴─────┴──────┘
     oldest              newest

    When adding CP₁₀:
    • Evict CP₀
    • Shift window
    • Add CP₁₀

    ┌──────┬──────┬──────┬─────┬──────┐
    │ CP₁  │ CP₂  │ CP₃  │ ... │ CP₁₀ │
    └──────┴──────┴──────┴─────┴──────┘


Recovery from Checkpoint:
═══════════════════════════════════════════════

    Timeline:
    ────────────────────────────────────────────

    0   10   20   30   40   50   60   70
    CP  CP   CP   CP   CP         X    Current
                        ▲         │
                        │         │ Failure!
                        │         │
                        └─────────┘
                        Recover from CP₄₀

    Steps:
    1. Find closest checkpoint ≤ current position
    2. Load checkpoint CP₄₀
    3. Recompute: position 40 → 70
    4. Verify result matches expected


Distributed Checkpoint Storage:
═══════════════════════════════════════════════

    Coordinator    Node 0    Node 1    Node 2    Observer
        │            │         │         │          │
        │◄─ State ───┤         │         │          │
        │◄─ State ───┼─────────┤         │          │
        │◄─ State ───┼─────────┼─────────┤          │
        │            │         │         │          │
    Aggregate states                                │
        │                                            │
        ├──── Store Checkpoint ────────────────────►│
        │                                            │

    Redundancy: Store on 3 observer nodes
    Quorum read: Need 2/3 matching copies
```

## Performance Characteristics

```
┌────────────────────────────────────────────────────────────────────┐
│                  Latency and Throughput Analysis                    │
└────────────────────────────────────────────────────────────────────┘

End-to-End Latency (No Failures):
═══════════════════════════════════════════════

    ┌───────────────────────────────────────────┐
    │ Phase          │ Latency    │ Sync?       │
    ├────────────────┼────────────┼─────────────┤
    │ Assignment     │ 1 RTT      │ Yes (start) │
    │ Computation    │ L/N × T_L  │ No (async!) │
    │ Aggregation    │ 1 RTT      │ Yes (end)   │
    │ Commitment     │ 1 RTT      │ Yes (BFT)   │
    └───────────────────────────────────────────┘

    Total: 3 RTT + compute_time

    Example (N=5 nodes, L=50 layers, T_L=10ms):
    • Compute: 50/5 × 10ms = 100ms
    • Sync: 3 × 1ms = 3ms
    • Total: 103ms

    Overhead: 3% (excellent!)


Throughput Scaling:
═══════════════════════════════════════════════

    Single request:
    T_total = 3 RTT + (L/N) × T_L

    With pipelining (P concurrent requests):
    Throughput = P / T_total

    Steady-state throughput (fully pipelined):
    Max throughput ≈ 1 / (T_L × L/N)

    Example:
    • T_L = 10ms per layer
    • L = 50 layers
    • N = 10 nodes
    • Throughput ≈ 1 / (10ms × 5) = 20 req/sec

    ┌──────────────────────────────────────────┐
    │ Nodes │ Layers/Node │ Throughput        │
    ├───────┼─────────────┼───────────────────┤
    │   5   │     10      │  10 req/sec       │
    │  10   │      5      │  20 req/sec       │
    │  20   │    2.5      │  40 req/sec       │
    └──────────────────────────────────────────┘

    Near-linear scaling!


Communication Complexity:
═══════════════════════════════════════════════

    Per inference request:

    Assignment phase:
    • Coordinator → All: O(N) messages
    • All → Coordinator: O(N) ACKs

    Computation phase:
    • Pipeline flow: O(L) tensor transfers
    • No all-to-all communication!

    Aggregation phase:
    • Final nodes → Coordinator: O(1) messages
    • Coordinator → All: O(N) releases

    Commitment phase:
    • PRE-PREPARE: O(N)
    • PREPARE: O(N²) worst-case, O(N) optimistic
    • COMMIT: O(N²) worst-case, O(N) optimistic

    Total (optimistic): O(N + L) messages
    Total (pessimistic): O(N²) messages

    ⚡ Key insight: L >> N typically, so O(L) dominates
    ⚡ Pipelining means O(1) messages per layer


Failure Impact:
═══════════════════════════════════════════════

    Failure-free: 100% throughput
    1 node fails (out of 7):
    • Detection time: ~300ms
    • Recovery time: ~500ms
    • Throughput during recovery: 85%
    • Post-recovery throughput: 85% (6/7 nodes)

    2 nodes fail (out of 7, f=2):
    • System still operational (5 ≥ 2f+1)
    • Throughput: 71% (5/7 nodes)

    3 nodes fail (out of 7):
    • System halts (5 < 2f+1=5)
    • Need reconfig or recovery

    ┌──────────────────────────────────────────┐
    │ Failed  │ Availability │ Throughput      │
    ├─────────┼──────────────┼─────────────────┤
    │   0     │   100%       │     100%        │
    │   1     │   100%       │     85%         │
    │   2     │   100%       │     71%         │
    │   3+    │    0%        │      0%         │
    └──────────────────────────────────────────┘

    Graceful degradation until quorum lost
```

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│              Butterfly Distributed Inference System                 │
└────────────────────────────────────────────────────────────────────┘

                        Client Application
                               │
                               │ gRPC / REST
                               ▼
                    ┌─────────────────────┐
                    │   API Gateway        │
                    │  (butterfly-api)     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Node 0  │   │  Node 1  │   │  Node 2  │
        │          │   │          │   │          │
        │ ┌──────┐ │   │ ┌──────┐ │   │ ┌──────┐ │
        │ │Coord │ │   │ │Coord │ │   │ │Coord │ │  Coordination
        │ │ SM   │ │   │ │ SM   │ │   │ │ SM   │ │  State Machine
        │ └──┬───┘ │   │ └──┬───┘ │   │ └──┬───┘ │
        │    │     │   │    │     │   │    │     │
        │ ┌──▼───┐ │   │ ┌──▼───┐ │   │ ┌──▼───┐ │
        │ │Work  │ │   │ │Work  │ │   │ │Work  │ │  Inference
        │ │Exec  │ │   │ │Exec  │ │   │ │Exec  │ │  Execution
        │ └──┬───┘ │   │ └──┬───┘ │   │ └──┬───┘ │
        │    │     │   │    │     │   │    │     │
        │ ┌──▼───┐ │   │ ┌──▼───┐ │   │ ┌──▼───┐ │
        │ │Model │ │   │ │Model │ │   │ │Model │ │  Model
        │ │Part  │ │   │ │Part  │ │   │ │Part  │ │  Partition
        │ └──────┘ │   │ └──────┘ │   │ └──────┘ │
        └──────────┘   └──────────┘   └──────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                    QUIC / gRPC (butterfly-comm)
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Observer 0  │ │  Observer 1  │ │  Observer 2  │
    │              │ │              │ │              │
    │ Checkpoints  │ │ Checkpoints  │ │ Checkpoints  │
    │ Metrics      │ │ Metrics      │ │ Metrics      │
    │ Monitoring   │ │ Monitoring   │ │ Monitoring   │
    └──────────────┘ └──────────────┘ └──────────────┘


Component Interactions:
═══════════════════════════════════════════════

    butterfly-coordination:
    • State machine management
    • Byzantine agreement
    • Failure detection
    • Checkpoint coordination
    • Work assignment

    butterfly-comm:
    • QUIC-based message transport
    • Reliable delivery
    • Flow control
    • Compression

    butterfly-schedule:
    • Task queuing
    • Priority management
    • Pipeline coordination

    butterfly-partition:
    • Model splitting
    • Layer distribution
    • Tensor management

    butterfly-metrics:
    • Performance monitoring
    • Failure tracking
    • Resource utilization
```

## Message Flow Example

```
┌────────────────────────────────────────────────────────────────────┐
│         Complete Inference Request Message Flow                     │
└────────────────────────────────────────────────────────────────────┘

Client      Gateway     Coord(N₀)     Node 1      Node 2     Observer
  │           │            │            │           │            │
  │           │            │            │           │            │
  ├─ Infer ──►│            │            │           │            │
  │           │            │            │           │            │
  │           ├─ Submit ──►│            │           │            │
  │           │            │            │           │            │
  │           │            ├─ Assign ──►│           │            │
  │           │            ├─ Assign ───┼──────────►│            │
  │           │            │            │           │            │
  │           │            │◄─── ACK ───┤           │            │
  │           │            │◄─── ACK ────┼───────────┤            │
  │           │            │            │           │            │
  │           │            │ [Barrier: Assignment Complete]       │
  │           │            │            │           │            │
  │           │            │ Compute L₀ │           │            │
  │           │            │ ────┐      │           │            │
  │           │            │     │      │           │            │
  │           │            │ ◄───┘      │           │            │
  │           │            ├─ Result ──►│           │            │
  │           │            │            │ Compute L₁│            │
  │           │            │            │ ────┐     │            │
  │           │            │            │     │     │            │
  │           │            │            │ ◄───┘     │            │
  │           │            │            ├─ Result ─►│            │
  │           │            │            │           │ Compute L₂ │
  │           │            │            │           │ ────┐      │
  │           │            │            │           │     │      │
  │           │            │            │           │ ◄───┘      │
  │           │            │            │           │            │
  │           │            │            │ ⏱ Checkpoint (K=10)    │
  │           │            ├─ CP ───────┼───────────┼───────────►│
  │           │            │            │           │            │
  │           │            │            │           │            │
  │           │            │◄─ Done ────┼───────────┤            │
  │           │            │◄─ Done ────┼───────────┼────────────┤
  │           │            │            │           │            │
  │           │            │ [Barrier: Computation Complete]      │
  │           │            │            │           │            │
  │           │            ├─ PrePrep ─►│           │            │
  │           │            ├─ PrePrep ──┼──────────►│            │
  │           │            │            │           │            │
  │           │            │◄─ Prepare ─┤           │            │
  │           │            │◄─ Prepare ─┼───────────┤            │
  │           │            │            │           │            │
  │           │            │ [Quorum: 2f+1 reached]               │
  │           │            │            │           │            │
  │           │            │◄─ Commit ──┤           │            │
  │           │            │◄─ Commit ───┼──────────┤            │
  │           │            │            │           │            │
  │           │            │ [Result Finalized]                  │
  │           │            │            │           │            │
  │           │◄─ Result ──┤            │           │            │
  │           │            │            │           │            │
  │◄─ Result ─┤            │            │           │            │
  │           │            │            │           │            │


Timeline:
═════════════════════════════════════════════

    0ms:   Client submits request
    1ms:   Work assignment distributed
    2ms:   All nodes ACK (barrier released)
    12ms:  Node 0 completes L₀, streams to Node 1
    22ms:  Node 1 completes L₁, streams to Node 2
    32ms:  Node 2 completes L₂
    33ms:  Aggregation barrier reached
    34ms:  PRE-PREPARE sent
    35ms:  PREPARE votes collected
    36ms:  COMMIT votes collected
    37ms:  Result returned to client

    Total: 37ms
    Compute: 30ms (81%)
    Coordination: 7ms (19%)
```
