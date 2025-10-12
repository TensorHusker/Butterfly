# Butterfly State Coherence Specification

## Executive Summary

This document formalizes the **consistency guarantees** and **coherence properties** of Butterfly's distributed state machine. It defines what "correctness" means for the system and provides rigorous frameworks for reasoning about distributed state.

**Core Philosophy**: State coherence is not just about avoiding inconsistency—it's about ensuring the system's distributed state tells a single, comprehensible story at all times.

---

## 1. Consistency Models

### 1.1 System-Wide Consistency Guarantees

Butterfly provides **Strong Consistency** for committed results with **Eventual Consistency** for health monitoring.

| Property               | Guarantee Level        | Mechanism                          | Trade-off                          |
|------------------------|------------------------|------------------------------------|------------------------------------|
| Inference Results      | **Linearizability**    | Byzantine agreement (PBFT)         | Availability during partitions     |
| Epoch Numbers          | **Sequential**         | Raft log replication               | Coordinator dependency             |
| Work Assignments       | **Causal**             | Vector clocks + checkpoints        | Memory overhead                    |
| Node Health            | **Eventual**           | Gossip + phi-accrual               | Temporary inconsistent views       |
| Metrics                | **Eventual**           | Async aggregation                  | Delayed visibility                 |

### 1.2 Formal Consistency Definitions

#### Linearizability (Inference Results)

**Definition**: Every operation appears to execute instantaneously at some point between invocation and response, respecting real-time ordering.

**Formally**:
```
For operations op₁ and op₂:
  If op₁.response_time < op₂.invocation_time (real-time order)
  Then op₁ must precede op₂ in the linearization

For concurrent operations:
  Any legal total order consistent with per-node order is valid
```

**Implication for Butterfly**:
- If client A receives result for inference request R₁ at time T₁
- And client B submits request R₂ at time T₂ > T₁
- Then R₂ observes all effects of R₁ (e.g., epoch advanced)

**Verification**:
```rust
// Test case: Sequential inference requests must observe monotonic epochs
#[test]
fn test_linearizability() {
    let result1 = cluster.infer(input1).await.unwrap();  // epoch N
    let result2 = cluster.infer(input2).await.unwrap();  // must be epoch N+1 or later

    assert!(result2.epoch > result1.epoch);
}
```

#### Sequential Consistency (Epochs)

**Definition**: All nodes observe operations in the same total order, which respects per-node program order.

**Formally**:
```
For each node i:
  Let local_order_i be the order node i performs operations

There exists a global total order such that:
  1. All nodes observe this same order
  2. For each node i, local_order_i ⊆ global_order
```

**Implication for Butterfly**:
- All nodes see epochs advance in the same order: 0 → 1 → 2 → ...
- No node sees epoch 3 before epoch 2
- Epochs may be observed at different real times (no wall-clock bound)

**Verification**:
```rust
// Test case: All nodes agree on epoch history
#[tokio::test]
async fn test_sequential_consistency() {
    let mut cluster = TestCluster::new(7, 2).await;

    // Run 10 inference requests
    for i in 0..10 {
        cluster.infer(/* input */).await.unwrap();
    }

    // Query epoch history from each node
    let histories: Vec<Vec<Epoch>> = cluster.nodes()
        .map(|n| n.get_epoch_history())
        .collect();

    // All histories should be prefixes of each other
    // (some nodes may lag, but order must match)
    for h in &histories {
        assert_is_prefix_of(&histories[0], h);
    }
}
```

#### Causal Consistency (Work Assignments)

**Definition**: Operations causally related are seen in the same order by all nodes; concurrent operations may be seen in different orders.

**Formally**:
```
Define happens-before relation →:
  - a → b if a and b occur on same node and a precedes b
  - a → b if a is send(m) and b is receive(m)
  - a → b if a → c and c → b (transitivity)

Causal consistency: If a → b, all nodes observe a before b
```

**Implication for Butterfly**:
- If checkpoint C₁ is created before checkpoint C₂ on causally related path
- All nodes observe C₁ before C₂
- Checkpoints on independent paths may be observed in any order

**Verification**:
```rust
// Test case: Causal dependencies respected
#[tokio::test]
async fn test_causal_consistency() {
    let mut cluster = TestCluster::new(5, 1).await;

    // Checkpoint sequence: C1 → C2 → C3
    let c1 = cluster.create_checkpoint().await.unwrap();
    let c2 = cluster.create_checkpoint_after(c1).await.unwrap();
    let c3 = cluster.create_checkpoint_after(c2).await.unwrap();

    // Query visible checkpoints from each node
    for node in cluster.nodes() {
        let visible = node.get_visible_checkpoints();

        // If node sees C3, it must also see C1 and C2
        if visible.contains(&c3) {
            assert!(visible.contains(&c1));
            assert!(visible.contains(&c2));
        }

        // If node sees C2, it must see C1
        if visible.contains(&c2) {
            assert!(visible.contains(&c1));
        }
    }
}
```

### 1.3 CAP Theorem Trade-offs

Butterfly makes explicit CAP choices:

**Partition Scenario**:
```
Initial cluster: 7 nodes (quorum = 5)
Network partition: [4 nodes] | [3 nodes]
```

**Behavior**:
- **Consistency**: ✅ Maintained
  - No group has quorum (4 < 5, 3 < 5)
  - No commits occur in either partition
  - No divergent state

- **Availability**: ❌ Sacrificed
  - Both partitions transition to DEGRADED → SHUTDOWN
  - New inference requests rejected
  - System unavailable until partition heals

**Rationale**:
- Inference result correctness is non-negotiable
- Availability loss is acceptable (retry at application layer)
- Partition healing is fast in datacenter networks (< 1 minute typical)

**Alternative Designs Rejected**:

| Alternative            | Problem                                                |
|------------------------|--------------------------------------------------------|
| Allow minority commits | Split-brain: divergent epochs, conflicting results     |
| Read-only mode         | Stateful computation cannot serve stale reads safely   |
| Optimistic commits     | Byzantine nodes could exploit partition to cheat       |

---

## 2. Coherence Properties

### 2.1 Vertical Coherence (Component ↔ System)

**Definition**: Component states must be valid projections of system state.

**Formal Constraint**:
```
∀ component c ∈ {coordinator, workers, tasks}:
  state(c) ∈ ValidStates(state(system))

Where ValidStates is defined by the table in Section 3.1 of state_machine_design.md
```

**Example Violations**:

❌ **Invalid**: Worker in COMPUTING while system in READY
```
System: READY
Worker: COMPUTING  ← VIOLATION: No work should be executing

Cause: Worker failed to transition after previous epoch
Fix: Force worker to READY state; log anomaly
```

❌ **Invalid**: Task in QUEUED while system in TERMINATED
```
System: TERMINATED
Task: QUEUED  ← VIOLATION: No pending work in terminated system

Cause: Task not cancelled during shutdown
Fix: Transition task to FAILED; cleanup resources
```

✅ **Valid**: Workers in mixed states during COMPUTING
```
System: COMPUTING
Worker A: COMPUTING  ← Valid: executing assigned layers
Worker B: AGGREGATING  ← Valid: finished local computation
Worker C: COMPUTING  ← Valid: still processing

Explanation: Pipeline parallelism allows state heterogeneity within Computing system state
```

**Enforcement Mechanism**:

```rust
pub fn validate_vertical_coherence(system: &SystemState) -> Result<(), CoherenceError> {
    for worker in &system.workers {
        if !is_valid_worker_state_for_system(worker.state, system.state) {
            return Err(CoherenceError::VerticalIncoherence {
                system_state: system.state,
                component_type: "Worker",
                component_id: worker.id,
                component_state: worker.state,
            });
        }
    }

    for task in &system.tasks {
        if !is_valid_task_state_for_system(task.state, system.state) {
            return Err(CoherenceError::VerticalIncoherence {
                system_state: system.state,
                component_type: "Task",
                component_id: task.id,
                component_state: task.state,
            });
        }
    }

    Ok(())
}
```

**Monitoring**:
- Run validation every 1 second
- Alert on violations
- Automatically repair if possible

### 2.2 Horizontal Coherence (Component ↔ Component)

**Definition**: Components at the same architectural level must maintain mutual consistency.

#### Worker-Worker Coherence

**Constraint**: All workers processing same epoch must be at compatible phases.

**Formal Rule**:
```
∀ workers w₁, w₂:
  w₁.epoch == w₂.epoch ⟹
    phase_distance(w₁.phase, w₂.phase) ≤ 1

Where phase_distance(p₁, p₂) = |index(p₁) - index(p₂)|
And phase indices: Assignment=0, Computation=1, Aggregation=2, Commitment=3
```

**Allowed**:
```
Worker 1: epoch=5, phase=COMPUTATION
Worker 2: epoch=5, phase=AGGREGATION  ← phase_distance = 1 ✅
```

**Forbidden**:
```
Worker 1: epoch=5, phase=COMPUTATION
Worker 2: epoch=5, phase=COMMITMENT  ← phase_distance = 2 ❌
```

**Why**: Worker 2 cannot commit before Worker 1 completes computation—violates causality.

#### Task-Task Coherence

**Constraint**: Tasks in same batch must progress together.

**Formal Rule**:
```
∀ tasks t₁, t₂ in same batch B:
  state(t₁) ≥ state(t₂) - 1  (in state ordering)

State ordering: QUEUED < SCHEDULED < DISPATCHED < EXECUTING < VALIDATING < COMPLETED
```

**Rationale**: Batch processing requires all tasks start together; some may finish earlier, but none can be left behind by more than one state.

### 2.3 Temporal Coherence (State ↔ Time)

**Definition**: State changes must respect temporal causality and bounded staleness.

#### Timestamp Ordering

**Rule**: State transitions must have monotonically increasing timestamps.

**Formal Constraint**:
```
For state transitions s₁ → s₂ on same component:
  timestamp(s₂) > timestamp(s₁)
```

**Implementation**:
```rust
pub struct StateTransition {
    from: ComponentState,
    to: ComponentState,
    timestamp: i64,  // Monotonic clock (not wall clock)
    epoch: Epoch,
}

impl StateTransition {
    pub fn new(from: ComponentState, to: ComponentState, epoch: Epoch) -> Self {
        let timestamp = get_monotonic_time();  // Guaranteed increasing

        // Validate temporal ordering
        if let Some(prev) = LAST_TRANSITION.lock().unwrap().as_ref() {
            assert!(timestamp > prev.timestamp, "Temporal causality violated");
        }

        *LAST_TRANSITION.lock().unwrap() = Some((timestamp, epoch));

        Self { from, to, timestamp, epoch }
    }
}
```

#### Bounded Staleness

**Rule**: Component state may lag system state, but not by more than Δ time.

**Formal Constraint**:
```
For worker w, coordinator c:
  |timestamp(state(w)) - timestamp(state(c))| ≤ Δ_max

Where Δ_max = 2 × heartbeat_interval = 200ms
```

**Rationale**: If worker state is stale by > 200ms, it's likely failed; mark as SUSPECT.

**Monitoring**:
```rust
pub async fn monitor_staleness(cluster: &Cluster) {
    loop {
        let coordinator_time = cluster.coordinator.state_timestamp();

        for worker in cluster.workers() {
            let worker_time = worker.state_timestamp();
            let staleness = coordinator_time - worker_time;

            if staleness > Duration::from_millis(200) {
                warn!(
                    worker_id = ?worker.id,
                    staleness_ms = staleness.as_millis(),
                    "Worker state stale, suspecting failure"
                );

                cluster.coordinator.suspect_node(worker.id).await;
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

---

## 3. Formal Verification Framework

### 3.1 State Invariants as First-Class Citizens

Every state has an invariant predicate that must hold.

**Rust Implementation**:

```rust
pub trait StateInvariant {
    /// Check if invariant holds for this state
    fn check(&self) -> Result<(), InvariantViolation>;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Severity if violated
    fn severity(&self) -> Severity;
}

/// System-level invariants
pub struct SystemInvariants;

impl SystemInvariants {
    pub fn single_coordinator() -> impl StateInvariant {
        Invariant {
            name: "SingleCoordinator",
            check: |system: &SystemState| {
                let coordinator_count = system.nodes.iter()
                    .filter(|n| n.is_coordinator())
                    .count();

                if coordinator_count == 1 {
                    Ok(())
                } else {
                    Err(InvariantViolation::MultipleCoordinators {
                        count: coordinator_count,
                    })
                }
            },
            severity: Severity::Critical,
            description: "Exactly one coordinator must exist",
        }
    }

    pub fn quorum_operational() -> impl StateInvariant {
        Invariant {
            name: "QuorumOperational",
            check: |system: &SystemState| {
                let operational = system.operational_node_count();
                let quorum = system.quorum_size;

                if system.state == SystemState::SHUTDOWN || system.state == SystemState::TERMINATED {
                    return Ok(()); // Quorum not required when shutting down
                }

                if operational >= quorum {
                    Ok(())
                } else {
                    Err(InvariantViolation::QuorumLost {
                        operational,
                        required: quorum,
                    })
                }
            },
            severity: Severity::Critical,
            description: "Quorum (2f+1) nodes must be operational",
        }
    }

    pub fn epoch_monotonic() -> impl StateInvariant {
        Invariant {
            name: "EpochMonotonic",
            check: |system: &SystemState| {
                let epochs: Vec<Epoch> = system.nodes.iter().map(|n| n.epoch).collect();
                let max_epoch = epochs.iter().max().unwrap();
                let min_epoch = epochs.iter().min().unwrap();

                // Allow at most 1 epoch skew during transition
                if max_epoch - min_epoch <= 1 {
                    Ok(())
                } else {
                    Err(InvariantViolation::EpochSkew {
                        min: *min_epoch,
                        max: *max_epoch,
                    })
                }
            },
            severity: Severity::High,
            description: "Epochs must be monotonically increasing across nodes",
        }
    }
}

/// Coordinator invariants
pub struct CoordinatorInvariants;

impl CoordinatorInvariants {
    pub fn assignment_covers_model() -> impl StateInvariant {
        Invariant {
            name: "AssignmentCoversModel",
            check: |coordinator: &CoordinatorState| {
                let Some(assignment) = &coordinator.work_assignment else {
                    return Ok(()); // No assignment yet
                };

                let total_layers = coordinator.model_config.total_layers;
                let assigned_layers: HashSet<usize> = assignment.assignments
                    .values()
                    .flat_map(|a| &a.layer_ids)
                    .copied()
                    .collect();

                if assigned_layers.len() == total_layers {
                    Ok(())
                } else {
                    Err(InvariantViolation::IncompleteAssignment {
                        assigned: assigned_layers.len(),
                        total: total_layers,
                    })
                }
            },
            severity: Severity::Critical,
            description: "Work assignment must cover all model layers",
        }
    }

    pub fn no_assignment_overlap() -> impl StateInvariant {
        Invariant {
            name: "NoAssignmentOverlap",
            check: |coordinator: &CoordinatorState| {
                let Some(assignment) = &coordinator.work_assignment else {
                    return Ok(());
                };

                let mut seen_layers = HashSet::new();

                for layer_assignment in assignment.assignments.values() {
                    for &layer_id in &layer_assignment.layer_ids {
                        if !seen_layers.insert(layer_id) {
                            return Err(InvariantViolation::OverlappingAssignment {
                                layer_id,
                            });
                        }
                    }
                }

                Ok(())
            },
            severity: Severity::Critical,
            description: "No layer assigned to multiple nodes",
        }
    }
}
```

### 3.2 Invariant Checking in Production

**Continuous Monitoring**:

```rust
pub struct InvariantMonitor {
    invariants: Vec<Box<dyn StateInvariant>>,
    check_interval: Duration,
    alert_sender: mpsc::Sender<Alert>,
}

impl InvariantMonitor {
    pub async fn run(&self, system: Arc<RwLock<SystemState>>) {
        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            interval.tick().await;

            let system = system.read().await;

            for invariant in &self.invariants {
                match invariant.check(&*system) {
                    Ok(()) => {
                        // Invariant holds
                        metrics::increment_counter!("invariant_checks_passed",
                            "invariant" => invariant.name());
                    }
                    Err(violation) => {
                        // Invariant violated
                        error!(
                            invariant = invariant.name(),
                            violation = ?violation,
                            severity = ?invariant.severity(),
                            "State invariant violated"
                        );

                        metrics::increment_counter!("invariant_violations",
                            "invariant" => invariant.name(),
                            "severity" => format!("{:?}", invariant.severity()));

                        // Send alert
                        self.alert_sender.send(Alert {
                            severity: invariant.severity(),
                            message: format!("Invariant {} violated: {:?}",
                                invariant.name(), violation),
                            timestamp: Utc::now(),
                        }).await.ok();

                        // Auto-remediation for certain violations
                        if invariant.severity() == Severity::Critical {
                            self.trigger_remediation(&violation, &system).await;
                        }
                    }
                }
            }
        }
    }

    async fn trigger_remediation(&self, violation: &InvariantViolation, system: &SystemState) {
        match violation {
            InvariantViolation::QuorumLost { .. } => {
                // Initiate emergency shutdown
                warn!("Quorum lost, initiating emergency shutdown");
                system.transition_to(SystemState::SHUTDOWN).await;
            }

            InvariantViolation::MultipleCoordinators { count } => {
                // Force new election
                warn!(count = count, "Multiple coordinators detected, forcing re-election");
                system.force_coordinator_election().await;
            }

            InvariantViolation::EpochSkew { min, max } => {
                // Synchronize epochs
                warn!(min = min, max = max, "Epoch skew detected, synchronizing");
                system.synchronize_epochs().await;
            }

            _ => {
                // Log and alert, but don't auto-fix
                warn!(violation = ?violation, "Invariant violation detected, manual intervention required");
            }
        }
    }
}
```

### 3.3 Proof Obligations

For each state transition, we define **proof obligations**—properties that must be provable before the transition is allowed.

**Example: COMPUTING → COMMITTING**

```rust
/// Proof obligation: Can only commit if quorum agrees on result
pub struct CommitmentProofObligation;

impl ProofObligation for CommitmentProofObligation {
    fn verify(&self, context: &TransitionContext) -> Result<(), ProofError> {
        let results = context.collected_results();

        // 1. Proof obligation: At least 2f+1 results received
        if results.len() < context.quorum_size() {
            return Err(ProofError::InsufficientResults {
                received: results.len(),
                required: context.quorum_size(),
            });
        }

        // 2. Proof obligation: All results have valid proofs
        for (node_id, result) in results {
            if !result.proof.verify() {
                return Err(ProofError::InvalidProof {
                    node: *node_id,
                    reason: "Signature verification failed",
                });
            }
        }

        // 3. Proof obligation: Quorum agrees on result hash
        let result_hashes: HashMap<ResultHash, usize> = results
            .values()
            .map(|r| r.hash)
            .fold(HashMap::new(), |mut acc, hash| {
                *acc.entry(hash).or_insert(0) += 1;
                acc
            });

        let (majority_hash, vote_count) = result_hashes.iter()
            .max_by_key(|(_, count)| *count)
            .ok_or(ProofError::NoConsensus)?;

        if *vote_count < context.quorum_size() {
            return Err(ProofError::NoConsensus {
                best_vote_count: *vote_count,
                required: context.quorum_size(),
            });
        }

        // 4. Proof obligation: Byzantine nodes identified
        let byzantine_nodes: Vec<NodeId> = results
            .iter()
            .filter(|(_, result)| result.hash != *majority_hash)
            .map(|(node_id, _)| *node_id)
            .collect();

        if byzantine_nodes.len() > context.max_byzantine() {
            return Err(ProofError::TooManyByzantine {
                detected: byzantine_nodes.len(),
                max_tolerated: context.max_byzantine(),
            });
        }

        // All proof obligations satisfied
        Ok(())
    }

    fn description(&self) -> &str {
        "Commitment proof: Quorum agreement on result with valid proofs"
    }
}
```

**Usage**:

```rust
impl CoordinatorStateMachine {
    pub async fn transition_to_committing(&mut self) -> Result<(), StateError> {
        let context = TransitionContext {
            current_state: self.state,
            target_state: NodeState::Committing,
            epoch: self.epoch,
            collected_results: self.results.clone(),
            quorum_size: self.quorum_size,
            max_byzantine: self.max_byzantine,
        };

        // Verify proof obligations before transition
        CommitmentProofObligation.verify(&context)?;

        // Proof satisfied, perform transition
        self.state = NodeState::Committing;

        Ok(())
    }
}
```

---

## 4. Consistency Under Failures

### 4.1 Single Node Failure

**Scenario**: One worker node crashes during COMPUTING phase.

**State Evolution**:

```
T0: System in COMPUTING, all nodes operational
    System: COMPUTING
    Worker A: COMPUTING
    Worker B: COMPUTING  ← CRASH
    Worker C: COMPUTING

T1: Coordinator detects missing heartbeats from Worker B
    φ_B increases > threshold
    Coordinator: marks B as SUSPECTED

T2: Coordinator confirms failure
    Quorum of nodes report B unresponsive
    System transitions: COMPUTING → DEGRADED
    Worker B: SUSPECT → FAILED

T3: Coordinator redistributes B's work
    Work assignment updated: B's layers → Worker D (standby)
    Worker D: JOINED → COMPUTING
    Checkpoint transferred to D

T4: Worker D completes recovery
    Recomputes from checkpoint
    Catches up to current position
    Worker D: COMPUTING → AGGREGATING

T5: All workers reach AGGREGATING
    System transitions: DEGRADED → COMMITTING
    Normal flow resumes
```

**State Consistency Properties Maintained**:

✅ **Epoch Consistency**: All workers remain on same epoch
✅ **Result Determinism**: Worker D recomputes exact same intermediate results as B
✅ **Quorum Safety**: 4 of 5 workers operational (quorum = 3)
✅ **Temporal Ordering**: State transitions respect causality

**Verification**:

```rust
#[tokio::test]
async fn test_single_node_failure_consistency() {
    let mut cluster = TestCluster::new(5, 1).await;
    cluster.bootstrap().await;

    // Start inference
    let task = cluster.submit_inference(/* ... */).await;

    // Wait for COMPUTING state
    cluster.wait_for_state(SystemState::COMPUTING).await;

    // Kill one worker
    cluster.kill_node(NodeId(1)).await;

    // System should transition to DEGRADED
    tokio::time::timeout(
        Duration::from_secs(1),
        cluster.wait_for_state(SystemState::DEGRADED)
    ).await.expect("Should transition to DEGRADED within 1s");

    // Recovery should occur
    tokio::time::timeout(
        Duration::from_secs(5),
        cluster.wait_for_state(SystemState::COMPUTING)  // Back to computing after recovery
    ).await.expect("Should recover within 5s");

    // Task should eventually complete
    let result = cluster.wait_for_task(task).await.unwrap();

    // Verify result is deterministic (same as without failure)
    let expected_result = compute_expected_result(/* ... */);
    assert_eq!(result.hash, expected_result.hash);
}
```

### 4.2 Coordinator Failure

**Scenario**: Coordinator crashes during COORDINATING phase.

**State Evolution**:

```
T0: System in COMPUTING, coordinator active
    Coordinator (Node 0): COORDINATING
    Workers: COMPUTING

T1: Coordinator crashes
    Coordinator (Node 0): COORDINATING → FAILED
    Workers: Notice missing heartbeats from coordinator

T2: Raft election timeout
    Workers initiate leader election
    Candidates: Nodes 1, 2, 3 (all eligible)

T3: New coordinator elected
    Node 1 wins election (highest term)
    Node 1 transitions: COMPUTING → ELECTED

T4: New coordinator reconstructs state
    Reads Raft log for:
      - Current epoch
      - Work assignments
      - Phase progress
    Node 1 transitions: ELECTED → COORDINATING

T5: New coordinator resumes coordination
    Queries workers for current progress
    Continues monitoring computation
    System remains in COMPUTING state (transparent failover)

T6: Computation completes normally
    Workers report to new coordinator
    System transitions: COMPUTING → COMMITTING
```

**State Consistency Properties Maintained**:

✅ **Epoch Consistency**: New coordinator reads epoch from Raft log
✅ **Work Assignment**: Reconstructed from checkpoint
✅ **Progress Tracking**: Workers report current state
✅ **Single Coordinator**: Raft ensures only one leader per term

**Key Mechanism**: Raft's replicated log provides durable state storage.

**Verification**:

```rust
#[tokio::test]
async fn test_coordinator_failure_consistency() {
    let mut cluster = TestCluster::new(5, 1).await;
    cluster.bootstrap().await;

    let task = cluster.submit_inference(/* ... */).await;
    cluster.wait_for_state(SystemState::COMPUTING).await;

    // Kill coordinator
    let old_coordinator = cluster.coordinator_id();
    cluster.kill_node(old_coordinator).await;

    // New coordinator should be elected within 2 seconds
    let new_coordinator = tokio::time::timeout(
        Duration::from_secs(2),
        cluster.wait_for_new_coordinator()
    ).await.expect("New coordinator should be elected");

    assert_ne!(old_coordinator, new_coordinator);

    // Task should still complete
    let result = cluster.wait_for_task(task).await.unwrap();
    assert!(result.is_valid());

    // Verify state consistency
    let system_state = cluster.get_system_state().await;
    assert_eq!(system_state.epoch, cluster.get_epoch_from_raft_log().await);
}
```

### 4.3 Byzantine Node Behavior

**Scenario**: Malicious node reports incorrect computation results.

**State Evolution**:

```
T0: System in COMPUTING
    All nodes: COMPUTING
    Byzantine Node B: Prepares incorrect result

T1: Nodes complete computation
    Honest nodes compute: result_hash = 0xABCD...
    Byzantine Node B computes: result_hash = 0x1234... (wrong)

T2: Coordinator collects results
    3 nodes report: 0xABCD...
    1 node (B) reports: 0x1234...
    Coordinator detects disagreement

T3: Byzantine agreement protocol
    Coordinator proposes: 0xABCD... (majority)
    Quorum votes: 3 votes for 0xABCD...
    Node B's vote ignored (minority)

T4: Commitment
    Correct result 0xABCD... committed
    Node B marked as BYZANTINE_SUSPECTED

T5: Byzantine behavior confirmation
    Node B isolated after repeated violations
    System continues with reduced capacity
    Quorum still maintained (3 of 4 honest nodes)
```

**State Consistency Properties Maintained**:

✅ **Result Integrity**: Byzantine node cannot corrupt result (quorum voting)
✅ **Detection**: Disagreement detected during aggregation
✅ **Isolation**: Byzantine node marked and eventually removed
✅ **Safety**: Up to f Byzantine nodes tolerated without result corruption

**Verification**:

```rust
#[tokio::test]
async fn test_byzantine_node_detection() {
    let mut cluster = TestCluster::new(7, 2).await;  // f=2, can tolerate 2 Byzantine
    cluster.bootstrap().await;

    // Inject Byzantine behavior in Node 1
    cluster.make_node_byzantine(NodeId(1), ByzantineBehavior::IncorrectResults);

    // Submit inference
    let task = cluster.submit_inference(/* ... */).await;

    // Task should complete with correct result despite Byzantine node
    let result = cluster.wait_for_task(task).await.unwrap();
    let expected_result = compute_expected_result(/* ... */);
    assert_eq!(result.hash, expected_result.hash);

    // Verify Byzantine node was detected
    let byzantine_nodes = cluster.get_suspected_byzantine_nodes().await;
    assert!(byzantine_nodes.contains(&NodeId(1)));
}
```

---

## 5. Testing State Coherence

### 5.1 Coherence Test Categories

| Test Type           | Purpose                                    | Example                                      |
|---------------------|--------------------------------------------|----------------------------------------------|
| **Unit Tests**      | Single component invariants                | Worker state transitions valid               |
| **Integration Tests** | Multi-component coherence                | Worker-coordinator phase synchronization     |
| **Property Tests**  | Invariants hold under random inputs        | Epoch monotonicity across all scenarios      |
| **Chaos Tests**     | Coherence under failures                   | Network partition doesn't violate consistency|
| **Formal Verification** | Mathematical proof of properties       | TLA+ model checking                          |

### 5.2 Property-Based Coherence Tests

Using `proptest` to verify invariants:

```rust
#[cfg(test)]
mod coherence_proptests {
    use proptest::prelude::*;

    proptest! {
        /// Property: Epoch monotonicity holds under arbitrary transitions
        #[test]
        fn prop_epoch_monotonic(
            transitions in prop::collection::vec(valid_transition(), 1..100)
        ) {
            let mut system = SystemState::new();
            let mut prev_epoch = system.epoch();

            for transition in transitions {
                system.apply_transition(transition);
                let curr_epoch = system.epoch();

                prop_assert!(curr_epoch >= prev_epoch,
                    "Epoch decreased: {} → {}", prev_epoch, curr_epoch);

                prev_epoch = curr_epoch;
            }
        }

        /// Property: Vertical coherence maintained under failures
        #[test]
        fn prop_vertical_coherence_under_failures(
            cluster_size in 5usize..10,
            max_byzantine in 1usize..3,
            operations in prop::collection::vec(cluster_operation(), 1..50)
        ) {
            prop_assume!(cluster_size >= 2 * max_byzantine + 1);

            let mut cluster = TestCluster::new(cluster_size, max_byzantine);

            for op in operations {
                cluster.apply_operation(op);

                // Invariant: Vertical coherence always holds
                let system_state = cluster.system_state();
                for worker in cluster.workers() {
                    prop_assert!(
                        is_valid_worker_state_for_system(worker.state, system_state),
                        "Vertical coherence violated: system={:?}, worker={:?}",
                        system_state, worker.state
                    );
                }
            }
        }

        /// Property: Quorum safety under arbitrary failures
        #[test]
        fn prop_quorum_safety(
            cluster_size in 5usize..10,
            max_byzantine in 1usize..3,
            failure_sequence in prop::collection::vec(any::<usize>(), 0..10)
        ) {
            prop_assume!(cluster_size >= 2 * max_byzantine + 1);

            let mut cluster = TestCluster::new(cluster_size, max_byzantine);
            let quorum_size = 2 * max_byzantine + 1;

            for node_index in failure_sequence {
                let node_id = NodeId(node_index % cluster_size);
                cluster.kill_node(node_id);

                let operational = cluster.operational_count();
                let system_state = cluster.system_state();

                if operational < quorum_size {
                    // Invariant: System must shut down if quorum lost
                    prop_assert!(
                        matches!(system_state, SystemState::DEGRADED | SystemState::SHUTDOWN),
                        "System should shut down when quorum lost: operational={}, quorum={}",
                        operational, quorum_size
                    );
                } else {
                    // Invariant: System can continue if quorum maintained
                    prop_assert!(
                        !matches!(system_state, SystemState::SHUTDOWN),
                        "System should not shut down with quorum: operational={}, quorum={}",
                        operational, quorum_size
                    );
                }
            }
        }
    }
}
```

### 5.3 Linearizability Testing

Verify operations appear atomic and ordered:

```rust
#[tokio::test]
async fn test_linearizability_with_jepsen() {
    // Setup: 5-node cluster
    let cluster = TestCluster::new(5, 1).await;
    cluster.bootstrap().await;

    // Concurrent clients submitting requests
    let clients = 10;
    let requests_per_client = 20;

    let handles: Vec<_> = (0..clients)
        .map(|client_id| {
            let cluster = cluster.clone();
            tokio::spawn(async move {
                let mut results = Vec::new();
                for i in 0..requests_per_client {
                    let input = generate_input(client_id, i);
                    let result = cluster.infer(input).await.unwrap();
                    results.push((i, result));
                }
                results
            })
        })
        .collect();

    // Collect all results
    let mut all_results = Vec::new();
    for handle in handles {
        all_results.extend(handle.await.unwrap());
    }

    // Verify linearizability: exists a total order consistent with real-time order
    assert!(is_linearizable(&all_results), "Operations are not linearizable");
}

fn is_linearizable(results: &[(usize, InferenceResult)]) -> bool {
    // Sort by epoch (gives total order)
    let mut ordered = results.to_vec();
    ordered.sort_by_key(|(_, r)| r.epoch);

    // Check epochs are contiguous and monotonic
    for window in ordered.windows(2) {
        let (_, r1) = &window[0];
        let (_, r2) = &window[1];

        if r2.epoch < r1.epoch {
            return false;  // Not monotonic
        }
    }

    true
}
```

---

## 6. Operational Coherence Monitoring

### 6.1 Real-Time Coherence Dashboard

**Metrics to Display**:

1. **Vertical Coherence Score**: Percentage of components with valid state for system state
   ```
   vertical_coherence = (valid_components / total_components) × 100%

   Target: 100%
   Alert if < 95%
   ```

2. **Horizontal Coherence Lag**: Maximum phase distance between workers
   ```
   horizontal_lag = max(phase_distance(w_i, w_j)) for all worker pairs

   Target: ≤ 1
   Alert if > 2
   ```

3. **Temporal Staleness**: Maximum state timestamp lag
   ```
   temporal_staleness = max(coordinator_time - worker_time) for all workers

   Target: < 200ms
   Alert if > 500ms
   ```

4. **Invariant Violation Rate**: Violations per minute
   ```
   violation_rate = count(violations) / time_window

   Target: 0
   Alert if > 0
   ```

**Prometheus Queries**:

```promql
# Vertical coherence score
100 * (
  sum(butterfly_component_state_valid == 1) /
  sum(butterfly_component_state_total)
)

# Horizontal phase lag
max(butterfly_worker_phase_index) - min(butterfly_worker_phase_index)

# Temporal staleness
max(butterfly_coordinator_state_timestamp - butterfly_worker_state_timestamp)

# Invariant violations
sum(rate(butterfly_invariant_violations_total[1m]))
```

**Grafana Dashboard Layout**:

```
┌──────────────────────────────────────────────┐
│  System State: COMPUTING     Epoch: 42       │
│  Vertical Coherence: 100%    Violations: 0   │
└──────────────────────────────────────────────┘

┌────────────────────────┬─────────────────────┐
│ Worker States          │ Phase Distribution  │
│                        │                     │
│ COMPUTING:  3 nodes    │ [====||||||||]      │
│ AGGREGATING: 2 nodes   │ Assignment: 0       │
│ READY: 0 nodes         │ Computation: 3      │
│ FAILED: 0 nodes        │ Aggregation: 2      │
│                        │ Commitment: 0       │
└────────────────────────┴─────────────────────┘

┌──────────────────────────────────────────────┐
│ Temporal Staleness (ms)                      │
│ ████████████░░░░░░░░░░░░░░░░░░░░░░ 120ms    │
│ Worker 1: 80ms   Worker 4: 120ms             │
│ Worker 2: 95ms   Worker 5: 110ms             │
│ Worker 3: 105ms                              │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│ Invariant Status                             │
│ ✓ SingleCoordinator                          │
│ ✓ QuorumOperational (5/5)                    │
│ ✓ EpochMonotonic                             │
│ ✓ AssignmentCoversModel                      │
│ ✓ VerticalCoherence                          │
└──────────────────────────────────────────────┘
```

### 6.2 Alerting Rules

**Critical Alerts** (Page on-call):

```yaml
- alert: QuorumLost
  expr: butterfly_operational_nodes < butterfly_quorum_size
  for: 10s
  severity: critical
  description: "Cluster lost quorum: {{ $value }} operational nodes (need {{ $labels.quorum_size }})"

- alert: MultipleCoordinators
  expr: count(butterfly_node_role{role="coordinator"}) > 1
  for: 5s
  severity: critical
  description: "Multiple coordinators detected: {{ $value }} coordinators active"

- alert: InvariantViolation
  expr: rate(butterfly_invariant_violations_total[1m]) > 0
  for: 10s
  severity: critical
  description: "State invariant violated: {{ $labels.invariant }}"
```

**Warning Alerts** (Slack notification):

```yaml
- alert: VerticalCoherenceDegraded
  expr: butterfly_vertical_coherence_score < 95
  for: 30s
  severity: warning
  description: "Vertical coherence degraded: {{ $value }}%"

- alert: HorizontalPhaseLag
  expr: butterfly_horizontal_phase_lag > 1
  for: 1m
  severity: warning
  description: "Workers have high phase lag: {{ $value }}"

- alert: TemporalStaleness
  expr: max(butterfly_temporal_staleness_ms) > 500
  for: 30s
  severity: warning
  description: "Worker state stale: {{ $value }}ms lag"
```

---

## 7. Summary: The Coherence Guarantee

### 7.1 What Butterfly Promises

**Strong Guarantees** (always hold):
1. ✅ **Linearizable results**: Every inference appears atomic and ordered
2. ✅ **Deterministic computation**: Same (model, input) → same output
3. ✅ **No partial commits**: Either all 2f+1 nodes commit, or none do
4. ✅ **Quorum safety**: No progress without 2f+1 operational nodes
5. ✅ **Byzantine tolerance**: Up to f nodes can be malicious without corrupting results

**Weaker Guarantees** (hold with high probability):
1. ⚠️ **Bounded failure detection**: Failed nodes detected within ~300ms (99th percentile)
2. ⚠️ **Bounded recovery time**: Recovery completes within ~900ms (typical case)
3. ⚠️ **Bounded staleness**: Worker state lags coordinator by < 200ms (target)

**No Guarantees** (best-effort):
1. ❌ **Availability during partitions**: System may be unavailable (CP in CAP)
2. ❌ **Real-time bounds**: No hard real-time guarantees (soft real-time targets exist)
3. ❌ **Metrics consistency**: Telemetry may lag or be incomplete (eventual consistency)

### 7.2 Coherence as System Health

State coherence is not just a correctness property—it's a **health indicator**:

- **100% coherence**: System healthy, operating normally
- **95-99% coherence**: Minor issues, investigate warnings
- **<95% coherence**: Serious problems, system degraded
- **<80% coherence**: Critical issues, imminent failure likely

**Monitoring coherence** is monitoring the system's **cognitive integrity**—whether it "knows" what it's doing and all parts agree on reality.

### 7.3 The Coherence Story

Just as the state machine tells a story (see state_machine_design.md Section 9.1), **state coherence tells the story of agreement**:

- **Vertical coherence**: Parts agree with the whole
- **Horizontal coherence**: Peers agree with each other
- **Temporal coherence**: Past agrees with present
- **Causal coherence**: Effects agree with causes

When coherence breaks, the system's story becomes **incoherent**—different parts believe different things, and the system cannot reliably reason about its own behavior. **Maintaining coherence is maintaining the system's sanity.**

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Complete Specification
**Next Review**: After initial production deployment
