# Butterfly State Management Module Specifications

## Overview

This document specifies the **state management modules** for each crate in the Butterfly workspace. Each module provides a consistent interface for state manipulation, transition validation, persistence, and monitoring.

**Design Principles**:
1. **Type Safety**: Impossible states should be unrepresentable
2. **Explicit Transitions**: All state changes go through validated transition functions
3. **Auditability**: Every state change is logged and traceable
4. **Observability**: State is queryable at runtime for debugging and monitoring
5. **Testability**: State machines are unit-testable in isolation

---

## 1. Core State Types (butterfly-core)

### 1.1 Module Structure

```
crates/butterfly-core/src/
├── state/
│   ├── mod.rs              # Public API
│   ├── system.rs           # System-level state
│   ├── component.rs        # Component state traits
│   ├── transition.rs       # Transition validation
│   ├── invariants.rs       # Invariant checking
│   └── persistence.rs      # State serialization
└── lib.rs
```

### 1.2 System State Types

**File**: `crates/butterfly-core/src/state/system.rs`

```rust
//! System-level state definitions shared across all components

use serde::{Deserialize, Serialize};
use std::fmt;

/// The overall state of the Butterfly distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SystemState {
    /// System exists but not initialized
    Uninitialized,
    /// Discovering nodes and establishing cluster
    Bootstrapping,
    /// Operational and waiting for requests
    Ready,
    /// Executing distributed inference
    Computing,
    /// Operating with reduced capacity due to failures
    Degraded,
    /// Committing inference result via Byzantine agreement
    Committing,
    /// Gracefully shutting down
    Shutdown,
    /// Fully terminated
    Terminated,
}

impl SystemState {
    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            Self::Uninitialized => "System not initialized",
            Self::Bootstrapping => "Establishing cluster",
            Self::Ready => "Ready for inference requests",
            Self::Computing => "Executing distributed computation",
            Self::Degraded => "Operating with failures",
            Self::Committing => "Finalizing results",
            Self::Shutdown => "Shutting down",
            Self::Terminated => "Terminated",
        }
    }

    /// Check if state is operational (can process requests)
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Ready | Self::Computing | Self::Degraded | Self::Committing)
    }

    /// Check if state is terminal (no more transitions)
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// Check if state allows accepting new work
    pub fn accepts_new_work(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

impl fmt::Display for SystemState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Execution phase within a computation cycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Phase {
    /// Work assignment distribution
    Assignment = 0,
    /// Pipelined computation execution
    Computation = 1,
    /// Result aggregation at barriers
    Aggregation = 2,
    /// Byzantine agreement and commitment
    Commitment = 3,
}

impl Phase {
    /// Get next phase in sequence
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Assignment => Some(Self::Computation),
            Self::Computation => Some(Self::Aggregation),
            Self::Aggregation => Some(Self::Commitment),
            Self::Commitment => None, // Cycles back to Assignment with new epoch
        }
    }

    /// Get phase index for ordering
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Calculate distance between phases
    pub fn distance_to(&self, other: &Phase) -> usize {
        (other.index() as isize - self.index() as isize).unsigned_abs()
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Unique identifier for state transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionId(pub u64);

/// Trigger that caused a state transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionTrigger {
    /// Explicit command from operator
    Command(String),
    /// Automatic transition due to system event
    Event(String),
    /// Triggered by another component's state change
    Cascade { source: String, reason: String },
    /// Timeout or deadline reached
    Timeout { duration_ms: u64 },
    /// Failure detected
    Failure { node: String, evidence: String },
    /// Recovery completed
    Recovery { recovered_nodes: Vec<String> },
}

impl fmt::Display for TransitionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command(cmd) => write!(f, "Command: {}", cmd),
            Self::Event(evt) => write!(f, "Event: {}", evt),
            Self::Cascade { source, reason } => write!(f, "Cascade from {}: {}", source, reason),
            Self::Timeout { duration_ms } => write!(f, "Timeout after {}ms", duration_ms),
            Self::Failure { node, evidence } => write!(f, "Failure of {}: {}", node, evidence),
            Self::Recovery { recovered_nodes } => write!(f, "Recovery of {:?}", recovered_nodes),
        }
    }
}

/// Record of a state transition with full context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionRecord<S> {
    /// Unique transition identifier
    pub id: TransitionId,
    /// State before transition
    pub from: S,
    /// State after transition
    pub to: S,
    /// What triggered the transition
    pub trigger: TransitionTrigger,
    /// Monotonic timestamp (nanoseconds since system boot)
    pub timestamp: i64,
    /// Wall clock time for human readability
    pub wall_time: chrono::DateTime<chrono::Utc>,
    /// Epoch at time of transition
    pub epoch: u64,
    /// How long the transition took
    pub duration_us: u64,
}

impl<S: fmt::Debug> fmt::Display for StateTransitionRecord<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {:?} → {:?} ({}) [{}µs]",
            self.wall_time.format("%H:%M:%S%.3f"),
            self.from,
            self.to,
            self.trigger,
            self.duration_us
        )
    }
}
```

### 1.3 Component State Trait

**File**: `crates/butterfly-core/src/state/component.rs`

```rust
//! Trait for components that maintain state machines

use super::{StateTransitionRecord, TransitionTrigger};
use std::fmt::Debug;
use thiserror::Error;

/// Error types for state management
#[derive(Debug, Error)]
pub enum StateError {
    #[error("Invalid transition from {from:?} to {to:?}: {reason}")]
    InvalidTransition {
        from: String,
        to: String,
        reason: String,
    },

    #[error("Transition precondition failed: {0}")]
    PreconditionFailed(String),

    #[error("State invariant violated: {0}")]
    InvariantViolation(String),

    #[error("Concurrent modification detected")]
    ConcurrentModification,

    #[error("State persistence failed: {0}")]
    PersistenceError(String),
}

/// Trait for types that represent component state
pub trait ComponentState: Debug + Clone + PartialEq + Send + Sync + 'static {
    /// Type of component this state belongs to
    fn component_type(&self) -> &str;

    /// Human-readable description of this state
    fn description(&self) -> &str;

    /// Check if this state is a failure state
    fn is_failure_state(&self) -> bool {
        false
    }

    /// Check if this state is operational
    fn is_operational(&self) -> bool {
        !self.is_failure_state()
    }

    /// Check if this state is terminal (no more transitions possible)
    fn is_terminal(&self) -> bool {
        false
    }
}

/// Trait for components that manage state transitions
pub trait StateMachine: Send + Sync {
    /// The state type this machine manages
    type State: ComponentState;

    /// Get current state
    fn current_state(&self) -> Self::State;

    /// Attempt to transition to a new state
    fn transition_to(
        &mut self,
        new_state: Self::State,
        trigger: TransitionTrigger,
    ) -> Result<StateTransitionRecord<Self::State>, StateError>;

    /// Check if a transition is valid without performing it
    fn can_transition_to(&self, new_state: &Self::State) -> Result<(), StateError>;

    /// Get state transition history (last N transitions)
    fn transition_history(&self) -> &[StateTransitionRecord<Self::State>];

    /// Get time spent in current state
    fn time_in_current_state(&self) -> std::time::Duration;
}

/// Trait for state machines that support persistence
pub trait PersistentStateMachine: StateMachine {
    /// Serialize state to bytes
    fn serialize_state(&self) -> Result<Vec<u8>, StateError>;

    /// Deserialize state from bytes
    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), StateError>;

    /// Save state to persistent storage
    fn save_state(&self) -> Result<(), StateError> {
        let data = self.serialize_state()?;
        // Implementation-specific persistence logic
        // For now, return Ok - actual impl in specific crates
        let _ = data;
        Ok(())
    }

    /// Load state from persistent storage
    fn load_state(&mut self) -> Result<(), StateError> {
        // Implementation-specific loading logic
        // For now, return Ok - actual impl in specific crates
        Ok(())
    }
}
```

### 1.4 Transition Validation Framework

**File**: `crates/butterfly-core/src/state/transition.rs`

```rust
//! Transition validation and proof obligations

use super::{ComponentState, StateError};
use std::fmt::Debug;

/// A transition rule that validates state changes
pub trait TransitionRule<S: ComponentState> {
    /// Check if transition is allowed
    fn is_valid(&self, from: &S, to: &S) -> bool;

    /// Get reason why transition is invalid (if it is)
    fn invalid_reason(&self, from: &S, to: &S) -> Option<String>;

    /// Human-readable name for this rule
    fn name(&self) -> &str;
}

/// Set of transition rules for a state machine
pub struct TransitionRules<S: ComponentState> {
    rules: Vec<Box<dyn TransitionRule<S>>>,
}

impl<S: ComponentState> TransitionRules<S> {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: impl TransitionRule<S> + 'static) {
        self.rules.push(Box::new(rule));
    }

    /// Validate a transition against all rules
    pub fn validate(&self, from: &S, to: &S) -> Result<(), StateError> {
        for rule in &self.rules {
            if !rule.is_valid(from, to) {
                return Err(StateError::InvalidTransition {
                    from: format!("{:?}", from),
                    to: format!("{:?}", to),
                    reason: rule.invalid_reason(from, to)
                        .unwrap_or_else(|| format!("Transition rule '{}' failed", rule.name())),
                });
            }
        }
        Ok(())
    }
}

impl<S: ComponentState> Default for TransitionRules<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof obligation that must be satisfied before transition
pub trait ProofObligation<S: ComponentState>: Send + Sync {
    /// Verify proof obligation is satisfied
    fn verify(&self, context: &TransitionContext<S>) -> Result<(), StateError>;

    /// Description of what this proof obligation ensures
    fn description(&self) -> &str;
}

/// Context provided to proof obligations for verification
pub struct TransitionContext<S: ComponentState> {
    pub from: S,
    pub to: S,
    pub epoch: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

impl<S: ComponentState> TransitionContext<S> {
    pub fn new(from: S, to: S, epoch: u64) -> Self {
        Self {
            from,
            to,
            epoch,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}
```

---

## 2. Coordinator State Management (butterfly-coordination)

### 2.1 Module Addition

Add to `crates/butterfly-coordination/src/`:

```rust
// File: crates/butterfly-coordination/src/state/mod.rs

pub mod coordinator;
pub mod transition_rules;
pub mod invariants;

pub use coordinator::CoordinatorStateMachine;
```

### 2.2 Enhanced Coordinator State Machine

**File**: `crates/butterfly-coordination/src/state/coordinator.rs`

```rust
//! Enhanced coordinator state machine with full transition validation

use butterfly_core::state::{
    ComponentState, StateMachine, PersistentStateMachine, StateError,
    StateTransitionRecord, TransitionTrigger, TransitionId,
};
use butterfly_core::NodeId;
use crate::state_machine::NodeState;  // Existing type
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};

impl ComponentState for NodeState {
    fn component_type(&self) -> &str {
        "Coordinator"
    }

    fn description(&self) -> &str {
        match self {
            Self::Initializing => "Loading configuration and preparing",
            Self::Ready => "Monitoring cluster, ready for work",
            Self::Computing => "Coordinating distributed computation",
            Self::Aggregating => "Collecting and validating results",
            Self::Committing => "Finalizing Byzantine agreement",
            Self::Degraded => "Operating with reduced capacity",
            Self::Recovering => "Recovering from failures",
            Self::Failed => "Coordinator has failed",
        }
    }

    fn is_failure_state(&self) -> bool {
        matches!(self, Self::Failed)
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Enhanced coordinator state machine with validation
pub struct CoordinatorStateMachine {
    /// Current state
    state: Arc<RwLock<NodeState>>,
    /// Transition history (last 100 transitions)
    history: Arc<RwLock<VecDeque<StateTransitionRecord<NodeState>>>>,
    /// Transition validation rules
    rules: super::transition_rules::CoordinatorTransitionRules,
    /// When we entered current state
    state_entered_at: Arc<RwLock<std::time::Instant>>,
    /// Current epoch
    epoch: Arc<RwLock<u64>>,
    /// Node ID of this coordinator
    node_id: NodeId,
}

impl CoordinatorStateMachine {
    pub fn new(node_id: NodeId, initial_state: NodeState) -> Self {
        let mut history = VecDeque::new();
        history.push_back(StateTransitionRecord {
            id: TransitionId(0),
            from: initial_state,
            to: initial_state,
            trigger: TransitionTrigger::Event("Initialization".to_string()),
            timestamp: get_monotonic_time(),
            wall_time: chrono::Utc::now(),
            epoch: 0,
            duration_us: 0,
        });

        Self {
            state: Arc::new(RwLock::new(initial_state)),
            history: Arc::new(RwLock::new(history)),
            rules: super::transition_rules::CoordinatorTransitionRules::new(),
            state_entered_at: Arc::new(RwLock::new(std::time::Instant::now())),
            epoch: Arc::new(RwLock::new(0)),
            node_id,
        }
    }

    /// Force state change (for recovery scenarios only)
    pub fn force_state(&mut self, new_state: NodeState, reason: &str) {
        warn!(
            node_id = ?self.node_id,
            old_state = ?self.current_state(),
            new_state = ?new_state,
            reason = reason,
            "Forcing state change (bypassing validation)"
        );

        *self.state.write().unwrap() = new_state;
        *self.state_entered_at.write().unwrap() = std::time::Instant::now();
    }
}

impl StateMachine for CoordinatorStateMachine {
    type State = NodeState;

    fn current_state(&self) -> Self::State {
        *self.state.read().unwrap()
    }

    fn transition_to(
        &mut self,
        new_state: Self::State,
        trigger: TransitionTrigger,
    ) -> Result<StateTransitionRecord<Self::State>, StateError> {
        let start = std::time::Instant::now();
        let from = self.current_state();

        // Validate transition
        self.can_transition_to(&new_state)?;

        // Perform transition
        let transition_id = TransitionId(self.history.read().unwrap().len() as u64);
        let epoch = *self.epoch.read().unwrap();

        *self.state.write().unwrap() = new_state;
        *self.state_entered_at.write().unwrap() = std::time::Instant::now();

        let record = StateTransitionRecord {
            id: transition_id,
            from,
            to: new_state,
            trigger: trigger.clone(),
            timestamp: get_monotonic_time(),
            wall_time: chrono::Utc::now(),
            epoch,
            duration_us: start.elapsed().as_micros() as u64,
        };

        // Record transition
        let mut history = self.history.write().unwrap();
        history.push_back(record.clone());
        if history.len() > 100 {
            history.pop_front();
        }

        info!(
            node_id = ?self.node_id,
            transition = %record,
            "Coordinator state transition"
        );

        Ok(record)
    }

    fn can_transition_to(&self, new_state: &Self::State) -> Result<(), StateError> {
        let current = self.current_state();
        self.rules.validate(&current, new_state)
    }

    fn transition_history(&self) -> &[StateTransitionRecord<Self::State>] {
        // Note: This returns a slice which requires careful lifetime management
        // In practice, you'd return a Vec clone
        unimplemented!("Use get_transition_history() instead")
    }

    fn time_in_current_state(&self) -> std::time::Duration {
        self.state_entered_at.read().unwrap().elapsed()
    }
}

impl CoordinatorStateMachine {
    /// Get cloned transition history
    pub fn get_transition_history(&self) -> Vec<StateTransitionRecord<NodeState>> {
        self.history.read().unwrap().iter().cloned().collect()
    }

    /// Increment epoch (only allowed from Committing → Ready transition)
    pub fn increment_epoch(&mut self) -> u64 {
        let mut epoch = self.epoch.write().unwrap();
        *epoch += 1;
        *epoch
    }

    /// Get current epoch
    pub fn epoch(&self) -> u64 {
        *self.epoch.read().unwrap()
    }
}

impl PersistentStateMachine for CoordinatorStateMachine {
    fn serialize_state(&self) -> Result<Vec<u8>, StateError> {
        use serde::Serialize;

        #[derive(Serialize)]
        struct SerializedState {
            state: NodeState,
            epoch: u64,
            timestamp: i64,
        }

        let state = SerializedState {
            state: self.current_state(),
            epoch: self.epoch(),
            timestamp: get_monotonic_time(),
        };

        bincode::serialize(&state)
            .map_err(|e| StateError::PersistenceError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), StateError> {
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct SerializedState {
            state: NodeState,
            epoch: u64,
            timestamp: i64,
        }

        let restored: SerializedState = bincode::deserialize(data)
            .map_err(|e| StateError::PersistenceError(e.to_string()))?;

        *self.state.write().unwrap() = restored.state;
        *self.epoch.write().unwrap() = restored.epoch;
        *self.state_entered_at.write().unwrap() = std::time::Instant::now();

        Ok(())
    }
}

// Helper to get monotonic time
fn get_monotonic_time() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64
}
```

### 2.3 Transition Rules

**File**: `crates/butterfly-coordination/src/state/transition_rules.rs`

```rust
//! Coordinator-specific transition validation rules

use butterfly_core::state::{TransitionRule, TransitionRules};
use crate::state_machine::NodeState;

pub struct CoordinatorTransitionRules {
    rules: TransitionRules<NodeState>,
}

impl CoordinatorTransitionRules {
    pub fn new() -> Self {
        let mut rules = TransitionRules::new();

        // Add all transition rules
        rules.add_rule(InitializingToReady);
        rules.add_rule(ReadyToComputing);
        rules.add_rule(ComputingToAggregating);
        rules.add_rule(AggregatingToCommitting);
        rules.add_rule(CommittingToReady);
        rules.add_rule(AnyToDegraded);
        rules.add_rule(DegradedToRecovering);
        rules.add_rule(RecoveringToReady);

        Self { rules }
    }

    pub fn validate(&self, from: &NodeState, to: &NodeState) -> Result<(), butterfly_core::state::StateError> {
        self.rules.validate(from, to)
    }
}

impl Default for CoordinatorTransitionRules {
    fn default() -> Self {
        Self::new()
    }
}

// Individual transition rules

struct InitializingToReady;
impl TransitionRule<NodeState> for InitializingToReady {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Initializing, NodeState::Ready))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Can only transition to Ready from Initializing".to_string())
    }

    fn name(&self) -> &str {
        "InitializingToReady"
    }
}

struct ReadyToComputing;
impl TransitionRule<NodeState> for ReadyToComputing {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Ready, NodeState::Computing))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Can only start computing from Ready state".to_string())
    }

    fn name(&self) -> &str {
        "ReadyToComputing"
    }
}

struct ComputingToAggregating;
impl TransitionRule<NodeState> for ComputingToAggregating {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Computing, NodeState::Aggregating))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Must finish computing before aggregating".to_string())
    }

    fn name(&self) -> &str {
        "ComputingToAggregating"
    }
}

struct AggregatingToCommitting;
impl TransitionRule<NodeState> for AggregatingToCommitting {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Aggregating, NodeState::Committing))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Must aggregate results before committing".to_string())
    }

    fn name(&self) -> &str {
        "AggregatingToCommitting"
    }
}

struct CommittingToReady;
impl TransitionRule<NodeState> for CommittingToReady {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Committing, NodeState::Ready))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Must commit before returning to Ready".to_string())
    }

    fn name(&self) -> &str {
        "CommittingToReady"
    }
}

struct AnyToDegraded;
impl TransitionRule<NodeState> for AnyToDegraded {
    fn is_valid(&self, _from: &NodeState, to: &NodeState) -> bool {
        matches!(to, NodeState::Degraded)
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        None // Always allowed
    }

    fn name(&self) -> &str {
        "AnyToDegraded"
    }
}

struct DegradedToRecovering;
impl TransitionRule<NodeState> for DegradedToRecovering {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Degraded, NodeState::Recovering))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Can only start recovery from Degraded state".to_string())
    }

    fn name(&self) -> &str {
        "DegradedToRecovering"
    }
}

struct RecoveringToReady;
impl TransitionRule<NodeState> for RecoveringToReady {
    fn is_valid(&self, from: &NodeState, to: &NodeState) -> bool {
        matches!((from, to), (NodeState::Recovering, NodeState::Ready))
    }

    fn invalid_reason(&self, _from: &NodeState, _to: &NodeState) -> Option<String> {
        Some("Must finish recovery before returning to Ready".to_string())
    }

    fn name(&self) -> &str {
        "RecoveringToReady"
    }
}
```

---

## 3. Worker State Management

Similar patterns apply to worker nodes. The structure is:

```
crates/butterfly-worker/src/
├── state/
│   ├── mod.rs
│   ├── worker.rs              # Worker state machine
│   ├── transition_rules.rs    # Worker-specific transitions
│   └── partition_state.rs     # Partition lifecycle management
```

**Key differences**:
- Workers track partition loading states
- Workers maintain connection state to coordinator
- Workers have more fine-grained computation states

---

## 4. API State Management (butterfly-api)

### 4.1 Request State Machine

**File**: `crates/butterfly-api/src/state/request.rs`

```rust
//! State machine for inference requests

use butterfly_core::state::{ComponentState, StateMachine, StateError, StateTransitionRecord, TransitionTrigger};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    /// Request received and validated
    Queued,
    /// Assigned to execution slot
    Scheduled,
    /// Sent to workers for execution
    Dispatched,
    /// Currently executing
    Executing,
    /// Awaiting Byzantine validation
    Validating,
    /// Successfully completed
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled by client
    Cancelled,
}

impl ComponentState for RequestState {
    fn component_type(&self) -> &str {
        "InferenceRequest"
    }

    fn description(&self) -> &str {
        match self {
            Self::Queued => "Waiting in queue",
            Self::Scheduled => "Scheduled for execution",
            Self::Dispatched => "Dispatched to workers",
            Self::Executing => "Executing on cluster",
            Self::Validating => "Validating results",
            Self::Completed => "Successfully completed",
            Self::Failed => "Failed with error",
            Self::Cancelled => "Cancelled",
        }
    }

    fn is_failure_state(&self) -> bool {
        matches!(self, Self::Failed | Self::Cancelled)
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

pub struct InferenceRequestStateMachine {
    request_id: Uuid,
    state: Arc<RwLock<RequestState>>,
    state_entered_at: Arc<RwLock<std::time::Instant>>,
    created_at: std::time::Instant,
}

impl InferenceRequestStateMachine {
    pub fn new(request_id: Uuid) -> Self {
        Self {
            request_id,
            state: Arc::new(RwLock::new(RequestState::Queued)),
            state_entered_at: Arc::new(RwLock::new(std::time::Instant::now())),
            created_at: std::time::Instant::now(),
        }
    }

    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub fn total_time(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl StateMachine for InferenceRequestStateMachine {
    type State = RequestState;

    fn current_state(&self) -> Self::State {
        *self.state.read().unwrap()
    }

    fn transition_to(
        &mut self,
        new_state: Self::State,
        trigger: TransitionTrigger,
    ) -> Result<StateTransitionRecord<Self::State>, StateError> {
        let from = self.current_state();

        // Validate transitions
        let valid = match (from, new_state) {
            (RequestState::Queued, RequestState::Scheduled) => true,
            (RequestState::Scheduled, RequestState::Dispatched) => true,
            (RequestState::Dispatched, RequestState::Executing) => true,
            (RequestState::Executing, RequestState::Validating) => true,
            (RequestState::Validating, RequestState::Completed) => true,
            (_, RequestState::Failed) => true,  // Can fail from any state
            (_, RequestState::Cancelled) => !matches!(from, RequestState::Completed | RequestState::Failed),  // Can't cancel if done
            _ => false,
        };

        if !valid {
            return Err(StateError::InvalidTransition {
                from: format!("{:?}", from),
                to: format!("{:?}", new_state),
                reason: "Transition not allowed".to_string(),
            });
        }

        *self.state.write().unwrap() = new_state;
        *self.state_entered_at.write().unwrap() = std::time::Instant::now();

        Ok(StateTransitionRecord {
            id: butterfly_core::state::TransitionId(0),  // Simplified
            from,
            to: new_state,
            trigger,
            timestamp: 0,
            wall_time: chrono::Utc::now(),
            epoch: 0,
            duration_us: 0,
        })
    }

    fn can_transition_to(&self, new_state: &Self::State) -> Result<(), StateError> {
        // Simplified validation
        Ok(())
    }

    fn transition_history(&self) -> &[StateTransitionRecord<Self::State>] {
        &[]  // Simplified
    }

    fn time_in_current_state(&self) -> std::time::Duration {
        self.state_entered_at.read().unwrap().elapsed()
    }
}
```

---

## 5. Integration and Usage

### 5.1 Example: Coordinator Using State Machine

```rust
// In coordinator main loop
use butterfly_coordination::state::CoordinatorStateMachine;
use butterfly_core::state::{StateMachine, TransitionTrigger};

pub struct Coordinator {
    state_machine: CoordinatorStateMachine,
    // ... other fields
}

impl Coordinator {
    pub async fn handle_inference_request(&mut self, request: InferenceRequest) -> Result<(), Error> {
        // Transition to Computing state
        self.state_machine.transition_to(
            NodeState::Computing,
            TransitionTrigger::Event("Inference request received".to_string())
        )?;

        // Assign work to workers
        let assignment = self.create_work_assignment(&request)?;
        self.distribute_assignment(assignment).await?;

        // Transition to Aggregating
        self.state_machine.transition_to(
            NodeState::Aggregating,
            TransitionTrigger::Event("Workers acknowledged assignment".to_string())
        )?;

        // ... rest of inference logic

        // Transition back to Ready
        self.state_machine.transition_to(
            NodeState::Ready,
            TransitionTrigger::Event("Inference completed".to_string())
        )?;

        // Increment epoch
        let new_epoch = self.state_machine.increment_epoch();
        tracing::info!(epoch = new_epoch, "Epoch advanced");

        Ok(())
    }

    pub async fn handle_node_failure(&mut self, failed_node: NodeId) -> Result<(), Error> {
        // Transition to Degraded
        self.state_machine.transition_to(
            NodeState::Degraded,
            TransitionTrigger::Failure {
                node: format!("{:?}", failed_node),
                evidence: "Heartbeat timeout".to_string(),
            }
        )?;

        // Initiate recovery
        self.start_recovery_protocol(failed_node).await?;

        self.state_machine.transition_to(
            NodeState::Recovering,
            TransitionTrigger::Event("Recovery initiated".to_string())
        )?;

        Ok(())
    }
}
```

### 5.2 State Monitoring Endpoint

```rust
// In API server
use axum::{Json, extract::State};

#[derive(Serialize)]
struct StateResponse {
    state: String,
    description: String,
    time_in_state_ms: u64,
    epoch: u64,
    recent_transitions: Vec<TransitionSummary>,
}

#[derive(Serialize)]
struct TransitionSummary {
    from: String,
    to: String,
    trigger: String,
    timestamp: String,
}

async fn get_coordinator_state(
    State(coordinator): State<Arc<RwLock<Coordinator>>>
) -> Json<StateResponse> {
    let coordinator = coordinator.read().unwrap();
    let state = coordinator.state_machine.current_state();
    let history = coordinator.state_machine.get_transition_history();

    Json(StateResponse {
        state: format!("{:?}", state),
        description: state.description().to_string(),
        time_in_state_ms: coordinator.state_machine.time_in_current_state().as_millis() as u64,
        epoch: coordinator.state_machine.epoch(),
        recent_transitions: history.iter().rev().take(10).map(|t| TransitionSummary {
            from: format!("{:?}", t.from),
            to: format!("{:?}", t.to),
            trigger: format!("{}", t.trigger),
            timestamp: t.wall_time.to_rfc3339(),
        }).collect(),
    })
}
```

---

## 6. Summary

This specification provides:

1. ✅ **Type-safe state machines** in `butterfly-core`
2. ✅ **Transition validation** via rules and proof obligations
3. ✅ **Audit trails** through transition history
4. ✅ **Persistence support** for crash recovery
5. ✅ **Observable state** via monitoring APIs
6. ✅ **Testable components** with clear interfaces

**Next Steps**:
1. Implement these modules in each crate
2. Add comprehensive unit tests for state machines
3. Integrate with existing coordination code
4. Add metrics and observability
5. Document state machine behavior in API docs

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Implementation Specification
**Dependencies**: Requires `butterfly-core` state module implemented first
