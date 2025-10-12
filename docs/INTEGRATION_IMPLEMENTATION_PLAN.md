# Butterfly Integration Implementation Plan

## Overview

This document provides actionable implementation steps to achieve full integration of Butterfly's components. Each section specifies exact files to create/modify and the code to add.

## Current State

- 7 components with excellent individual implementations
- 0% integration between components
- All components are isolated islands

## Target State

- Full integration with clear data flows
- Service-oriented architecture with dependency injection
- Event-driven metrics collection
- Production-ready distributed inference system

---

## Phase 1: Foundation (butterfly-core)

### Task 1.1: Add Service Trait Definitions

**File: `crates/butterfly-core/src/service.rs`** (NEW)

```rust
//! Service traits for component integration

use async_trait::async_trait;
use crate::{LayerInfo, ModelPartition, NodeId, PartitionStrategy};

// Re-exports from other crates will go here
pub use crate::error::ButterflyError;

/// Common result type for all services
pub type ServiceResult<T> = Result<T, ButterflyError>;

/// Trait for lifecycle management
#[async_trait]
pub trait Service: Send + Sync {
    /// Initialize the service
    async fn start(&self) -> ServiceResult<()>;

    /// Gracefully shutdown the service
    async fn shutdown(&self) -> ServiceResult<()>;

    /// Check if service is healthy
    async fn health_check(&self) -> bool;
}
```

**Add to `crates/butterfly-core/src/lib.rs`:**

```rust
pub mod service;
pub mod error;
pub mod config;
pub mod system;

pub use service::{Service, ServiceResult};
pub use error::{ButterflyError, ApiError, SchedulingError, PartitionError, CoordinationError, NetworkError};
pub use config::SystemConfig;
pub use system::ButterflySystem;
```

### Task 1.2: Add Error Type Hierarchy

**File: `crates/butterfly-core/src/error.rs`** (NEW)

```rust
//! Unified error types for the Butterfly system

use thiserror::Error;
use crate::NodeId;

#[derive(Debug, Error)]
pub enum ButterflyError {
    #[error("API error: {0}")]
    Api(#[from] ApiError),

    #[error("Scheduling error: {0}")]
    Scheduling(#[from] SchedulingError),

    #[error("Partition error: {0}")]
    Partition(#[from] PartitionError),

    #[error("Coordination error: {0}")]
    Coordination(#[from] CoordinationError),

    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    #[error("System error: {0}")]
    System(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

#[derive(Debug, Error, Clone)]
pub enum ApiError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Task not found: {0}")]
    TaskNotFound(u64),

    #[error("Service unavailable")]
    ServiceUnavailable,

    #[error("Internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Error, Clone)]
pub enum SchedulingError {
    #[error("Queue full")]
    QueueFull,

    #[error("Task already exists: {0}")]
    TaskAlreadyExists(u64),

    #[error("Task not found: {0}")]
    TaskNotFound(u64),

    #[error("Coordination unavailable: {0}")]
    CoordinationUnavailable(String),

    #[error("Partition failed: {0}")]
    PartitionFailed(String),

    #[error("Invalid task: {0}")]
    InvalidTask(String),
}

#[derive(Debug, Error, Clone)]
pub enum PartitionError {
    #[error("No nodes available")]
    NoNodesAvailable,

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Unknown strategy")]
    UnknownStrategy,
}

#[derive(Debug, Error, Clone)]
pub enum CoordinationError {
    #[error("Consensus timeout")]
    ConsensusTimeout,

    #[error("Insufficient nodes")]
    InsufficientNodes,

    #[error("Byzantine violation: {0}")]
    ByzantineViolation(String),

    #[error("Phase transition failed: {0}")]
    PhaseTransitionFailed(String),

    #[error("Network failed: {0}")]
    NetworkFailed(String),

    #[error("Invalid message")]
    InvalidMessage,

    #[error("Internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Error, Clone)]
pub enum NetworkError {
    #[error("Connection failed to node {0:?}")]
    ConnectionFailed(NodeId),

    #[error("Send timeout")]
    SendTimeout,

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Node unreachable: {0:?}")]
    Unreachable(NodeId),

    #[error("Circuit breaker open for node {0:?}")]
    CircuitOpen(NodeId),

    #[error("No messages available")]
    NoMessages,
}
```

### Task 1.3: Add System Configuration

**File: `crates/butterfly-core/src/config.rs`** (NEW)

```rust
//! System configuration types

use serde::{Deserialize, Serialize};
use std::time::Duration;

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
    pub scheduler_type: String,
    pub max_queue_size: usize,
    pub dispatch_batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub default_strategy: String,
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
    pub backend_type: String,
    pub listen_address: String,
    pub peer_addresses: Vec<String>,
    pub max_connections: usize,
    pub send_timeout_ms: u64,
    pub retry_attempts: usize,
    pub retry_backoff_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub collector_type: String,
    pub export_interval_secs: u64,
    pub retention_duration_secs: u64,
    pub channel_capacity: usize,
}

impl SystemConfig {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
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
                channel_capacity: 10000,
            },
        }
    }
}
```

**Update `crates/butterfly-core/Cargo.toml`:**

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
async-trait = "0.1"
toml = "0.8"
```

---

## Phase 2: Metrics Infrastructure (butterfly-metrics)

### Task 2.1: Add MetricsEvent Types

**File: `crates/butterfly-metrics/src/events.rs`** (NEW)

```rust
//! Metrics event types

use butterfly_core::{NodeId, PartitionStrategy};
use butterfly_coordination::Phase;
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsEvent {
    TaskSubmitted {
        task_id: u64,
        priority: u32,
        timestamp: SystemTime,
    },
    TaskEnqueued {
        task_id: u64,
        queue_size: usize,
        timestamp: SystemTime,
    },
    TaskDispatching {
        task_id: u64,
    },
    TaskDispatched {
        task_id: u64,
    },
    PartitionComputed {
        num_partitions: usize,
        duration_ms: f64,
        strategy: PartitionStrategy,
    },
    PartitionCacheHit,
    PartitionCacheMiss,
    PhaseTransition {
        from: Phase,
        to: Phase,
        timestamp: SystemTime,
    },
    ConsensusReached {
        task_id: u64,
        rounds: usize,
        duration_ms: f64,
    },
    MessageSent {
        to: NodeId,
        size_bytes: usize,
        message_type: String,
    },
    MessageReceived {
        from: NodeId,
        size_bytes: usize,
        message_type: String,
    },
    BytesTransferred {
        from: NodeId,
        to: NodeId,
        bytes: usize,
        duration_ms: f64,
    },
    NodeFailure {
        node_id: NodeId,
        evidence: String,
        timestamp: SystemTime,
    },
    TaskCompleted {
        task_id: u64,
        latency_ms: f64,
        nodes_involved: Vec<NodeId>,
    },
    TaskFailed {
        task_id: u64,
        error: String,
    },
    TaskCancelled {
        task_id: u64,
    },
}
```

### Task 2.2: Create MetricsService

**File: `crates/butterfly-metrics/src/service.rs`** (NEW)

```rust
//! Metrics service implementation

use crate::{MetricsCollector, InMemoryCollector, MetricsEvent};
use butterfly_core::{MetricsConfig, Service, ServiceResult};
use tokio::sync::{broadcast, RwLock};
use std::sync::Arc;
use std::time::Duration;

pub struct MetricsService {
    event_tx: broadcast::Sender<MetricsEvent>,
    collector: Arc<RwLock<Box<dyn MetricsCollector>>>,
    config: MetricsConfig,
}

impl MetricsService {
    pub fn new(config: MetricsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.channel_capacity);

        let collector: Box<dyn MetricsCollector> = Box::new(InMemoryCollector::new());
        let collector = Arc::new(RwLock::new(collector));

        // Spawn aggregator task
        let collector_clone = Arc::clone(&collector);
        let mut event_rx = event_tx.subscribe();

        tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                let mut c = collector_clone.write().await;
                c.process_event(event);
            }
        });

        // Spawn periodic export task
        if config.export_interval_secs > 0 {
            let collector_clone = Arc::clone(&collector);
            let interval = config.export_interval_secs;

            tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(
                    Duration::from_secs(interval)
                );

                loop {
                    interval_timer.tick().await;

                    let c = collector_clone.read().await;
                    let snapshot = c.snapshot();

                    tracing::info!("Metrics snapshot: {:?}", snapshot);
                }
            });
        }

        Self {
            event_tx,
            collector,
            config,
        }
    }

    pub fn record_event(&self, event: MetricsEvent) {
        let _ = self.event_tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<MetricsEvent> {
        self.event_tx.subscribe()
    }

    pub async fn export_all(&self) -> crate::MetricsSnapshot {
        let collector = self.collector.read().await;
        collector.snapshot()
    }

    pub async fn export_prometheus(&self) -> String {
        let collector = self.collector.read().await;
        collector.export_prometheus()
    }
}

#[async_trait::async_trait]
impl Service for MetricsService {
    async fn start(&self) -> ServiceResult<()> {
        Ok(())
    }

    async fn shutdown(&self) -> ServiceResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> bool {
        true
    }
}
```

**Update `crates/butterfly-metrics/src/lib.rs`:**

```rust
pub mod events;
pub mod service;

pub use events::MetricsEvent;
pub use service::MetricsService;

// Keep existing MetricsCollector, InMemoryCollector, etc.
```

**Update `crates/butterfly-metrics/Cargo.toml`:**

```toml
[dependencies]
butterfly-core = { path = "../butterfly-core" }
butterfly-coordination = { path = "../butterfly-coordination" }
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
async-trait = "0.1"
tracing = "0.1"
```

---

## Phase 3: Network Service (butterfly-comm)

### Task 3.1: Add Service Interface

**File: `crates/butterfly-comm/src/interface.rs`** (NEW)

```rust
//! Network service interface

use async_trait::async_trait;
use butterfly_core::{NodeId, ServiceResult};
use crate::{Message, CommunicationError};

#[async_trait]
pub trait NetworkInterface: Send + Sync {
    async fn send(&self, target: NodeId, message: Message) -> Result<(), CommunicationError>;
    async fn broadcast(&self, message: Message) -> Result<(), CommunicationError>;
    async fn receive(&self) -> Result<Message, CommunicationError>;
}
```

### Task 3.2: Create NetworkService Wrapper

**File: `crates/butterfly-comm/src/service.rs`** (NEW)

```rust
//! Network service implementation

use crate::{CommunicationBackend, Message, NetworkInterface};
use butterfly_core::{NodeId, Service, ServiceResult};
use butterfly_metrics::{MetricsService, MetricsEvent};
use std::sync::Arc;
use async_trait::async_trait;

pub struct NetworkService {
    backend: Arc<dyn CommunicationBackend>,
    metrics: Arc<MetricsService>,
}

impl NetworkService {
    pub fn new(
        backend: Arc<dyn CommunicationBackend>,
        metrics: Arc<MetricsService>,
    ) -> Self {
        Self { backend, metrics }
    }
}

#[async_trait]
impl NetworkInterface for NetworkService {
    async fn send(&self, target: NodeId, message: Message) -> Result<(), crate::CommunicationError> {
        let size = estimate_message_size(&message);
        let message_type = message.type_name().to_string();

        let result = self.backend.send(target, message).await;

        if result.is_ok() {
            self.metrics.record_event(MetricsEvent::MessageSent {
                to: target,
                size_bytes: size,
                message_type,
            });
        }

        result
    }

    async fn broadcast(&self, message: Message) -> Result<(), crate::CommunicationError> {
        self.backend.broadcast(message).await
    }

    async fn receive(&self) -> Result<Message, crate::CommunicationError> {
        let message = self.backend.receive().await?;

        let size = estimate_message_size(&message);
        let message_type = message.type_name().to_string();

        if let Some(from) = message.source_node() {
            self.metrics.record_event(MetricsEvent::MessageReceived {
                from,
                size_bytes: size,
                message_type,
            });
        }

        Ok(message)
    }
}

#[async_trait]
impl Service for NetworkService {
    async fn start(&self) -> ServiceResult<()> {
        Ok(())
    }

    async fn shutdown(&self) -> ServiceResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> bool {
        true
    }
}

fn estimate_message_size(message: &Message) -> usize {
    // Simple estimation - in production use actual serialized size
    match message {
        Message::TensorData { data, .. } => data.len() * 4,
        _ => 128,
    }
}
```

**Update Message enum to have helper methods:**

**File: `crates/butterfly-comm/src/lib.rs`** (modify existing)

```rust
impl Message {
    pub fn type_name(&self) -> &str {
        match self {
            Message::TensorData { .. } => "tensor_data",
            Message::Heartbeat { .. } => "heartbeat",
            Message::Ack { .. } => "ack",
        }
    }

    pub fn source_node(&self) -> Option<NodeId> {
        match self {
            Message::TensorData { from, .. } => Some(*from),
            Message::Heartbeat { node_id } => Some(*node_id),
            _ => None,
        }
    }
}

pub mod interface;
pub mod service;

pub use interface::NetworkInterface;
pub use service::NetworkService;
```

**Update `crates/butterfly-comm/Cargo.toml`:**

```toml
[dependencies]
butterfly-core = { path = "../butterfly-core" }
butterfly-metrics = { path = "../butterfly-metrics" }
async-trait = "0.1"
# ... existing dependencies
```

---

## Phase 4: Partition Service (butterfly-partition)

### Task 4.1: Add Service Interface

**File: `crates/butterfly-partition/src/interface.rs`** (NEW)

```rust
//! Partition service interface

use async_trait::async_trait;
use butterfly_core::{LayerInfo, ModelPartition, PartitionStrategy, PartitionError};
use crate::{NodeCapability, PartitionQuality};

#[async_trait]
pub trait PartitionInterface: Send + Sync {
    async fn compute_partitions(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
        strategy: PartitionStrategy,
    ) -> Result<Vec<ModelPartition>, PartitionError>;

    async fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<PartitionQuality, PartitionError>;
}
```

### Task 4.2: Create PartitionService

**File: `crates/butterfly-partition/src/service.rs`** (NEW)

```rust
//! Partition service with caching and metrics

use crate::{
    NodeCapability, PartitionInterface, PartitionQuality, PartitionStrategyTrait,
    UniformPartitioner, LoadBalancedPartitioner, TopologyAwarePartitioner,
};
use butterfly_core::{
    LayerInfo, ModelPartition, PartitionConfig, PartitionError,
    PartitionStrategy, Service, ServiceResult,
};
use butterfly_metrics::{MetricsService, MetricsEvent};
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct PartitionService {
    strategies: Arc<RwLock<StrategyRegistry>>,
    cache: Arc<RwLock<LruCache<CacheKey, Vec<ModelPartition>>>>,
    metrics: Arc<MetricsService>,
    config: PartitionConfig,
}

#[derive(Hash, Eq, PartialEq)]
struct CacheKey {
    layers_hash: u64,
    nodes_hash: u64,
    strategy: PartitionStrategy,
}

struct StrategyRegistry {
    strategies: std::collections::HashMap<String, Box<dyn PartitionStrategyTrait>>,
}

impl PartitionService {
    pub fn new(config: PartitionConfig, metrics: Arc<MetricsService>) -> Self {
        let mut registry = StrategyRegistry::new();
        registry.register("uniform", Box::new(UniformPartitioner));
        registry.register("load_balanced", Box::new(LoadBalancedPartitioner));

        Self {
            strategies: Arc::new(RwLock::new(registry)),
            cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(config.cache_size).unwrap()
            ))),
            metrics,
            config,
        }
    }
}

#[async_trait::async_trait]
impl PartitionInterface for PartitionService {
    async fn compute_partitions(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
        strategy: PartitionStrategy,
    ) -> Result<Vec<ModelPartition>, PartitionError> {
        let start = std::time::Instant::now();

        let cache_key = CacheKey {
            layers_hash: compute_hash(layers),
            nodes_hash: compute_hash(nodes),
            strategy,
        };

        // Check cache
        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get(&cache_key) {
                self.metrics.record_event(MetricsEvent::PartitionCacheHit);
                return Ok(cached.clone());
            }
        }

        self.metrics.record_event(MetricsEvent::PartitionCacheMiss);

        // Compute partitions
        let strategies = self.strategies.read().await;
        let partitioner = strategies.get(&strategy)?;

        let partitions = partitioner.partition(layers, nodes)?;

        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.put(cache_key, partitions.clone());
        }

        let duration = start.elapsed();
        self.metrics.record_event(MetricsEvent::PartitionComputed {
            num_partitions: partitions.len(),
            duration_ms: duration.as_millis() as f64,
            strategy,
        });

        Ok(partitions)
    }

    async fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<PartitionQuality, PartitionError> {
        // Use default quality estimation
        Ok(PartitionQuality {
            load_balance: 0.9,
            communication_volume_mb: 0.0,
            peak_memory_gb: 0.0,
            estimated_latency_ms: 0.0,
        })
    }
}

impl StrategyRegistry {
    fn new() -> Self {
        Self {
            strategies: std::collections::HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, strategy: Box<dyn PartitionStrategyTrait>) {
        self.strategies.insert(name.to_string(), strategy);
    }

    fn get(&self, strategy: &PartitionStrategy) -> Result<&Box<dyn PartitionStrategyTrait>, PartitionError> {
        let name = match strategy {
            PartitionStrategy::Uniform => "uniform",
            PartitionStrategy::LoadBalanced => "load_balanced",
            PartitionStrategy::TopologyAware => "topology_aware",
            _ => return Err(PartitionError::UnknownStrategy),
        };
        self.strategies.get(name).ok_or(PartitionError::UnknownStrategy)
    }
}

fn compute_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}
```

**Update `crates/butterfly-partition/src/lib.rs`:**

```rust
pub mod interface;
pub mod service;

pub use interface::PartitionInterface;
pub use service::PartitionService;

// Existing exports...
```

**Update `crates/butterfly-partition/Cargo.toml`:**

```toml
[dependencies]
butterfly-core = { path = "../butterfly-core" }
butterfly-metrics = { path = "../butterfly-metrics" }
lru = "0.12"
# ... existing dependencies
```

---

## Phase 5: Coordination Service (butterfly-coordination)

### Task 5.1: Add Service Interface

**File: `crates/butterfly-coordination/src/interface.rs`** (NEW)

```rust
//! Coordination service interface

use async_trait::async_trait;
use butterfly_core::{NodeId, LayerInfo, CoordinationError};
use crate::{CoordinationMessage, WorkAssignment, Phase};

#[async_trait]
pub trait CoordinationInterface: Send + Sync {
    async fn assign_work(
        &self,
        task_id: u64,
        partitions: Vec<butterfly_core::ModelPartition>,
    ) -> Result<WorkAssignment, CoordinationError>;

    async fn handle_message(
        &self,
        from: NodeId,
        message: CoordinationMessage,
    ) -> Result<(), CoordinationError>;

    async fn current_phase(&self) -> Phase;
}
```

### Task 5.2: Create CoordinationService

**File: `crates/butterfly-coordination/src/service.rs`** (NEW)

```rust
//! Coordination service implementation

use crate::{
    DistributedCoordinator, CoordinationInterface, CoordinationMessage,
    WorkAssignment, Phase,
};
use butterfly_core::{NodeId, Service, ServiceResult, CoordinationError};
use butterfly_comm::NetworkInterface;
use butterfly_metrics::{MetricsService, MetricsEvent};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct CoordinationService {
    coordinator: Arc<RwLock<DistributedCoordinator>>,
    network: Arc<dyn NetworkInterface>,
    metrics: Arc<MetricsService>,
}

impl CoordinationService {
    pub fn new(
        config: butterfly_core::CoordinationConfig,
        network: Arc<dyn NetworkInterface>,
        metrics: Arc<MetricsService>,
    ) -> Self {
        let coordinator = Arc::new(RwLock::new(
            DistributedCoordinator::new(
                NodeId(config.node_id),
                config.cluster_size,
                config.max_byzantine,
            )
        ));

        Self {
            coordinator,
            network,
            metrics,
        }
    }

    pub async fn run_message_handler(&self) {
        loop {
            match self.network.receive().await {
                Ok(message) => {
                    // Process coordination messages
                    // (In real implementation, filter for coordination messages)
                }
                Err(e) => {
                    tracing::error!("Network receive error: {}", e);
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl CoordinationInterface for CoordinationService {
    async fn assign_work(
        &self,
        task_id: u64,
        partitions: Vec<butterfly_core::ModelPartition>,
    ) -> Result<WorkAssignment, CoordinationError> {
        let assignment = WorkAssignment {
            task_id,
            partitions: partitions.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        // Broadcast to all nodes
        // (Simplified - real implementation would convert to Message)

        Ok(assignment)
    }

    async fn handle_message(
        &self,
        from: NodeId,
        message: CoordinationMessage,
    ) -> Result<(), CoordinationError> {
        let coordinator = self.coordinator.write().await;
        coordinator.handle_message(message).await
    }

    async fn current_phase(&self) -> Phase {
        let coordinator = self.coordinator.read().await;
        coordinator.state_machine.read().await.current_phase()
    }
}
```

**Update `crates/butterfly-coordination/src/lib.rs`:**

```rust
pub mod interface;
pub mod service;

pub use interface::CoordinationInterface;
pub use service::CoordinationService;

// Existing exports...
```

**Update `crates/butterfly-coordination/Cargo.toml`:**

```toml
[dependencies]
butterfly-core = { path = "../butterfly-core" }
butterfly-comm = { path = "../butterfly-comm" }
butterfly-metrics = { path = "../butterfly-metrics" }
# ... existing dependencies
```

---

## Phase 6: Scheduling Service (butterfly-schedule)

### Task 6.1: Add Service Interface

**File: `crates/butterfly-schedule/src/interface.rs`** (NEW)

```rust
//! Scheduling service interface

use async_trait::async_trait;
use butterfly_core::SchedulingError;
use crate::{InferenceTask, TaskStatus};

#[async_trait]
pub trait SchedulingInterface: Send + Sync {
    async fn submit_task(&self, task: InferenceTask) -> Result<u64, SchedulingError>;
    async fn query_status(&self, task_id: u64) -> Result<TaskStatus, SchedulingError>;
    async fn cancel_task(&self, task_id: u64) -> Result<(), SchedulingError>;
}
```

### Task 6.2: Create SchedulingService

See detailed implementation in INTEGRATION_PATTERNS.md Section 2.

**Update `crates/butterfly-schedule/src/lib.rs`:**

```rust
pub mod interface;
pub mod service;

pub use interface::SchedulingInterface;
pub use service::SchedulingService;

// Existing exports...
```

---

## Phase 7: API Service (butterfly-api)

### Task 7.1: Update ApiService with Dependencies

**File: `crates/butterfly-api/src/service.rs`** (NEW)

```rust
//! API service implementation

use butterfly_schedule::{SchedulingInterface, InferenceTask};
use butterfly_metrics::{MetricsService, MetricsEvent};
use butterfly_core::{Service, ServiceResult, ApiError};
use axum::{Router, extract::State, Json, http::StatusCode};
use std::sync::Arc;

pub struct ApiService {
    scheduler: Arc<dyn SchedulingInterface>,
    metrics: Arc<MetricsService>,
}

impl ApiService {
    pub fn new(
        scheduler: Arc<dyn SchedulingInterface>,
        metrics: Arc<MetricsService>,
    ) -> Self {
        Self { scheduler, metrics }
    }

    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/health", axum::routing::get(health_check))
            .route("/inference", axum::routing::post(submit_inference))
            .route("/status/:task_id", axum::routing::get(query_status))
            .with_state(self)
    }

    pub async fn serve(self: Arc<Self>, addr: &str) -> ServiceResult<()> {
        let app = self.router();
        let listener = tokio::net::TcpListener::bind(addr).await
            .map_err(|e| butterfly_core::ButterflyError::System(e.to_string()))?;

        axum::serve(listener, app).await
            .map_err(|e| butterfly_core::ButterflyError::System(e.to_string()))?;

        Ok(())
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "healthy": true,
        "status": "ok"
    }))
}

async fn submit_inference(
    State(service): State<Arc<ApiService>>,
    Json(request): Json<super::InferenceRequest>,
) -> Result<Json<super::InferenceResponse>, StatusCode> {
    let task = InferenceTask {
        task_id: generate_task_id(),
        input: request.input,
        priority: request.priority.unwrap_or(0),
    };

    service.metrics.record_event(MetricsEvent::TaskSubmitted {
        task_id: task.task_id,
        priority: task.priority,
        timestamp: std::time::SystemTime::now(),
    });

    match service.scheduler.submit_task(task).await {
        Ok(task_id) => Ok(Json(super::InferenceResponse {
            task_id,
            status: "submitted".to_string(),
        })),
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

async fn query_status(
    State(service): State<Arc<ApiService>>,
    axum::extract::Path(task_id): axum::extract::Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match service.scheduler.query_status(task_id).await {
        Ok(status) => Ok(Json(serde_json::json!({
            "task_id": task_id,
            "status": format!("{:?}", status),
        }))),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

fn generate_task_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::SeqCst)
}
```

**Update `crates/butterfly-api/src/lib.rs`:**

```rust
pub mod service;

pub use service::ApiService;

// Existing types...
```

---

## Phase 8: System Integration (butterfly-core)

### Task 8.1: Create ButterflySystem

**File: `crates/butterfly-core/src/system.rs`** (NEW)

```rust
//! Main system orchestration

use crate::{SystemConfig, Service, ServiceResult, ButterflyError};
use butterfly_metrics::MetricsService;
use butterfly_comm::NetworkService;
use butterfly_partition::PartitionService;
use butterfly_coordination::CoordinationService;
use butterfly_schedule::SchedulingService;
use butterfly_api::ApiService;
use std::sync::Arc;

pub struct ButterflySystem {
    pub metrics: Arc<MetricsService>,
    pub network: Arc<NetworkService>,
    pub partition: Arc<PartitionService>,
    pub coordination: Arc<CoordinationService>,
    pub scheduling: Arc<SchedulingService>,
    pub api: Arc<ApiService>,
    config: SystemConfig,
}

impl ButterflySystem {
    pub async fn new(config: SystemConfig) -> Result<Self, ButterflyError> {
        // Phase 1: Metrics
        let metrics = Arc::new(MetricsService::new(config.metrics.clone()));

        // Phase 2: Network
        let network_backend = create_network_backend(&config.network)?;
        let network = Arc::new(NetworkService::new(network_backend, Arc::clone(&metrics)));

        // Phase 3: Partition
        let partition = Arc::new(PartitionService::new(
            config.partition.clone(),
            Arc::clone(&metrics),
        ));

        // Phase 4: Coordination
        let coordination = Arc::new(CoordinationService::new(
            config.coordination.clone(),
            Arc::clone(&network),
            Arc::clone(&metrics),
        ));

        // Phase 5: Scheduling
        let scheduler = Box::new(butterfly_schedule::FifoScheduler::new());
        let scheduling = Arc::new(SchedulingService::new(
            scheduler,
            Arc::clone(&coordination),
            Arc::clone(&partition),
            Arc::clone(&metrics),
            config.scheduling.clone(),
        ));

        // Phase 6: API
        let api = Arc::new(ApiService::new(
            Arc::clone(&scheduling),
            Arc::clone(&metrics),
        ));

        Ok(Self {
            metrics,
            network,
            partition,
            coordination,
            scheduling,
            api,
            config,
        })
    }

    pub async fn start(&self) -> ServiceResult<()> {
        self.metrics.start().await?;
        self.network.start().await?;
        // ...other services

        // Start HTTP server
        let api_clone = Arc::clone(&self.api);
        let bind_addr = self.config.api.bind_address.clone();

        tokio::spawn(async move {
            api_clone.serve(&bind_addr).await
        });

        Ok(())
    }

    pub async fn shutdown(self) -> ServiceResult<()> {
        // Graceful shutdown logic
        Ok(())
    }
}

fn create_network_backend(
    config: &crate::NetworkConfig,
) -> Result<Arc<dyn butterfly_comm::CommunicationBackend>, ButterflyError> {
    Ok(Arc::new(butterfly_comm::LocalBackend))
}
```

---

## Implementation Order

1. **Week 1**: Phase 1 (butterfly-core foundation)
2. **Week 2**: Phase 2 (metrics infrastructure)
3. **Week 3**: Phase 3 (network service)
4. **Week 4**: Phase 4 (partition service)
5. **Week 5**: Phase 5 (coordination service)
6. **Week 6**: Phase 6 (scheduling service)
7. **Week 7**: Phase 7 (API service updates)
8. **Week 8**: Phase 8 (system integration and testing)

## Testing Strategy

After each phase:
1. Unit tests for new code
2. Integration tests with mocks
3. End-to-end test for completed pipeline

## Success Criteria

- All 7 components communicate correctly
- Complete inference request flows end-to-end
- Metrics collected from all components
- Error propagation works correctly
- System can be initialized and shutdown cleanly
- Integration tests pass with >80% coverage

## Notes

- This is design specification - code examples are illustrative
- Actual implementation may require adjustments
- Each phase should be completed on a separate branch
- Merge to main only after tests pass
