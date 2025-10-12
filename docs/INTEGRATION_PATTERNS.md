# Butterfly Integration Patterns

## Design Patterns Used in Integration

This document provides concrete code examples for each integration pattern, demonstrating how components will communicate and work together.

## Table of Contents

1. [Service Layer Pattern](#service-layer-pattern)
2. [Dependency Injection Pattern](#dependency-injection-pattern)
3. [Event-Driven Architecture](#event-driven-architecture)
4. [Circuit Breaker Pattern](#circuit-breaker-pattern)
5. [Retry with Exponential Backoff](#retry-with-exponential-backoff)
6. [Observer Pattern for Metrics](#observer-pattern-for-metrics)
7. [Message Router Pattern](#message-router-pattern)
8. [State Machine Pattern](#state-machine-pattern)

---

## Service Layer Pattern

Each crate exposes a high-level service interface that encapsulates internal complexity.

### Pattern Structure

```rust
// Abstract interface (trait)
#[async_trait]
pub trait ServiceInterface: Send + Sync {
    async fn operation(&self, params: Params) -> Result<Output, Error>;
}

// Concrete implementation
pub struct ServiceImpl {
    internal_state: InternalState,
    dependencies: Vec<Arc<dyn OtherService>>,
}

impl ServiceImpl {
    pub fn new(config: Config, deps: Dependencies) -> Self {
        // Initialize with dependencies
    }
}

#[async_trait]
impl ServiceInterface for ServiceImpl {
    async fn operation(&self, params: Params) -> Result<Output, Error> {
        // Implementation
    }
}
```

### Example: PartitionService

**File: `crates/butterfly-partition/src/service.rs`**

```rust
use async_trait::async_trait;
use butterfly_core::{LayerInfo, ModelPartition, PartitionStrategy};
use butterfly_metrics::{MetricsService, MetricsEvent};
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;

/// Public interface for partition service
#[async_trait]
pub trait PartitionInterface: Send + Sync {
    /// Compute optimal partitions for given layers and nodes
    async fn compute_partitions(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
        strategy: PartitionStrategy,
    ) -> Result<Vec<ModelPartition>, PartitionError>;

    /// Estimate quality metrics for a partition
    async fn estimate_quality(
        &self,
        partitions: &[ModelPartition],
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
    ) -> Result<PartitionQuality, PartitionError>;

    /// Invalidate cache for given model
    async fn invalidate_cache(&self, model_id: &str);
}

/// Service implementation with caching and metrics
pub struct PartitionService {
    // Strategy implementations
    strategies: Arc<RwLock<StrategyRegistry>>,

    // Cache for computed partitions
    cache: Arc<RwLock<LruCache<CacheKey, Vec<ModelPartition>>>>,

    // Metrics service for observability
    metrics: Arc<MetricsService>,

    // Configuration
    config: PartitionConfig,
}

#[derive(Hash, Eq, PartialEq)]
struct CacheKey {
    layers_hash: u64,
    nodes_hash: u64,
    strategy: PartitionStrategy,
}

impl PartitionService {
    pub fn new(
        config: PartitionConfig,
        metrics: Arc<MetricsService>,
    ) -> Self {
        let mut strategies = StrategyRegistry::new();
        strategies.register("uniform", Box::new(UniformPartitioner));
        strategies.register("load_balanced", Box::new(LoadBalancedPartitioner));

        Self {
            strategies: Arc::new(RwLock::new(strategies)),
            cache: Arc::new(RwLock::new(LruCache::new(config.cache_size))),
            metrics,
            config,
        }
    }

    /// Register a custom partitioning strategy
    pub async fn register_strategy(
        &self,
        name: &str,
        strategy: Box<dyn PartitionStrategyTrait>,
    ) {
        let mut strategies = self.strategies.write().await;
        strategies.register(name, strategy);
    }
}

#[async_trait]
impl PartitionInterface for PartitionService {
    async fn compute_partitions(
        &self,
        layers: &[LayerInfo],
        nodes: &[NodeCapability],
        strategy: PartitionStrategy,
    ) -> Result<Vec<ModelPartition>, PartitionError> {
        let start = std::time::Instant::now();

        // Check cache first
        let cache_key = CacheKey {
            layers_hash: compute_hash(layers),
            nodes_hash: compute_hash(nodes),
            strategy,
        };

        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get(&cache_key) {
                self.metrics.record_event(MetricsEvent::PartitionCacheHit);
                return Ok(cached.clone());
            }
        }

        // Cache miss - compute partitions
        self.metrics.record_event(MetricsEvent::PartitionCacheMiss);

        let strategies = self.strategies.read().await;
        let partitioner = strategies.get(&strategy)
            .ok_or(PartitionError::UnknownStrategy)?;

        let partitions = partitioner
            .partition(layers, nodes)
            .await?;

        // Store in cache
        {
            let mut cache = self.cache.write().await;
            cache.put(cache_key, partitions.clone());
        }

        // Record metrics
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
        // Use default estimator or specialized one
        let quality = estimate_partition_quality(partitions, layers, nodes);
        Ok(quality)
    }

    async fn invalidate_cache(&self, model_id: &str) {
        let mut cache = self.cache.write().await;
        cache.clear(); // Simple implementation - clear all
        // Production: selective invalidation based on model_id
    }
}

struct StrategyRegistry {
    strategies: HashMap<String, Box<dyn PartitionStrategyTrait>>,
}

impl StrategyRegistry {
    fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, strategy: Box<dyn PartitionStrategyTrait>) {
        self.strategies.insert(name.to_string(), strategy);
    }

    fn get(&self, strategy: &PartitionStrategy) -> Option<&Box<dyn PartitionStrategyTrait>> {
        let name = match strategy {
            PartitionStrategy::Uniform => "uniform",
            PartitionStrategy::LoadBalanced => "load_balanced",
            PartitionStrategy::TopologyAware => "topology_aware",
            _ => return None,
        };
        self.strategies.get(name)
    }
}

fn compute_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}
```

**Testing Strategy:**

```rust
// File: crates/butterfly-partition/tests/service_tests.rs

#[tokio::test]
async fn test_partition_service_caching() {
    let metrics = Arc::new(MetricsService::new(MetricsConfig::test_defaults()));
    let config = PartitionConfig {
        cache_size: 10,
        ..Default::default()
    };

    let service = PartitionService::new(config, metrics);

    let layers = create_test_layers(10);
    let nodes = create_test_nodes(3);

    // First call - cache miss
    let start = Instant::now();
    let partitions1 = service
        .compute_partitions(&layers, &nodes, PartitionStrategy::LoadBalanced)
        .await
        .unwrap();
    let first_duration = start.elapsed();

    // Second call - cache hit (should be much faster)
    let start = Instant::now();
    let partitions2 = service
        .compute_partitions(&layers, &nodes, PartitionStrategy::LoadBalanced)
        .await
        .unwrap();
    let second_duration = start.elapsed();

    assert_eq!(partitions1, partitions2);
    assert!(second_duration < first_duration / 10);
}

#[tokio::test]
async fn test_partition_service_custom_strategy() {
    struct CustomStrategy;

    #[async_trait]
    impl PartitionStrategyTrait for CustomStrategy {
        async fn partition(
            &self,
            layers: &[LayerInfo],
            nodes: &[NodeCapability],
        ) -> Result<Vec<ModelPartition>, PartitionError> {
            // Custom logic
            Ok(vec![])
        }
    }

    let metrics = Arc::new(MetricsService::new(MetricsConfig::test_defaults()));
    let service = PartitionService::new(PartitionConfig::default(), metrics);

    service.register_strategy("custom", Box::new(CustomStrategy)).await;

    // Can now use custom strategy
}
```

---

## Dependency Injection Pattern

Components receive their dependencies through constructors, enabling loose coupling and testability.

### Example: SchedulingService with Multiple Dependencies

**File: `crates/butterfly-schedule/src/service.rs`**

```rust
use async_trait::async_trait;
use butterfly_coordination::CoordinationInterface;
use butterfly_partition::PartitionInterface;
use butterfly_metrics::{MetricsService, MetricsEvent};
use butterfly_core::{LayerInfo, NodeId, PartitionStrategy};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};

#[async_trait]
pub trait SchedulingInterface: Send + Sync {
    async fn submit_task(&self, task: InferenceTask) -> Result<u64, SchedulingError>;
    async fn query_status(&self, task_id: u64) -> Result<TaskStatus, SchedulingError>;
    async fn cancel_task(&self, task_id: u64) -> Result<(), SchedulingError>;
}

pub struct SchedulingService {
    // Core scheduler logic
    scheduler: Arc<RwLock<Box<dyn Scheduler>>>,

    // Dependencies (injected)
    coordination: Arc<dyn CoordinationInterface>,
    partition: Arc<dyn PartitionInterface>,
    metrics: Arc<MetricsService>,

    // Internal state
    task_registry: Arc<RwLock<TaskRegistry>>,
    dispatch_tx: mpsc::UnboundedSender<DispatchCommand>,

    // Configuration
    config: SchedulingConfig,
}

impl SchedulingService {
    pub fn new(
        scheduler: Box<dyn Scheduler>,
        coordination: Arc<dyn CoordinationInterface>,
        partition: Arc<dyn PartitionInterface>,
        metrics: Arc<MetricsService>,
        config: SchedulingConfig,
    ) -> Self {
        let (dispatch_tx, dispatch_rx) = mpsc::unbounded_channel();
        let task_registry = Arc::new(RwLock::new(TaskRegistry::new()));

        let service = Self {
            scheduler: Arc::new(RwLock::new(scheduler)),
            coordination,
            partition,
            metrics,
            task_registry: Arc::clone(&task_registry),
            dispatch_tx,
            config,
        };

        // Spawn dispatch loop
        service.spawn_dispatch_loop(dispatch_rx);

        service
    }

    fn spawn_dispatch_loop(&self, mut dispatch_rx: mpsc::UnboundedReceiver<DispatchCommand>) {
        let scheduler = Arc::clone(&self.scheduler);
        let coordination = Arc::clone(&self.coordination);
        let partition = Arc::clone(&self.partition);
        let metrics = Arc::clone(&self.metrics);
        let task_registry = Arc::clone(&self.task_registry);

        tokio::spawn(async move {
            while let Some(command) = dispatch_rx.recv().await {
                match command {
                    DispatchCommand::DispatchNext => {
                        if let Err(e) = Self::dispatch_next_task(
                            &scheduler,
                            &coordination,
                            &partition,
                            &metrics,
                            &task_registry,
                        ).await {
                            error!("Dispatch failed: {}", e);
                        }
                    }
                    DispatchCommand::Shutdown => break,
                }
            }
        });
    }

    async fn dispatch_next_task(
        scheduler: &Arc<RwLock<Box<dyn Scheduler>>>,
        coordination: &Arc<dyn CoordinationInterface>,
        partition: &Arc<dyn PartitionInterface>,
        metrics: &Arc<MetricsService>,
        task_registry: &Arc<RwLock<TaskRegistry>>,
    ) -> Result<(), SchedulingError> {
        // Get next task from scheduler
        let task = {
            let mut sched = scheduler.write().await;
            sched.next_task(NodeId(0)) // Primary node scheduling
        };

        let task = match task {
            Some(t) => t,
            None => return Ok(()), // No tasks to dispatch
        };

        metrics.record_event(MetricsEvent::TaskDispatching {
            task_id: task.task_id,
        });

        // Update registry
        {
            let mut registry = task_registry.write().await;
            registry.update_status(task.task_id, TaskStatus::Partitioning);
        }

        // Get model layers for this task
        let layers = get_model_layers(&task)?;
        let nodes = get_available_nodes().await?;

        // Compute partitions
        let partitions = partition
            .compute_partitions(&layers, &nodes, PartitionStrategy::LoadBalanced)
            .await
            .map_err(|e| SchedulingError::PartitionFailed(e.to_string()))?;

        // Update registry
        {
            let mut registry = task_registry.write().await;
            registry.update_status(task.task_id, TaskStatus::Coordinating);
        }

        // Assign work through coordination
        coordination
            .assign_work(task.task_id, partitions)
            .await
            .map_err(|e| SchedulingError::CoordinationFailed(e.to_string()))?;

        // Update registry
        {
            let mut registry = task_registry.write().await;
            registry.update_status(task.task_id, TaskStatus::Computing);
        }

        metrics.record_event(MetricsEvent::TaskDispatched {
            task_id: task.task_id,
        });

        Ok(())
    }
}

#[async_trait]
impl SchedulingInterface for SchedulingService {
    async fn submit_task(&self, task: InferenceTask) -> Result<u64, SchedulingError> {
        let task_id = task.task_id;

        // Validate task
        validate_task(&task)?;

        // Enqueue in scheduler
        {
            let mut scheduler = self.scheduler.write().await;
            scheduler.enqueue(task.clone());
        }

        // Register task
        {
            let mut registry = self.task_registry.write().await;
            registry.register(task_id, TaskStatus::Queued);
        }

        // Record metric
        self.metrics.record_event(MetricsEvent::TaskSubmitted {
            task_id,
            priority: task.priority,
        });

        // Trigger dispatch
        let _ = self.dispatch_tx.send(DispatchCommand::DispatchNext);

        Ok(task_id)
    }

    async fn query_status(&self, task_id: u64) -> Result<TaskStatus, SchedulingError> {
        let registry = self.task_registry.read().await;
        registry.get_status(task_id)
            .ok_or(SchedulingError::TaskNotFound(task_id))
    }

    async fn cancel_task(&self, task_id: u64) -> Result<(), SchedulingError> {
        // Update registry
        {
            let mut registry = self.task_registry.write().await;
            registry.update_status(task_id, TaskStatus::Cancelled);
        }

        // Remove from scheduler queue if still there
        {
            let mut scheduler = self.scheduler.write().await;
            scheduler.remove_task(task_id);
        }

        self.metrics.record_event(MetricsEvent::TaskCancelled { task_id });

        Ok(())
    }
}

enum DispatchCommand {
    DispatchNext,
    Shutdown,
}

struct TaskRegistry {
    tasks: HashMap<u64, TaskStatus>,
}

impl TaskRegistry {
    fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    fn register(&mut self, task_id: u64, status: TaskStatus) {
        self.tasks.insert(task_id, status);
    }

    fn update_status(&mut self, task_id: u64, status: TaskStatus) {
        self.tasks.insert(task_id, status);
    }

    fn get_status(&self, task_id: u64) -> Option<TaskStatus> {
        self.tasks.get(&task_id).cloned()
    }
}

fn validate_task(task: &InferenceTask) -> Result<(), SchedulingError> {
    if task.input.is_empty() {
        return Err(SchedulingError::InvalidTask("Empty input".to_string()));
    }
    Ok(())
}

fn get_model_layers(task: &InferenceTask) -> Result<Vec<LayerInfo>, SchedulingError> {
    // TODO: Load model architecture and return layers
    // For now, return mock data
    Ok(vec![])
}

async fn get_available_nodes() -> Result<Vec<NodeCapability>, SchedulingError> {
    // TODO: Query coordination service for available nodes
    Ok(vec![])
}
```

**Mock for Testing:**

```rust
// File: crates/butterfly-schedule/src/mock.rs

pub struct MockCoordinationService {
    assignments: Arc<RwLock<Vec<(u64, Vec<ModelPartition>)>>>,
}

impl MockCoordinationService {
    pub fn new() -> Self {
        Self {
            assignments: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn get_assignments(&self) -> Vec<(u64, Vec<ModelPartition>)> {
        self.assignments.read().await.clone()
    }
}

#[async_trait]
impl CoordinationInterface for MockCoordinationService {
    async fn assign_work(
        &self,
        task_id: u64,
        partitions: Vec<ModelPartition>,
    ) -> Result<WorkAssignment, CoordinationError> {
        let mut assignments = self.assignments.write().await;
        assignments.push((task_id, partitions.clone()));

        Ok(WorkAssignment {
            task_id,
            partitions,
            timestamp: SystemTime::now(),
        })
    }

    // ... other methods
}

#[tokio::test]
async fn test_scheduling_service_with_mocks() {
    let scheduler = Box::new(FifoScheduler::new());
    let coordination = Arc::new(MockCoordinationService::new());
    let partition = Arc::new(MockPartitionService::new());
    let metrics = Arc::new(MetricsService::new(MetricsConfig::test_defaults()));

    let service = SchedulingService::new(
        scheduler,
        coordination.clone(),
        partition,
        metrics,
        SchedulingConfig::default(),
    );

    let task = InferenceTask {
        task_id: 1,
        input: vec![1.0, 2.0, 3.0],
        priority: 0,
    };

    let task_id = service.submit_task(task).await.unwrap();

    // Wait for dispatch
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify coordination was called
    let assignments = coordination.get_assignments().await;
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].0, task_id);
}
```

---

## Event-Driven Architecture

Components communicate through events published to a broadcast channel.

### Example: Metrics Event Bus

**File: `crates/butterfly-metrics/src/service.rs`**

```rust
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;

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

pub struct MetricsService {
    // Event channel for broadcasting metrics
    event_tx: broadcast::Sender<MetricsEvent>,

    // Collector for aggregating metrics
    collector: Arc<RwLock<Box<dyn MetricsCollector>>>,

    // Configuration
    config: MetricsConfig,
}

impl MetricsService {
    pub fn new(config: MetricsConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.channel_capacity);

        let collector: Box<dyn MetricsCollector> = match config.collector_type.as_str() {
            "prometheus" => Box::new(PrometheusCollector::new()),
            _ => Box::new(InMemoryCollector::new()),
        };

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

                    info!("Metrics snapshot: {:?}", snapshot);
                }
            });
        }

        Self {
            event_tx,
            collector,
            config,
        }
    }

    /// Record a metrics event (non-blocking)
    pub fn record_event(&self, event: MetricsEvent) {
        // If channel is full, oldest events are dropped
        let _ = self.event_tx.send(event);
    }

    /// Get a subscriber to the metrics event stream
    pub fn subscribe(&self) -> broadcast::Receiver<MetricsEvent> {
        self.event_tx.subscribe()
    }

    /// Query current metrics
    pub async fn query(&self, query: MetricsQuery) -> MetricsSnapshot {
        let collector = self.collector.read().await;
        collector.query(query)
    }

    /// Export all metrics
    pub async fn export_all(&self) -> MetricsSnapshot {
        let collector = self.collector.read().await;
        collector.snapshot()
    }

    /// Export Prometheus-formatted metrics
    pub async fn export_prometheus(&self) -> String {
        let collector = self.collector.read().await;
        collector.export_prometheus()
    }
}

// Trait for different collector implementations
pub trait MetricsCollector: Send + Sync {
    fn process_event(&mut self, event: MetricsEvent);
    fn query(&self, query: MetricsQuery) -> MetricsSnapshot;
    fn snapshot(&self) -> MetricsSnapshot;
    fn export_prometheus(&self) -> String;
}

// In-memory collector using histograms and counters
pub struct InMemoryCollector {
    task_latencies: Histogram,
    task_counts: HashMap<String, u64>, // status -> count
    partition_durations: Histogram,
    consensus_rounds: Histogram,
    bytes_transferred: u64,
    node_failures: Vec<(NodeId, SystemTime)>,
}

impl InMemoryCollector {
    pub fn new() -> Self {
        Self {
            task_latencies: Histogram::new(),
            task_counts: HashMap::new(),
            partition_durations: Histogram::new(),
            consensus_rounds: Histogram::new(),
            bytes_transferred: 0,
            node_failures: Vec::new(),
        }
    }
}

impl MetricsCollector for InMemoryCollector {
    fn process_event(&mut self, event: MetricsEvent) {
        match event {
            MetricsEvent::TaskCompleted { latency_ms, .. } => {
                self.task_latencies.record(latency_ms);
                *self.task_counts.entry("completed".to_string()).or_insert(0) += 1;
            }
            MetricsEvent::TaskFailed { .. } => {
                *self.task_counts.entry("failed".to_string()).or_insert(0) += 1;
            }
            MetricsEvent::PartitionComputed { duration_ms, .. } => {
                self.partition_durations.record(duration_ms);
            }
            MetricsEvent::ConsensusReached { rounds, .. } => {
                self.consensus_rounds.record(rounds as f64);
            }
            MetricsEvent::BytesTransferred { bytes, .. } => {
                self.bytes_transferred += bytes as u64;
            }
            MetricsEvent::NodeFailure { node_id, timestamp, .. } => {
                self.node_failures.push((node_id, timestamp));
            }
            _ => {}
        }
    }

    fn query(&self, query: MetricsQuery) -> MetricsSnapshot {
        // Implement query logic
        MetricsSnapshot::default()
    }

    fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            task_latency_p50: self.task_latencies.percentile(0.5),
            task_latency_p95: self.task_latencies.percentile(0.95),
            task_latency_p99: self.task_latencies.percentile(0.99),
            tasks_completed: *self.task_counts.get("completed").unwrap_or(&0),
            tasks_failed: *self.task_counts.get("failed").unwrap_or(&0),
            partition_duration_avg: self.partition_durations.mean(),
            consensus_rounds_avg: self.consensus_rounds.mean(),
            bytes_transferred: self.bytes_transferred,
            node_failures: self.node_failures.len(),
        }
    }

    fn export_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "# HELP task_latency_seconds Task completion latency\n\
             # TYPE task_latency_seconds histogram\n\
             task_latency_seconds{{quantile=\"0.5\"}} {}\n\
             task_latency_seconds{{quantile=\"0.95\"}} {}\n\
             task_latency_seconds{{quantile=\"0.99\"}} {}\n",
            self.task_latencies.percentile(0.5) / 1000.0,
            self.task_latencies.percentile(0.95) / 1000.0,
            self.task_latencies.percentile(0.99) / 1000.0,
        ));

        output.push_str(&format!(
            "# HELP tasks_total Total tasks processed\n\
             # TYPE tasks_total counter\n\
             tasks_total{{status=\"completed\"}} {}\n\
             tasks_total{{status=\"failed\"}} {}\n",
            self.task_counts.get("completed").unwrap_or(&0),
            self.task_counts.get("failed").unwrap_or(&0),
        ));

        output.push_str(&format!(
            "# HELP bytes_transferred_total Total bytes transferred\n\
             # TYPE bytes_transferred_total counter\n\
             bytes_transferred_total {}\n",
            self.bytes_transferred,
        ));

        output
    }
}

// Simple histogram implementation
struct Histogram {
    values: Vec<f64>,
}

impl Histogram {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn record(&mut self, value: f64) {
        self.values.push(value);
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((sorted.len() as f64) * p) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub task_latency_p50: f64,
    pub task_latency_p95: f64,
    pub task_latency_p99: f64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub partition_duration_avg: f64,
    pub consensus_rounds_avg: f64,
    pub bytes_transferred: u64,
    pub node_failures: usize,
}

pub enum MetricsQuery {
    TaskLatency { since: SystemTime },
    SystemUtilization,
    NodeHealth { node_id: NodeId },
}
```

**Usage in Other Components:**

```rust
// File: crates/butterfly-api/src/lib.rs

pub struct ApiService {
    scheduler: Arc<dyn SchedulingInterface>,
    metrics: Arc<MetricsService>,
}

impl ApiService {
    async fn submit_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, ApiError> {
        let start = Instant::now();

        let task = InferenceTask {
            task_id: generate_task_id(),
            input: request.input,
            priority: request.priority.unwrap_or(0),
        };

        // Record submission
        self.metrics.record_event(MetricsEvent::TaskSubmitted {
            task_id: task.task_id,
            priority: task.priority,
            timestamp: SystemTime::now(),
        });

        // Submit to scheduler
        match self.scheduler.submit_task(task).await {
            Ok(task_id) => {
                Ok(InferenceResponse {
                    task_id,
                    status: "submitted".to_string(),
                })
            }
            Err(e) => {
                self.metrics.record_event(MetricsEvent::TaskFailed {
                    task_id: task.task_id,
                    error: e.to_string(),
                });
                Err(ApiError::Scheduling(e))
            }
        }
    }
}
```

---

## Circuit Breaker Pattern

Prevent cascading failures by detecting and handling repeated errors.

### Example: Network Service with Circuit Breaker

**File: `crates/butterfly-comm/src/circuit_breaker.rs`**

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Failures detected, reject requests
    HalfOpen,    // Testing if service recovered
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<usize>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,     // Failures before opening
    pub timeout_duration: Duration,   // How long to keep circuit open
    pub success_threshold: usize,     // Successes in half-open before closing
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout_duration: Duration::from_secs(60),
            success_threshold: 2,
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Call a function with circuit breaker protection
    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        // Check circuit state
        let state = self.get_state().await;

        match state {
            CircuitState::Open => {
                // Check if timeout has elapsed
                if self.should_attempt_reset().await {
                    self.transition_to_half_open().await;
                } else {
                    return Err(CircuitBreakerError::CircuitOpen);
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {
                // Proceed with call
            }
        }

        // Execute the function
        match f.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(CircuitBreakerError::CallFailed(e))
            }
        }
    }

    async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }

    async fn should_attempt_reset(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;

        if let Some(last_failure) = *last_failure {
            last_failure.elapsed() >= self.config.timeout_duration
        } else {
            false
        }
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen;

        let mut failure_count = self.failure_count.write().await;
        *failure_count = 0;
    }

    async fn on_success(&self) {
        let current_state = self.get_state().await;

        match current_state {
            CircuitState::HalfOpen => {
                // Count successes in half-open state
                let mut failure_count = self.failure_count.write().await;
                *failure_count = 0;

                // If enough successes, close circuit
                let mut state = self.state.write().await;
                *state = CircuitState::Closed;
            }
            CircuitState::Closed => {
                // Reset failure count on success
                let mut failure_count = self.failure_count.write().await;
                *failure_count = 0;
            }
            _ => {}
        }
    }

    async fn on_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;

        let mut last_failure_time = self.last_failure_time.write().await;
        *last_failure_time = Some(Instant::now());

        if *failure_count >= self.config.failure_threshold {
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }
}

#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    CircuitOpen,
    CallFailed(E),
}
```

**Usage in NetworkService:**

```rust
// File: crates/butterfly-comm/src/service.rs

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError};

pub struct NetworkService {
    backend: Arc<dyn CommunicationBackend>,
    circuit_breakers: Arc<RwLock<HashMap<NodeId, CircuitBreaker>>>,
    metrics: Arc<MetricsService>,
}

impl NetworkService {
    pub fn new(
        backend: Arc<dyn CommunicationBackend>,
        metrics: Arc<MetricsService>,
    ) -> Self {
        Self {
            backend,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            metrics,
        }
    }

    async fn get_or_create_circuit_breaker(&self, node_id: NodeId) -> CircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;
        breakers.entry(node_id)
            .or_insert_with(|| CircuitBreaker::new(CircuitBreakerConfig::default()))
            .clone()
    }
}

#[async_trait]
impl NetworkInterface for NetworkService {
    async fn send(&self, target: NodeId, message: Message) -> Result<(), NetworkError> {
        let breaker = self.get_or_create_circuit_breaker(target).await;

        match breaker.call(self.backend.send(target, message.clone())).await {
            Ok(()) => {
                self.metrics.record_event(MetricsEvent::MessageSent {
                    to: target,
                    size_bytes: message.size(),
                    message_type: message.type_name(),
                });
                Ok(())
            }
            Err(CircuitBreakerError::CircuitOpen) => {
                warn!("Circuit breaker open for node {:?}", target);
                Err(NetworkError::CircuitOpen(target))
            }
            Err(CircuitBreakerError::CallFailed(e)) => {
                error!("Send failed to {:?}: {:?}", target, e);
                Err(e)
            }
        }
    }

    // ... other methods
}
```

---

## Retry with Exponential Backoff

Handle transient failures with intelligent retry logic.

### Example: Retry Logic in Network Send

**File: `crates/butterfly-comm/src/retry.rs`**

```rust
use std::time::Duration;
use tokio::time::sleep;

pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

pub async fn retry_with_backoff<F, T, E, Fut>(
    config: RetryConfig,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut backoff = config.initial_backoff_ms;

    loop {
        attempt += 1;

        match f().await {
            Ok(result) => {
                if attempt > 1 {
                    info!("Operation succeeded after {} attempts", attempt);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt >= config.max_attempts {
                    error!("Operation failed after {} attempts: {}", attempt, e);
                    return Err(e);
                }

                warn!("Attempt {} failed: {}. Retrying in {}ms...",
                      attempt, e, backoff);

                sleep(Duration::from_millis(backoff)).await;

                // Exponential backoff with cap
                backoff = (backoff as f64 * config.backoff_multiplier) as u64;
                backoff = backoff.min(config.max_backoff_ms);
            }
        }
    }
}
```

**Usage:**

```rust
// File: crates/butterfly-comm/src/service.rs

impl NetworkService {
    async fn send_with_retry(
        &self,
        target: NodeId,
        message: Message,
    ) -> Result<(), NetworkError> {
        let backend = Arc::clone(&self.backend);
        let message_clone = message.clone();

        retry_with_backoff(
            RetryConfig::default(),
            || async {
                backend.send(target, message_clone.clone()).await
            }
        ).await
    }
}
```

---

## Message Router Pattern

Route different message types to appropriate handlers.

### Example: Coordination Message Router

**File: `crates/butterfly-coordination/src/router.rs`**

```rust
use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;

/// Handler trait for processing specific message types
#[async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle(&self, message: CoordinationMessage) -> Result<(), CoordinationError>;
}

/// Router that dispatches messages to registered handlers
pub struct MessageRouter {
    handlers: HashMap<String, Arc<dyn MessageHandler>>,
}

impl MessageRouter {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register<H>(&mut self, message_type: &str, handler: H)
    where
        H: MessageHandler + 'static,
    {
        self.handlers.insert(
            message_type.to_string(),
            Arc::new(handler),
        );
    }

    pub async fn route(&self, message: CoordinationMessage) -> Result<(), CoordinationError> {
        let message_type = message.type_name();

        if let Some(handler) = self.handlers.get(message_type) {
            handler.handle(message).await
        } else {
            warn!("No handler registered for message type: {}", message_type);
            Ok(())
        }
    }
}

// Example handlers

struct WorkAssignmentHandler {
    state_machine: Arc<RwLock<CoordinationStateMachine>>,
}

#[async_trait]
impl MessageHandler for WorkAssignmentHandler {
    async fn handle(&self, message: CoordinationMessage) -> Result<(), CoordinationError> {
        if let CoordinationMessage::WorkAssignment(assignment) = message {
            let mut sm = self.state_machine.write().await;
            sm.apply_work_assignment(assignment)?;
            Ok(())
        } else {
            Err(CoordinationError::InvalidMessage)
        }
    }
}

struct BarrierHandler {
    barrier: Arc<BarrierCoordinator>,
}

#[async_trait]
impl MessageHandler for BarrierHandler {
    async fn handle(&self, message: CoordinationMessage) -> Result<(), CoordinationError> {
        if let CoordinationMessage::BarrierReady(node_id, hash) = message {
            self.barrier.node_ready(node_id, hash).await?;
            Ok(())
        } else {
            Err(CoordinationError::InvalidMessage)
        }
    }
}

// Usage in CoordinationService

impl CoordinationService {
    pub fn new(...) -> Self {
        let mut router = MessageRouter::new();

        router.register("work_assignment", WorkAssignmentHandler {
            state_machine: Arc::clone(&state_machine),
        });

        router.register("barrier_ready", BarrierHandler {
            barrier: Arc::clone(&barrier),
        });

        // ... register other handlers

        Self {
            router,
            // ... other fields
        }
    }

    pub async fn handle_message(
        &self,
        message: CoordinationMessage,
    ) -> Result<(), CoordinationError> {
        self.router.route(message).await
    }
}
```

---

## State Machine Pattern

Manage complex state transitions with formal state machines.

### Example: Task State Machine

**File: `crates/butterfly-schedule/src/state_machine.rs`**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskState {
    Submitted,
    Queued,
    Partitioning,
    Coordinating,
    Computing,
    Aggregating,
    Committing,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum TaskEvent {
    Submit,
    Enqueue,
    StartPartitioning,
    PartitionComplete,
    StartCoordinating,
    CoordinationComplete,
    StartComputing,
    ComputationComplete,
    StartAggregating,
    AggregationComplete,
    StartCommitting,
    CommitComplete,
    Fail(String),
    Cancel,
}

pub struct TaskStateMachine {
    current_state: TaskState,
    task_id: u64,
}

impl TaskStateMachine {
    pub fn new(task_id: u64) -> Self {
        Self {
            current_state: TaskState::Submitted,
            task_id,
        }
    }

    pub fn current_state(&self) -> &TaskState {
        &self.current_state
    }

    pub fn transition(&mut self, event: TaskEvent) -> Result<TaskState, StateMachineError> {
        let next_state = self.compute_next_state(event)?;

        info!(
            "Task {} transitioning: {:?} -> {:?}",
            self.task_id,
            self.current_state,
            next_state
        );

        self.current_state = next_state.clone();
        Ok(next_state)
    }

    fn compute_next_state(&self, event: TaskEvent) -> Result<TaskState, StateMachineError> {
        use TaskState::*;
        use TaskEvent::*;

        let next = match (&self.current_state, event) {
            // Valid transitions
            (Submitted, Enqueue) => Queued,
            (Queued, StartPartitioning) => Partitioning,
            (Partitioning, PartitionComplete) => Coordinating,
            (Coordinating, CoordinationComplete) => Computing,
            (Computing, ComputationComplete) => Aggregating,
            (Aggregating, AggregationComplete) => Committing,
            (Committing, CommitComplete) => Completed,

            // Failure transitions (from any state)
            (_, Fail(msg)) => Failed(msg),

            // Cancellation (only from Submitted, Queued, Partitioning)
            (Submitted | Queued | Partitioning, Cancel) => Cancelled,

            // Invalid transitions
            (current, event) => {
                return Err(StateMachineError::InvalidTransition {
                    from: current.clone(),
                    event: format!("{:?}", event),
                });
            }
        };

        Ok(next)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.current_state,
            TaskState::Completed | TaskState::Failed(_) | TaskState::Cancelled
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StateMachineError {
    #[error("Invalid transition from {:?} with event {}", .from, .event)]
    InvalidTransition {
        from: TaskState,
        event: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_path() {
        let mut sm = TaskStateMachine::new(1);

        assert_eq!(sm.current_state(), &TaskState::Submitted);

        sm.transition(TaskEvent::Enqueue).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Queued);

        sm.transition(TaskEvent::StartPartitioning).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Partitioning);

        sm.transition(TaskEvent::PartitionComplete).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Coordinating);

        sm.transition(TaskEvent::CoordinationComplete).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Computing);

        sm.transition(TaskEvent::ComputationComplete).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Aggregating);

        sm.transition(TaskEvent::AggregationComplete).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Committing);

        sm.transition(TaskEvent::CommitComplete).unwrap();
        assert_eq!(sm.current_state(), &TaskState::Completed);

        assert!(sm.is_terminal());
    }

    #[test]
    fn test_failure_path() {
        let mut sm = TaskStateMachine::new(2);

        sm.transition(TaskEvent::Enqueue).unwrap();
        sm.transition(TaskEvent::StartPartitioning).unwrap();

        sm.transition(TaskEvent::Fail("Partition failed".to_string())).unwrap();

        assert!(matches!(sm.current_state(), TaskState::Failed(_)));
        assert!(sm.is_terminal());
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = TaskStateMachine::new(3);

        // Can't go directly from Submitted to Computing
        let result = sm.transition(TaskEvent::StartComputing);
        assert!(result.is_err());
    }

    #[test]
    fn test_cancellation() {
        let mut sm = TaskStateMachine::new(4);

        sm.transition(TaskEvent::Enqueue).unwrap();
        sm.transition(TaskEvent::Cancel).unwrap();

        assert_eq!(sm.current_state(), &TaskState::Cancelled);

        // Can't transition after cancellation
        let result = sm.transition(TaskEvent::StartPartitioning);
        assert!(result.is_err());
    }
}
```

---

## Complete Example: End-to-End Integration

Let's see how all these patterns work together in a complete flow.

**File: `examples/complete_integration.rs`**

```rust
use butterfly_api::ApiService;
use butterfly_schedule::{SchedulingService, FifoScheduler};
use butterfly_partition::PartitionService;
use butterfly_coordination::CoordinationService;
use butterfly_comm::NetworkService;
use butterfly_metrics::MetricsService;
use butterfly_core::SystemConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = SystemConfig::from_file("config.toml")
        .unwrap_or_else(|_| SystemConfig::test_defaults());

    println!("Initializing Butterfly distributed inference system...");

    // Phase 1: Create metrics service (no dependencies)
    let metrics_service = Arc::new(MetricsService::new(config.metrics.clone()));
    println!("✓ Metrics service initialized");

    // Phase 2: Create network service
    let network_backend = create_network_backend(&config.network)?;
    let network_service = Arc::new(NetworkService::new(
        network_backend,
        Arc::clone(&metrics_service),
    ));
    println!("✓ Network service initialized");

    // Phase 3: Create partition service
    let partition_service = Arc::new(PartitionService::new(
        config.partition.clone(),
        Arc::clone(&metrics_service),
    ));
    println!("✓ Partition service initialized");

    // Phase 4: Create coordination service
    let coordination_service = Arc::new(CoordinationService::new(
        config.coordination.clone(),
        Arc::clone(&network_service),
        Arc::clone(&partition_service),
        Arc::clone(&metrics_service),
    ));
    println!("✓ Coordination service initialized");

    // Phase 5: Create scheduling service
    let scheduler = Box::new(FifoScheduler::new());
    let scheduling_service = Arc::new(SchedulingService::new(
        scheduler,
        Arc::clone(&coordination_service),
        Arc::clone(&partition_service),
        Arc::clone(&metrics_service),
        config.scheduling.clone(),
    ));
    println!("✓ Scheduling service initialized");

    // Phase 6: Create API service
    let api_service = ApiService::new(
        Arc::clone(&scheduling_service),
        Arc::clone(&metrics_service),
    );
    println!("✓ API service initialized");

    println!("\nSystem ready! Starting HTTP server on {}...", config.api.bind_address);

    // Start HTTP server
    api_service.serve(&config.api.bind_address).await?;

    Ok(())
}

fn create_network_backend(
    config: &NetworkConfig,
) -> Result<Arc<dyn CommunicationBackend>, Box<dyn std::error::Error>> {
    match config.backend_type.as_str() {
        "local" => Ok(Arc::new(LocalBackend)),
        "tcp" => Ok(Arc::new(TcpBackend::new(config)?)),
        "libp2p" => Ok(Arc::new(Libp2pBackend::new(config)?)),
        _ => Err("Unknown network backend type".into()),
    }
}
```

**Example configuration file: `config.toml`**

```toml
[api]
bind_address = "0.0.0.0:8080"
max_request_size_mb = 100
request_timeout_secs = 30

[scheduling]
scheduler_type = "priority"
max_queue_size = 1000
dispatch_batch_size = 10

[partition]
default_strategy = "load_balanced"
cache_size = 100
recompute_threshold = 0.1

[coordination]
node_id = 0
cluster_size = 3
max_byzantine = 1
phase_timeout_ms = 5000
checkpoint_interval = 10

[network]
backend_type = "tcp"
listen_address = "0.0.0.0:9000"
peer_addresses = ["192.168.1.101:9000", "192.168.1.102:9000"]
max_connections = 100
send_timeout_ms = 1000
retry_attempts = 3
retry_backoff_ms = 100

[metrics]
collector_type = "prometheus"
export_interval_secs = 60
retention_duration_secs = 3600
```

---

## Summary

This document provides concrete implementations of all major integration patterns used in Butterfly:

1. **Service Layer**: Clean abstractions with `*Interface` traits and `*Service` implementations
2. **Dependency Injection**: Constructor injection with Arc for shared ownership
3. **Event-Driven**: Broadcast channels for metrics and system events
4. **Circuit Breaker**: Prevent cascading failures with state-based protection
5. **Retry Logic**: Exponential backoff for transient failures
6. **Observer Pattern**: Metrics collection through event streams
7. **Message Router**: Type-based routing to specialized handlers
8. **State Machine**: Formal state transitions with validation

All patterns emphasize:
- **Testability**: Easy to mock and test in isolation
- **Observability**: Integrated metrics at every layer
- **Fault Tolerance**: Circuit breakers, retries, and graceful degradation
- **Type Safety**: Strong typing with async traits
- **Performance**: Zero-copy where possible, async throughout

Next steps for implementation:
1. Start with butterfly-core service traits
2. Implement metrics infrastructure
3. Build service wrappers for each crate
4. Create integration tests
5. Performance profiling
