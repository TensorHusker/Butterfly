# butterfly-cluster Crate Specification

## Overview

The `butterfly-cluster` crate implements cluster formation, membership management, and initialization orchestration for the Butterfly distributed inference system. It provides the core abstractions and protocols for nodes to discover each other, form a cluster, elect a coordinator, and prepare for distributed inference.

## Design Principles

1. **Separation of Concerns**: Cluster management is independent of inference logic
2. **Pluggable Discovery**: Support multiple discovery backends (static, DNS, Consul, etcd)
3. **Testability**: All components mockable for unit testing
4. **Observable**: Rich metrics and logging for operational visibility
5. **Fault-Tolerant**: Graceful handling of partial failures

## Module Structure

```
butterfly-cluster/
├── src/
│   ├── lib.rs                    # Public API
│   ├── node.rs                   # Node abstraction
│   ├── cluster.rs                # Cluster state management
│   ├── discovery/
│   │   ├── mod.rs                # Discovery trait
│   │   ├── static.rs             # Static seed list
│   │   ├── dns.rs                # DNS SRV records
│   │   ├── consul.rs             # Consul service discovery
│   │   └── etcd.rs               # etcd discovery
│   ├── membership/
│   │   ├── mod.rs                # Membership protocol
│   │   ├── join.rs               # Join/leave handling
│   │   └── health.rs             # Health monitoring
│   ├── election/
│   │   ├── mod.rs                # Leader election
│   │   ├── raft.rs               # Raft implementation
│   │   └── state_machine.rs     # Election state machine
│   ├── assignment/
│   │   ├── mod.rs                # Partition assignment
│   │   ├── algorithm.rs          # Assignment algorithms
│   │   └── distribution.rs       # Assignment distribution
│   ├── model/
│   │   ├── mod.rs                # Model management
│   │   ├── manifest.rs           # Model manifest handling
│   │   ├── loader.rs             # Weight loading
│   │   ├── source.rs             # Weight sources (S3, FS, HTTP)
│   │   └── validation.rs         # Cross-validation
│   ├── bootstrap/
│   │   ├── mod.rs                # Bootstrap orchestration
│   │   ├── sequence.rs           # Startup sequence
│   │   └── recovery.rs           # Failure recovery
│   └── transport/
│       ├── mod.rs                # Transport abstraction
│       ├── quic.rs               # QUIC transport
│       ├── grpc.rs               # gRPC transport
│       └── shm.rs                # Shared memory transport
├── tests/
│   ├── integration/              # Integration tests
│   ├── chaos/                    # Chaos engineering tests
│   └── fixtures/                 # Test fixtures
└── benches/                      # Performance benchmarks
```

## Core Types

### Node Representation

```rust
/// Represents a node in the Butterfly cluster
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique node identifier
    pub id: NodeId,

    /// Node's role (coordinator or worker)
    pub role: NodeRole,

    /// Network address
    pub address: SocketAddr,

    /// Hardware capabilities
    pub capabilities: NodeCapabilities,

    /// Current state
    pub state: NodeState,

    /// When this node joined the cluster
    pub joined_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Coordinator,
    Worker,
}

/// Possible states during initialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Cold,
    Bootstrap,
    Joining,
    Loading,
    Validating,
    Ready,
    Operational,
    Degraded,
    Recovering,
    Failed,
}
```

### Cluster Configuration

```rust
/// Cluster-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster name (for multi-cluster isolation)
    pub cluster_name: String,

    /// Discovery method
    pub discovery: DiscoveryConfig,

    /// Number of nodes required for quorum (2f+1)
    pub quorum_size: usize,

    /// Maximum Byzantine failures tolerated
    pub max_byzantine: usize,

    /// Model to load
    pub model: ModelConfig,

    /// Timeout configuration
    pub timeouts: TimeoutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DiscoveryConfig {
    Static {
        seed_nodes: Vec<SocketAddr>,
    },
    Dns {
        service_name: String,
        domain: String,
    },
    Consul {
        address: String,
        service_name: String,
    },
    Etcd {
        endpoints: Vec<String>,
        key_prefix: String,
    },
}
```

## API Design

### Cluster Builder

```rust
/// Builder for creating a cluster instance
pub struct ClusterBuilder {
    config: ClusterConfig,
    transport: Option<Box<dyn Transport>>,
    discovery: Option<Box<dyn Discovery>>,
    capabilities: Option<NodeCapabilities>,
}

impl ClusterBuilder {
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config,
            transport: None,
            discovery: None,
            capabilities: None,
        }
    }

    pub fn with_transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Some(Box::new(transport));
        self
    }

    pub fn with_discovery(mut self, discovery: impl Discovery + 'static) -> Self {
        self.discovery = Some(Box::new(discovery));
        self
    }

    pub fn with_capabilities(mut self, capabilities: NodeCapabilities) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    pub async fn build(self) -> Result<Cluster> {
        // Auto-detect capabilities if not provided
        let capabilities = match self.capabilities {
            Some(caps) => caps,
            None => detect_hardware_capabilities().await?,
        };

        // Initialize transport
        let transport = match self.transport {
            Some(t) => t,
            None => Box::new(QuicTransport::new(&self.config)?),
        };

        // Initialize discovery
        let discovery = match self.discovery {
            Some(d) => d,
            None => create_discovery_from_config(&self.config.discovery)?,
        };

        Cluster::new(self.config, transport, discovery, capabilities).await
    }
}
```

### Cluster Interface

```rust
/// Main cluster management interface
pub struct Cluster {
    config: ClusterConfig,
    local_node: Node,
    members: Arc<RwLock<HashMap<NodeId, Node>>>,
    coordinator: Arc<RwLock<Option<NodeId>>>,
    transport: Arc<dyn Transport>,
    discovery: Arc<dyn Discovery>,
    bootstrap: BootstrapOrchestrator,
    membership: MembershipManager,
    election: ElectionManager,
}

impl Cluster {
    /// Start cluster initialization sequence
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Starting cluster initialization");

        // Phase 1: Discovery
        let peers = self.discovery.discover_peers().await?;
        info!("Discovered {} peers", peers.len());

        // Phase 2: Join cluster
        self.join_cluster(peers).await?;

        // Phase 3: Wait for coordinator election
        self.wait_for_coordinator().await?;

        // Phase 4: Load model partition
        self.load_partition().await?;

        // Phase 5: Cross-validate
        self.validate_cluster().await?;

        info!("Cluster initialization complete");
        Ok(())
    }

    /// Get current cluster state
    pub fn state(&self) -> ClusterState {
        ClusterState {
            members: self.members.read().clone(),
            coordinator: *self.coordinator.read(),
            local_node: self.local_node.clone(),
        }
    }

    /// Check if cluster is ready for inference
    pub fn is_ready(&self) -> bool {
        self.local_node.state == NodeState::Operational
            && self.members.read().values()
                .filter(|n| n.state == NodeState::Operational)
                .count() >= self.config.quorum_size
    }

    /// Gracefully shut down this node
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Initiating graceful shutdown");

        // Notify cluster of departure
        self.membership.send_leave_notification().await?;

        // Wait for in-flight requests to complete
        self.wait_for_drain().await?;

        // Clean up resources
        self.transport.close().await?;

        info!("Shutdown complete");
        Ok(())
    }
}
```

## Discovery Trait

```rust
/// Trait for implementing different discovery mechanisms
#[async_trait]
pub trait Discovery: Send + Sync {
    /// Discover peers in the cluster
    async fn discover_peers(&self) -> Result<Vec<PeerInfo>>;

    /// Register this node with discovery service
    async fn register(&self, node: &Node) -> Result<()>;

    /// Deregister this node from discovery service
    async fn deregister(&self, node_id: NodeId) -> Result<()>;

    /// Watch for cluster membership changes
    async fn watch(&self) -> Result<impl Stream<Item = MembershipEvent>>;
}

pub enum MembershipEvent {
    NodeJoined(PeerInfo),
    NodeLeft(NodeId),
    NodeUpdated(PeerInfo),
}
```

### Discovery Implementations

```rust
/// Static seed list discovery
pub struct StaticDiscovery {
    seed_nodes: Vec<SocketAddr>,
}

#[async_trait]
impl Discovery for StaticDiscovery {
    async fn discover_peers(&self) -> Result<Vec<PeerInfo>> {
        let mut peers = Vec::new();

        for addr in &self.seed_nodes {
            match connect_and_handshake(addr).await {
                Ok(peer) => peers.push(peer),
                Err(e) => warn!("Failed to connect to seed {}: {}", addr, e),
            }
        }

        Ok(peers)
    }

    async fn register(&self, _node: &Node) -> Result<()> {
        // No-op for static discovery
        Ok(())
    }

    async fn deregister(&self, _node_id: NodeId) -> Result<()> {
        // No-op for static discovery
        Ok(())
    }

    async fn watch(&self) -> Result<impl Stream<Item = MembershipEvent>> {
        // Static discovery doesn't support watching
        Ok(stream::empty())
    }
}
```

## Transport Trait

```rust
/// Trait for implementing different transport mechanisms
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a message to a specific node
    async fn send(&self, to: NodeId, message: BootstrapMessage) -> Result<()>;

    /// Broadcast a message to all nodes
    async fn broadcast(&self, message: BootstrapMessage) -> Result<()>;

    /// Receive next message
    async fn recv(&self) -> Result<BootstrapMessage>;

    /// Create a stream for a specific message type
    async fn subscribe(&self, message_type: MessageType) -> Result<impl Stream<Item = BootstrapMessage>>;

    /// Close the transport
    async fn close(&self) -> Result<()>;
}
```

## Bootstrap Orchestrator

```rust
/// Orchestrates the bootstrap sequence
pub struct BootstrapOrchestrator {
    cluster: Arc<Cluster>,
    state_machine: StateMachine<NodeState>,
    timeout_config: TimeoutConfig,
}

impl BootstrapOrchestrator {
    /// Execute full bootstrap sequence
    pub async fn bootstrap(&mut self) -> Result<()> {
        self.transition_to(NodeState::Bootstrap).await?;

        // Step 1: Load configuration
        self.load_configuration().await?;

        // Step 2: Initialize network
        self.transition_to(NodeState::Joining).await?;
        self.initialize_network().await?;

        // Step 3: Discover peers
        let peers = self.discover_peers().await?;

        // Step 4: Join cluster
        self.join_cluster(peers).await?;

        // Step 5: Wait for partition assignment
        self.transition_to(NodeState::Loading).await?;
        let assignment = self.wait_for_assignment().await?;

        // Step 6: Load model weights
        self.load_model(assignment).await?;

        // Step 7: Validate
        self.transition_to(NodeState::Validating).await?;
        self.validate_weights().await?;

        // Step 8: Ready
        self.transition_to(NodeState::Ready).await?;

        Ok(())
    }

    async fn transition_to(&mut self, new_state: NodeState) -> Result<()> {
        info!("State transition: {:?} -> {:?}", self.state_machine.current(), new_state);

        self.state_machine.transition(new_state)?;

        // Report state change to coordinator
        if let Some(coordinator) = self.cluster.coordinator.read().as_ref() {
            self.cluster.transport.send(
                *coordinator,
                BootstrapMessage::state_change(new_state),
            ).await?;
        }

        Ok(())
    }
}
```

## Model Loading

```rust
/// Handles model loading and validation
pub struct ModelLoader {
    source: Box<dyn ModelSource>,
    manifest: ModelManifest,
    assignment: PartitionAssignment,
}

impl ModelLoader {
    pub fn new(
        source: Box<dyn ModelSource>,
        manifest: ModelManifest,
        assignment: PartitionAssignment,
    ) -> Self {
        Self {
            source,
            manifest,
            assignment,
        }
    }

    /// Load assigned partition
    pub async fn load(&self) -> Result<LoadedPartition> {
        let start = Instant::now();
        let total_bytes = self.assignment.required_files
            .iter()
            .map(|f| f.size_bytes)
            .sum();

        info!(
            "Loading partition: layers {}-{}, {} files, {} GB",
            self.assignment.layer_range.start,
            self.assignment.layer_range.end,
            self.assignment.required_files.len(),
            total_bytes / (1024 * 1024 * 1024),
        );

        let mut tensors = Vec::new();
        let mut bytes_loaded = 0u64;

        for file in &self.assignment.required_files {
            info!("Loading file: {}", file.path);

            // Download file
            let data = self.source.fetch(&file.path).await?;

            // Verify checksum
            self.verify_checksum(&data, &file.checksum)?;

            // Deserialize tensors
            let file_tensors = self.deserialize_weights(&data, &file.format)?;
            tensors.extend(file_tensors);

            bytes_loaded += data.len() as u64;

            // Report progress
            let progress = (bytes_loaded as f64 / total_bytes as f64) * 100.0;
            self.report_progress(progress).await?;
        }

        // Upload to GPU
        let gpu_tensors = self.upload_to_gpu(tensors).await?;

        // Compile kernels
        let kernels = self.compile_kernels().await?;

        // Run warmup inference
        let warmup_latency = self.run_warmup(&gpu_tensors, &kernels).await?;

        let load_time = start.elapsed();

        info!(
            "Load complete: {:.2}s, warmup latency: {:.2}ms",
            load_time.as_secs_f64(),
            warmup_latency.as_millis(),
        );

        Ok(LoadedPartition {
            assignment: self.assignment.clone(),
            tensors: gpu_tensors,
            kernels,
            memory_used_bytes: bytes_loaded,
            load_time,
            warmup_latency,
            checksum: self.compute_partition_checksum(&gpu_tensors),
        })
    }
}
```

## Election Manager

```rust
/// Manages Raft-based leader election
pub struct ElectionManager {
    raft: RaftStateMachine,
    transport: Arc<dyn Transport>,
    config: ClusterConfig,
}

impl ElectionManager {
    /// Start election process
    pub async fn start_election(&mut self) -> Result<ElectionResult> {
        info!("Starting leader election");

        // Increment term
        self.raft.increment_term();
        self.raft.vote_for_self();

        // Request votes from all peers
        let request = RequestVotePayload {
            term: self.raft.current_term(),
            candidate_id: self.raft.node_id(),
            last_log_index: self.raft.last_log_index(),
            last_log_term: self.raft.last_log_term(),
            pre_vote: false,
        };

        self.transport.broadcast(BootstrapMessage::request_vote(request)).await?;

        // Wait for votes with timeout
        let votes = self.collect_votes().await?;

        if votes.len() >= self.config.quorum_size {
            info!("Won election with {} votes", votes.len());
            self.raft.become_leader();
            Ok(ElectionResult::Won)
        } else {
            info!("Lost election: only {} votes", votes.len());
            self.raft.become_follower();
            Ok(ElectionResult::Lost)
        }
    }

    /// Handle incoming vote request
    pub async fn handle_vote_request(&mut self, request: RequestVotePayload) -> VoteResponse {
        // Check term
        if request.term < self.raft.current_term() {
            return VoteResponse {
                term: self.raft.current_term(),
                vote_granted: false,
                rejection_reason: Some(VoteRejectionReason::StaleTerm {
                    our_term: self.raft.current_term(),
                }),
            };
        }

        // Check if already voted
        if let Some(voted_for) = self.raft.voted_for() {
            if voted_for != request.candidate_id {
                return VoteResponse {
                    term: self.raft.current_term(),
                    vote_granted: false,
                    rejection_reason: Some(VoteRejectionReason::AlreadyVoted {
                        voted_for,
                    }),
                };
            }
        }

        // Check log up-to-date
        if !self.is_log_up_to_date(&request) {
            return VoteResponse {
                term: self.raft.current_term(),
                vote_granted: false,
                rejection_reason: Some(VoteRejectionReason::LogNotUpToDate {
                    our_last_index: self.raft.last_log_index(),
                    our_last_term: self.raft.last_log_term(),
                }),
            };
        }

        // Grant vote
        self.raft.vote_for(request.candidate_id);

        VoteResponse {
            term: self.raft.current_term(),
            vote_granted: true,
            rejection_reason: None,
        }
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_formation() {
        let config = test_cluster_config(3);

        let mut nodes = Vec::new();
        for i in 0..3 {
            let mut cluster = ClusterBuilder::new(config.clone())
                .with_transport(MockTransport::new())
                .with_discovery(MockDiscovery::new())
                .build()
                .await
                .unwrap();

            nodes.push(cluster);
        }

        // All nodes should eventually form cluster
        for node in &mut nodes {
            node.initialize().await.unwrap();
        }

        // Verify cluster has one coordinator
        let coordinators: Vec<_> = nodes.iter()
            .filter(|n| n.local_node.role == NodeRole::Coordinator)
            .collect();

        assert_eq!(coordinators.len(), 1);

        // Verify all nodes know about each other
        for node in &nodes {
            assert_eq!(node.members.read().len(), 3);
        }
    }

    #[tokio::test]
    async fn test_coordinator_failover() {
        // Start cluster with 3 nodes
        let mut cluster = start_test_cluster(3).await;

        // Kill coordinator
        let coordinator_id = cluster[0].coordinator.read().unwrap();
        cluster.retain(|n| n.local_node.id != coordinator_id);

        // Wait for new election
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify new coordinator elected
        let new_coordinator = cluster[0].coordinator.read().unwrap();
        assert_ne!(new_coordinator, coordinator_id);
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_full_initialization_flow() {
    // Start 3 nodes
    let config = ClusterConfig {
        cluster_name: "test-cluster".to_string(),
        quorum_size: 3,
        max_byzantine: 1,
        model: test_model_config(),
        // ...
    };

    let nodes = start_cluster_nodes(3, config).await;

    // Wait for initialization
    for node in &nodes {
        wait_for_ready(node, Duration::from_secs(60)).await.unwrap();
    }

    // Verify cluster is operational
    for node in &nodes {
        assert!(node.is_ready());
    }

    // Verify model loaded correctly
    for node in &nodes {
        let state = node.state();
        assert!(state.local_node.state == NodeState::Operational);
    }
}
```

### Chaos Tests

```rust
#[tokio::test]
async fn test_initialization_with_random_failures() {
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let num_nodes = rng.gen_range(3..=10);
        let failure_rate = rng.gen_range(0.0..=0.3);

        let result = run_initialization_with_failures(num_nodes, failure_rate).await;

        // Cluster should either fully initialize or fail gracefully
        assert!(result.is_ok() || result.is_err());

        if let Ok(cluster) = result {
            assert!(cluster.is_ready());
        }
    }
}
```

## Performance Benchmarks

```rust
#[bench]
fn bench_discovery_latency(b: &mut Bencher) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let discovery = StaticDiscovery::new(test_seed_nodes());

    b.iter(|| {
        rt.block_on(async {
            discovery.discover_peers().await.unwrap()
        })
    });
}

#[bench]
fn bench_election_time(b: &mut Bencher) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    b.iter(|| {
        rt.block_on(async {
            let cluster = start_test_cluster(5).await;
            measure_election_time(cluster).await
        })
    });
}
```

## Dependencies

```toml
[dependencies]
butterfly-core = { path = "../butterfly-core" }
butterfly-comm = { path = "../butterfly-comm" }

tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
bincode = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }

# Networking
quinn = { workspace = true }
tonic = { workspace = true }
prost = { workspace = true }

# Discovery
trust-dns-resolver = "0.23"
consul = { version = "0.4", optional = true }
etcd-client = { version = "0.12", optional = true }

# Cryptography
ed25519-dalek = "2.1"
sha2 = "0.10"
rand = "0.8"

# Utilities
dashmap = { workspace = true }
parking_lot = { workspace = true }
futures = { workspace = true }
tokio-stream = "0.1"

[dev-dependencies]
proptest = { workspace = true }
criterion = { workspace = true }
mockall = "0.12"

[features]
default = []
consul-discovery = ["consul"]
etcd-discovery = ["etcd-client"]
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (2 weeks)

- [ ] Node abstraction
- [ ] Basic cluster state management
- [ ] Static discovery implementation
- [ ] QUIC transport implementation
- [ ] Unit tests

### Phase 2: Election & Membership (2 weeks)

- [ ] Raft election implementation
- [ ] Join/leave protocol
- [ ] Health monitoring
- [ ] Coordinator heartbeats
- [ ] Integration tests

### Phase 3: Model Loading (2 weeks)

- [ ] Model manifest handling
- [ ] Filesystem model source
- [ ] S3 model source
- [ ] Partition assignment algorithm
- [ ] Cross-validation protocol

### Phase 4: Advanced Features (2 weeks)

- [ ] DNS discovery
- [ ] Consul discovery
- [ ] Failure recovery
- [ ] Adaptive timeouts
- [ ] Performance optimization

### Phase 5: Production Readiness (1 week)

- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Examples
- [ ] Benchmarks
- [ ] Security audit

---

**Status**: Design Specification
**Version**: 1.0
**Last Updated**: 2025-10-11
**Authors**: Butterfly Distributed Systems Team
