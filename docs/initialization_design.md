# Butterfly System Initialization and Bootstrap Design

## Executive Summary

This document specifies the complete bootstrap sequence for the Butterfly distributed inference system, from cold start to ready-for-inference state. The design ensures deterministic initialization, Byzantine fault tolerance during startup, and graceful handling of partial failures.

**Key Properties:**
- **Atomic Cluster Formation**: Either the entire cluster initializes successfully or it fails safely
- **Leader Election Guarantees**: Raft-based coordinator election with bounded time
- **Model Consistency**: Cryptographic verification that all nodes have identical model weights
- **Partition Validation**: Proof that the cluster can support the requested model
- **Recovery-Ready**: System can restart from any valid checkpoint
- **Bounded Initialization Time**: Upper bound on startup latency (configurable)

## Table of Contents

1. [Initialization State Machine](#1-initialization-state-machine)
2. [Node Startup Sequence](#2-node-startup-sequence)
3. [Cluster Formation Protocol](#3-cluster-formation-protocol)
4. [Model Loading and Distribution](#4-model-loading-and-distribution)
5. [Partition Assignment](#5-partition-assignment)
6. [Ready State Conditions](#6-ready-state-conditions)
7. [Failure Scenarios and Recovery](#7-failure-scenarios-and-recovery)
8. [Timing Constraints](#8-timing-constraints)
9. [Security Considerations](#9-security-considerations)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Initialization State Machine

### 1.1 Node-Level State Machine

Each node progresses through a deterministic sequence of initialization states:

```
┌─────────────┐
│   COLD      │  Initial state when process starts
└──────┬──────┘
       │
       │ Load config, initialize logging
       ▼
┌─────────────┐
│  BOOTSTRAP  │  Loading configuration and capabilities
└──────┬──────┘
       │
       │ Join cluster, discover peers
       ▼
┌─────────────┐
│  JOINING    │  Registering with coordinator
└──────┬──────┘
       │
       │ Load model partition (if worker)
       │ Start election (if coordinator candidate)
       ▼
┌─────────────┐
│  LOADING    │  Loading model weights and compiling kernels
└──────┬──────┘
       │
       │ Validate weights, run warmup
       ▼
┌─────────────┐
│ VALIDATING  │  Self-test and capability verification
└──────┬──────┘
       │
       │ Register as operational
       ▼
┌─────────────┐
│   READY     │  Ready to accept inference work
└──────┬──────┘
       │
       │ Begin normal operation
       ▼
┌─────────────┐
│ OPERATIONAL │  Normal inference processing
└─────────────┘

Error Transitions:
  * → FAILED      On catastrophic error
  FAILED → COLD   Manual restart
```

### 1.2 Cluster-Level State Machine

The cluster as a whole progresses through these states:

```
┌──────────────┐
│  UNFORMED    │  No nodes connected
└──────┬───────┘
       │
       │ First node joins
       ▼
┌──────────────┐
│  FORMING     │  Collecting nodes, waiting for quorum
└──────┬───────┘
       │
       │ Quorum reached, elect coordinator
       ▼
┌──────────────┐
│  ELECTING    │  Raft leader election in progress
└──────┬───────┘
       │
       │ Coordinator elected
       ▼
┌──────────────┐
│ DISTRIBUTING │  Coordinator assigning model partitions
└──────┬───────┘
       │
       │ All nodes loaded their partitions
       ▼
┌──────────────┐
│  VERIFYING   │  Cross-validation of model weights
└──────┬───────┘
       │
       │ Verification passed
       ▼
┌──────────────┐
│   READY      │  Cluster ready for inference requests
└──────┬───────┘
       │
       │ Accept first request
       ▼
┌──────────────┐
│   ACTIVE     │  Serving inference traffic
└──────────────┘
```

### 1.3 State Transition Timeouts

Each state transition has a maximum duration:

| Transition | Timeout | Action on Timeout |
|------------|---------|-------------------|
| COLD → BOOTSTRAP | 10s | Restart node |
| BOOTSTRAP → JOINING | 30s | Retry join |
| JOINING → LOADING | 60s | Check cluster health |
| LOADING → VALIDATING | 300s | Fail node (model issue) |
| VALIDATING → READY | 30s | Fail node (hardware issue) |
| FORMING → ELECTING | 120s | Abort startup |
| ELECTING → DISTRIBUTING | 60s | Restart election |
| DISTRIBUTING → VERIFYING | 300s | Check slow nodes |
| VERIFYING → READY | 60s | Identify mismatch source |

**Total worst-case initialization time**: ~12 minutes (cold start with large model)

---

## 2. Node Startup Sequence

### 2.1 Cold Start Procedure

When a node process starts, it executes this sequence:

```rust
async fn cold_start() -> Result<NodeRuntime, StartupError> {
    // Phase 1: COLD → BOOTSTRAP
    let config = load_configuration()?;                 // 1s
    initialize_logging(&config)?;                       // <1s
    let capabilities = detect_hardware_capabilities()?; // 2s

    info!("Node starting with capabilities: {:?}", capabilities);

    // Phase 2: BOOTSTRAP → JOINING
    let transport = initialize_network(&config).await?; // 5s
    let discovery = peer_discovery(&config).await?;     // 10s

    // Phase 3: JOINING → LOADING
    let cluster = join_cluster(
        &config,
        &transport,
        &discovery,
        capabilities,
    ).await?;                                            // 30s

    // Phase 4: LOADING → VALIDATING
    let model = load_model_partition(
        &cluster.assignment,
        &config.model_source,
    ).await?;                                            // 60-300s (depends on size)

    // Phase 5: VALIDATING → READY
    validate_model_weights(&model, &cluster.manifest).await?; // 10s
    run_warmup_inference(&model).await?;                      // 20s

    // Phase 6: Register as ready
    cluster.coordinator.register_ready(node_id).await?;       // 1s

    Ok(NodeRuntime {
        config,
        cluster,
        model,
        transport,
    })
}
```

### 2.2 Configuration Loading

**Configuration Sources** (in priority order):
1. Command-line arguments
2. Environment variables
3. Configuration file (TOML/YAML)
4. Compiled defaults

**Required Configuration Parameters:**

```toml
[node]
id = "node-0"  # Unique node identifier (UUID or custom)
role = "worker"  # "coordinator" or "worker"

[cluster]
discovery_method = "static"  # "static" | "dns" | "consul" | "etcd"
seed_nodes = ["10.0.1.10:7000", "10.0.1.11:7000"]
cluster_name = "butterfly-prod"
quorum_size = 5  # 2f+1 for f Byzantine failures

[model]
source_type = "filesystem"  # "filesystem" | "s3" | "http"
source_path = "/models/llama-70b"
manifest_hash = "sha256:abc123..."  # Expected manifest checksum
partition_strategy = "layer_range"  # "layer_range" | "tensor_parallel"

[network]
bind_address = "0.0.0.0:7000"
external_address = "10.0.1.10:7000"
max_message_size = "1GB"
compression = "zstd"

[resources]
max_memory_gb = 80
max_gpu_memory_gb = 40
num_cpu_cores = 64

[timeouts]
join_timeout_secs = 60
model_load_timeout_secs = 300
heartbeat_interval_ms = 100
```

### 2.3 Hardware Capability Detection

The node automatically detects available resources:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: NodeId,
    pub hardware: HardwareSpec,
    pub performance: PerformanceProfile,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub cpu_cores: usize,
    pub cpu_model: String,
    pub ram_bytes: u64,
    pub gpu_devices: Vec<GpuSpec>,
    pub network_bandwidth_gbps: f64,
    pub storage_type: StorageType,  // SSD, NVMe, HDD
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    pub device_id: usize,
    pub name: String,
    pub memory_bytes: u64,
    pub compute_capability: String,  // e.g., "8.9" for H100
    pub flops_fp16: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub estimated_tflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub latency_percentiles: LatencyStats,
}

// Capability detection implementation
async fn detect_hardware_capabilities() -> Result<NodeCapabilities> {
    let cpu_info = sysinfo::System::new_all();
    let gpu_devices = detect_gpus().await?;  // Use nvml/rocm

    // Run micro-benchmarks
    let perf = benchmark_hardware(&gpu_devices).await?;

    Ok(NodeCapabilities {
        node_id: NodeId::new(),
        hardware: HardwareSpec {
            cpu_cores: cpu_info.cpus().len(),
            cpu_model: cpu_info.global_cpu_info().name().to_string(),
            ram_bytes: cpu_info.total_memory(),
            gpu_devices,
            network_bandwidth_gbps: estimate_network_bandwidth().await?,
            storage_type: detect_storage_type()?,
        },
        performance: perf,
        features: detect_features(),
    })
}
```

### 2.4 Network Initialization

The node establishes network connectivity:

```rust
async fn initialize_network(config: &Config) -> Result<Transport> {
    // Create QUIC transport for low-latency messaging
    let quic_config = quinn::ServerConfig::with_crypto(tls_config());
    let endpoint = Endpoint::server(quic_config, config.network.bind_address)?;

    // Create gRPC server for control plane
    let grpc_server = tonic::transport::Server::builder()
        .add_service(CoordinationServiceServer::new(handler))
        .serve(config.network.bind_address);

    // Create shared memory transport for same-machine nodes
    let shm = SharedMemoryTransport::new()?;

    Ok(Transport {
        quic_endpoint: endpoint,
        grpc_server,
        shared_memory: shm,
        compression: CompressionCodec::Zstd,
    })
}
```

---

## 3. Cluster Formation Protocol

### 3.1 Peer Discovery

Nodes discover each other through one of several methods:

#### Static Discovery

```rust
async fn static_discovery(seed_nodes: &[SocketAddr]) -> Result<Vec<PeerInfo>> {
    let mut peers = Vec::new();

    for addr in seed_nodes {
        match connect_to_peer(addr).await {
            Ok(peer) => {
                peers.push(peer);
                info!("Discovered peer: {}", peer.node_id);
            }
            Err(e) => {
                warn!("Failed to connect to seed {}: {}", addr, e);
            }
        }
    }

    if peers.is_empty() {
        return Err(DiscoveryError::NoReachablePeers);
    }

    Ok(peers)
}
```

#### DNS-based Discovery

```rust
async fn dns_discovery(service_name: &str) -> Result<Vec<PeerInfo>> {
    // Query SRV records for service
    let query = format!("_butterfly._tcp.{}", service_name);
    let resolver = TokioAsyncResolver::tokio(
        ResolverConfig::default(),
        ResolverOpts::default(),
    )?;

    let srv_records = resolver.srv_lookup(query).await?;

    let mut peers = Vec::new();
    for srv in srv_records {
        let addr = SocketAddr::new(srv.target().to_string(), srv.port());
        if let Ok(peer) = connect_to_peer(&addr).await {
            peers.push(peer);
        }
    }

    Ok(peers)
}
```

#### Consul/etcd Discovery

```rust
async fn consul_discovery(consul_addr: &str) -> Result<Vec<PeerInfo>> {
    let client = consul::Client::new(consul_addr)?;

    // Register this node
    client.agent().service_register(&ServiceRegistration {
        name: "butterfly-node".to_string(),
        id: node_id.to_string(),
        address: external_address,
        port: config.network.port,
        tags: vec!["butterfly".to_string()],
        meta: capabilities.to_meta(),
    }).await?;

    // Discover other nodes
    let services = client.catalog().service("butterfly-node", None).await?;

    let peers = services.iter()
        .filter(|s| s.id != node_id.to_string())
        .map(|s| PeerInfo {
            node_id: NodeId::from_str(&s.id),
            address: format!("{}:{}", s.address, s.port).parse()?,
            capabilities: parse_meta(&s.meta)?,
        })
        .collect();

    Ok(peers)
}
```

### 3.2 Cluster Join Protocol

Once peers are discovered, the node joins the cluster:

```
Node                           Coordinator (if exists)
  │                                    │
  │────── JOIN_REQUEST ───────────────>│
  │  {node_id, capabilities,           │
  │   protocol_version}                │
  │                                    │
  │                             Validate compatibility
  │                             Check cluster capacity
  │                                    │
  │<────── JOIN_RESPONSE ──────────────│
  │  {cluster_config, peer_list,       │
  │   coordinator_id, epoch}           │
  │                                    │
  │────── CAPABILITY_ACK ──────────────>│
  │                                    │
  │<────── ASSIGNMENT_PENDING ─────────│
  │                                    │
```

**Message Specifications:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinRequest {
    pub node_id: NodeId,
    pub capabilities: NodeCapabilities,
    pub protocol_version: Version,
    pub timestamp: i64,
    pub signature: Vec<u8>,  // Sign with node's private key
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResponse {
    pub status: JoinStatus,
    pub cluster_config: ClusterConfig,
    pub peer_list: Vec<PeerInfo>,
    pub coordinator_id: NodeId,
    pub epoch: Epoch,
    pub model_manifest: ModelManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinStatus {
    Accepted,
    Rejected { reason: String },
    Pending { wait_for: Duration },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub cluster_name: String,
    pub quorum_size: usize,
    pub max_byzantine: usize,
    pub current_member_count: usize,
    pub protocol_version: Version,
}
```

### 3.3 Coordinator Election (Raft-Based)

If no coordinator exists, nodes participate in leader election:

```
Initial State: All nodes in CANDIDATE state

Time=0: Node A, B, C start election
  A: Start election timer (random: 150ms)
  B: Start election timer (random: 200ms)
  C: Start election timer (random: 180ms)

Time=150ms: Node A timeout fires first
  A: Increment term to 1
  A: Vote for self
  A: Send REQUEST_VOTE to B, C

Time=155ms: B receives REQUEST_VOTE from A
  B: Check term (1 > 0), vote for A
  B: Send VOTE_RESPONSE(granted=true)

Time=156ms: C receives REQUEST_VOTE from A
  C: Check term (1 > 0), vote for A
  C: Send VOTE_RESPONSE(granted=true)

Time=160ms: A receives majority votes (3/3)
  A: Transition to LEADER
  A: Send HEARTBEAT to all nodes

Time=161ms: B, C receive HEARTBEAT
  B, C: Recognize A as leader
  B, C: Transition to FOLLOWER
```

**Raft Election Message Types:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    RequestVote {
        term: u64,
        candidate_id: NodeId,
        last_log_index: u64,
        last_log_term: u64,
    },
    VoteResponse {
        term: u64,
        vote_granted: bool,
    },
    AppendEntries {
        term: u64,
        leader_id: NodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    AppendEntriesResponse {
        term: u64,
        success: bool,
        match_index: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

pub struct RaftStateMachine {
    role: RaftRole,
    current_term: u64,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>,
    commit_index: u64,
    last_applied: u64,

    // Leader state
    next_index: HashMap<NodeId, u64>,
    match_index: HashMap<NodeId, u64>,

    // Timing
    election_timeout: Duration,
    heartbeat_interval: Duration,
    last_heartbeat: Instant,
}
```

**Election Guarantees:**

1. **Safety**: At most one leader per term
2. **Liveness**: A leader is eventually elected if majority available
3. **Bounded Time**: Election completes in O(RTT × log N) expected time

**Proof Sketch:**

- Each term has at most one leader (nodes vote once per term)
- If majority nodes are connected, one candidate will receive majority votes
- Random timeouts prevent split votes (with high probability)
- Expected elections until success: constant (typically 1-2)

### 3.4 Cluster Membership Changes

Adding or removing nodes while cluster is operational:

```rust
// Raft two-phase reconfiguration
async fn add_node(new_node: NodeId) -> Result<()> {
    // Phase 1: Add to joint consensus (C_old,new)
    let joint_config = ClusterConfig {
        old_members: current_members.clone(),
        new_members: current_members.union(new_node),
    };

    coordinator.replicate_log_entry(LogEntry::ConfigChange(joint_config)).await?;
    wait_for_commit().await?;

    // Phase 2: Transition to new configuration
    let new_config = ClusterConfig {
        old_members: vec![],
        new_members: current_members.union(new_node),
    };

    coordinator.replicate_log_entry(LogEntry::ConfigChange(new_config)).await?;
    wait_for_commit().await?;

    Ok(())
}
```

---

## 4. Model Loading and Distribution

### 4.1 Model Manifest

Every model has a manifest describing its structure:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub architecture: String,  // "llama", "gpt", "bert", etc.
    pub version: String,
    pub total_layers: usize,
    pub total_parameters: u64,
    pub weight_files: Vec<WeightFile>,
    pub partition_strategy: PartitionStrategy,
    pub checksum: ManifestChecksum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFile {
    pub path: String,
    pub size_bytes: u64,
    pub checksum: FileChecksum,
    pub layer_range: LayerRange,  // Which layers this file contains
    pub format: WeightFormat,  // "safetensors", "pytorch", "gguf"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: usize,
    pub end: usize,  // exclusive
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    GGUF,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChecksum {
    pub algorithm: ChecksumAlgorithm,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    SHA256,
    BLAKE3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestChecksum {
    pub merkle_root: [u8; 32],
    pub signature: Vec<u8>,  // Optional: signed by model publisher
}
```

### 4.2 Partition Assignment Algorithm

The coordinator determines how to divide the model:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    LayerRange,      // Assign contiguous layer ranges
    TensorParallel,  // Split tensors across nodes
    PipelineParallel, // Combination of both
}

async fn assign_partitions(
    manifest: &ModelManifest,
    nodes: &[NodeInfo],
    strategy: PartitionStrategy,
) -> Result<Vec<PartitionAssignment>> {
    match strategy {
        PartitionStrategy::LayerRange => {
            assign_layer_range_partitions(manifest, nodes).await
        }
        PartitionStrategy::TensorParallel => {
            assign_tensor_parallel_partitions(manifest, nodes).await
        }
        PartitionStrategy::PipelineParallel => {
            assign_pipeline_parallel_partitions(manifest, nodes).await
        }
    }
}

// Layer range partitioning (simplest, most common)
async fn assign_layer_range_partitions(
    manifest: &ModelManifest,
    nodes: &[NodeInfo],
) -> Result<Vec<PartitionAssignment>> {
    let total_layers = manifest.total_layers;
    let num_nodes = nodes.len();

    // Weight nodes by their compute capacity
    let total_capacity: f64 = nodes.iter()
        .map(|n| n.capabilities.performance.estimated_tflops)
        .sum();

    let mut assignments = Vec::new();
    let mut layer_offset = 0;

    for node in nodes {
        // Assign layers proportional to node capacity
        let fraction = node.capabilities.performance.estimated_tflops / total_capacity;
        let layers_for_node = (total_layers as f64 * fraction).ceil() as usize;
        let layers_for_node = layers_for_node.min(total_layers - layer_offset);

        let layer_range = LayerRange {
            start: layer_offset,
            end: layer_offset + layers_for_node,
        };

        // Determine which weight files this node needs
        let required_files: Vec<WeightFile> = manifest.weight_files
            .iter()
            .filter(|f| ranges_overlap(&f.layer_range, &layer_range))
            .cloned()
            .collect();

        assignments.push(PartitionAssignment {
            node_id: node.node_id,
            layer_range,
            required_files,
            estimated_memory_bytes: estimate_partition_memory(&required_files),
            estimated_load_time_secs: estimate_load_time(&required_files, node),
        });

        layer_offset += layers_for_node;
    }

    // Validate: all layers assigned exactly once
    assert_eq!(layer_offset, total_layers);

    Ok(assignments)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    pub node_id: NodeId,
    pub layer_range: LayerRange,
    pub required_files: Vec<WeightFile>,
    pub estimated_memory_bytes: u64,
    pub estimated_load_time_secs: f64,
}
```

### 4.3 Weight Loading Protocol

Nodes load their assigned model weights:

```
Coordinator                              Worker Node
     │                                        │
     │──── LOAD_PARTITION_CMD ───────────────>│
     │  {assignment, manifest, source}        │
     │                                        │
     │                                   Fetch weights
     │                                   Verify checksums
     │                                   Allocate GPU memory
     │                                   Load into memory
     │                                        │
     │<──── LOAD_PROGRESS ────────────────────│
     │  {percent: 25%, eta: 45s}              │
     │                                        │
     │<──── LOAD_PROGRESS ────────────────────│
     │  {percent: 50%, eta: 30s}              │
     │                                        │
     │<──── LOAD_COMPLETE ────────────────────│
     │  {checksum, memory_used, load_time}    │
     │                                        │
```

**Implementation:**

```rust
async fn load_partition(
    assignment: &PartitionAssignment,
    source: &ModelSource,
) -> Result<LoadedPartition> {
    let start_time = Instant::now();
    let mut total_bytes_loaded = 0u64;
    let total_bytes = assignment.required_files.iter()
        .map(|f| f.size_bytes)
        .sum::<u64>();

    let mut tensors = Vec::new();

    for file in &assignment.required_files {
        info!("Loading weight file: {}", file.path);

        // Fetch file (may be local filesystem, S3, HTTP, etc.)
        let data = fetch_weight_file(source, &file.path).await?;

        // Verify checksum before using
        let computed_checksum = compute_checksum(&data, &file.checksum.algorithm);
        if computed_checksum != file.checksum.value {
            return Err(LoadError::ChecksumMismatch {
                file: file.path.clone(),
                expected: file.checksum.value.clone(),
                actual: computed_checksum,
            });
        }

        // Deserialize tensors
        let file_tensors = deserialize_weights(&data, &file.format)?;
        tensors.extend(file_tensors);

        total_bytes_loaded += file.size_bytes;

        // Report progress
        let progress = (total_bytes_loaded as f64 / total_bytes as f64) * 100.0;
        report_progress(progress).await?;
    }

    // Allocate GPU memory and copy tensors
    let gpu_tensors = allocate_and_upload_to_gpu(tensors).await?;

    // Compile CUDA/ROCm kernels
    let compiled_kernels = compile_kernels_for_layers(
        assignment.layer_range.start,
        assignment.layer_range.end,
    ).await?;

    let load_time = start_time.elapsed();

    Ok(LoadedPartition {
        assignment: assignment.clone(),
        tensors: gpu_tensors,
        kernels: compiled_kernels,
        memory_used_bytes: total_bytes_loaded,
        load_time,
        checksum: compute_partition_checksum(&gpu_tensors),
    })
}

#[derive(Debug)]
pub struct LoadedPartition {
    pub assignment: PartitionAssignment,
    pub tensors: Vec<GpuTensor>,
    pub kernels: CompiledKernels,
    pub memory_used_bytes: u64,
    pub load_time: Duration,
    pub checksum: PartitionChecksum,
}
```

### 4.4 Weight Distribution Strategies

Depending on deployment, weights can be distributed in several ways:

#### Shared Filesystem (NFS, Lustre)

```rust
async fn fetch_from_shared_fs(path: &str) -> Result<Vec<u8>> {
    // Direct file read - all nodes see same filesystem
    tokio::fs::read(path).await.map_err(Into::into)
}
```

#### Object Storage (S3, GCS, Azure Blob)

```rust
async fn fetch_from_s3(bucket: &str, key: &str) -> Result<Vec<u8>> {
    let client = aws_sdk_s3::Client::new(&aws_config::load_from_env().await);

    let resp = client.get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let bytes = resp.body.collect().await?.into_bytes();
    Ok(bytes.to_vec())
}
```

#### Coordinator Distribution (for small models)

```rust
async fn fetch_from_coordinator(
    coordinator_addr: &SocketAddr,
    file_id: &str,
) -> Result<Vec<u8>> {
    let mut stream = coordinator_client.stream_weight_file(file_id).await?;

    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk?);
    }

    Ok(buffer)
}
```

#### Peer-to-Peer Distribution (BitTorrent-style)

```rust
async fn fetch_from_peers(
    file_id: &str,
    peers: &[NodeId],
) -> Result<Vec<u8>> {
    // Download different chunks from different peers in parallel
    let chunks = request_chunks_from_peers(file_id, peers).await?;

    // Reassemble and verify
    let data = assemble_chunks(chunks)?;
    Ok(data)
}
```

### 4.5 Cross-Validation of Model Weights

After all nodes load their partitions, they must verify consistency:

```
Coordinator                              Worker Nodes
     │                                    │     │     │
     │──── REQUEST_CHECKSUMS ────────────>│     │     │
     │                                    │     │     │
     │<──── CHECKSUM_REPORT ──────────────│     │     │
     │  {node_id: A, checksum: 0x123...}  │     │     │
     │                                    │     │     │
     │<──── CHECKSUM_REPORT ──────────────┼─────│     │
     │  {node_id: B, checksum: 0x456...}  │     │     │
     │                                    │     │     │
     │<──── CHECKSUM_REPORT ──────────────┼─────┼─────│
     │  {node_id: C, checksum: 0x789...}  │     │     │
     │                                    │     │     │
     │  Compute Merkle tree of all checksums    │     │
     │  Broadcast Merkle root                   │     │
     │                                    │     │     │
     │──── VALIDATION_RESULT ────────────>│     │     │
     │  {merkle_root: 0xabc..., status: OK}     │     │
     │                                    │     │     │
```

**Merkle Tree Construction:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionChecksum {
    pub node_id: NodeId,
    pub layer_range: LayerRange,
    pub weight_hash: [u8; 32],
    pub timestamp: i64,
}

fn build_checksum_merkle_tree(
    checksums: &[PartitionChecksum],
) -> MerkleTree {
    let mut leaves: Vec<[u8; 32]> = checksums
        .iter()
        .map(|cs| {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&cs.node_id.0.to_le_bytes());
            hasher.update(&cs.layer_range.start.to_le_bytes());
            hasher.update(&cs.layer_range.end.to_le_bytes());
            hasher.update(&cs.weight_hash);
            hasher.finalize().into()
        })
        .collect();

    // Build tree bottom-up
    while leaves.len() > 1 {
        let mut next_level = Vec::new();

        for pair in leaves.chunks(2) {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            if pair.len() > 1 {
                hasher.update(&pair[1]);
            }
            next_level.push(hasher.finalize().into());
        }

        leaves = next_level;
    }

    MerkleTree {
        root: leaves[0],
        leaf_count: checksums.len(),
    }
}

async fn validate_cluster_model_consistency(
    coordinator: &Coordinator,
    nodes: &[NodeId],
) -> Result<()> {
    // Collect checksums from all nodes
    let mut checksums = Vec::new();
    for node_id in nodes {
        let checksum = coordinator.request_checksum(node_id).await?;
        checksums.push(checksum);
    }

    // Build Merkle tree
    let tree = build_checksum_merkle_tree(&checksums);

    // Broadcast root to all nodes for verification
    coordinator.broadcast_merkle_root(tree.root).await?;

    // Each node verifies they contributed to this root
    for node_id in nodes {
        let verified = coordinator.verify_node_checksum(node_id, tree.root).await?;
        if !verified {
            return Err(ValidationError::ChecksumMismatch {
                node_id: *node_id,
                merkle_root: tree.root,
            });
        }
    }

    info!("Model consistency validated: merkle_root={:?}", tree.root);
    Ok(())
}
```

---

## 5. Partition Assignment

### 5.1 Partition Feasibility Check

Before assigning partitions, coordinator validates the cluster can handle the model:

```rust
async fn check_partition_feasibility(
    manifest: &ModelManifest,
    nodes: &[NodeInfo],
) -> Result<FeasibilityReport> {
    // 1. Check total memory
    let total_memory_required = manifest.total_parameters * size_of_parameter();
    let total_memory_available: u64 = nodes.iter()
        .map(|n| n.capabilities.hardware.ram_bytes)
        .sum();

    if total_memory_required > total_memory_available {
        return Ok(FeasibilityReport {
            feasible: false,
            reason: format!(
                "Insufficient memory: need {} GB, have {} GB",
                total_memory_required / GB,
                total_memory_available / GB,
            ),
        });
    }

    // 2. Check network bandwidth for communication
    let estimated_activation_size = estimate_activation_memory(manifest);
    let required_bandwidth = estimated_activation_size * inference_rate;
    let available_bandwidth: f64 = nodes.iter()
        .map(|n| n.capabilities.hardware.network_bandwidth_gbps)
        .min()
        .unwrap_or(0.0);

    if required_bandwidth > available_bandwidth * 0.8 {  // 80% utilization threshold
        return Ok(FeasibilityReport {
            feasible: false,
            reason: format!(
                "Insufficient network bandwidth: need {} Gbps, have {} Gbps",
                required_bandwidth,
                available_bandwidth,
            ),
        });
    }

    // 3. Check computational capacity
    let required_tflops = estimate_required_tflops(manifest, target_latency);
    let available_tflops: f64 = nodes.iter()
        .map(|n| n.capabilities.performance.estimated_tflops)
        .sum();

    if required_tflops > available_tflops {
        return Ok(FeasibilityReport {
            feasible: false,
            reason: format!(
                "Insufficient compute: need {} TFLOPS, have {} TFLOPS",
                required_tflops,
                available_tflops,
            ),
        });
    }

    Ok(FeasibilityReport {
        feasible: true,
        reason: "Cluster can support this model".to_string(),
    })
}
```

### 5.2 Optimal Partition Algorithm

The coordinator computes an optimal partition using dynamic programming:

```rust
// Objective: Minimize total inference latency
// Subject to: Memory constraints, network topology
async fn compute_optimal_partition(
    manifest: &ModelManifest,
    nodes: &[NodeInfo],
) -> Result<Vec<PartitionAssignment>> {
    // Model as optimization problem:
    //   min Σ_i (compute_time_i + comm_time_i)
    //   s.t. memory_i ≤ capacity_i for all i
    //        Σ_i layers_i = total_layers

    let n = nodes.len();
    let L = manifest.total_layers;

    // dp[layer][node] = (min_time, assignment)
    let mut dp = vec![vec![None; n]; L + 1];
    dp[0][0] = Some((0.0, vec![]));

    // Dynamic programming over layer ranges
    for end_layer in 1..=L {
        for node_idx in 0..n {
            for start_layer in 0..end_layer {
                let prev_node = if node_idx > 0 { node_idx - 1 } else { 0 };

                if let Some((prev_time, mut prev_assignment)) = dp[start_layer][prev_node].clone() {
                    let layer_range = LayerRange {
                        start: start_layer,
                        end: end_layer,
                    };

                    // Check if this assignment fits in node memory
                    let memory_needed = estimate_memory_for_range(&layer_range, manifest);
                    if memory_needed > nodes[node_idx].capabilities.hardware.ram_bytes {
                        continue;
                    }

                    // Compute time for this node to process this range
                    let compute_time = estimate_compute_time(
                        &layer_range,
                        &nodes[node_idx].capabilities.performance,
                    );

                    // Communication time to next node
                    let comm_time = if node_idx < n - 1 {
                        estimate_comm_time(
                            &nodes[node_idx],
                            &nodes[node_idx + 1],
                            manifest,
                        )
                    } else {
                        0.0
                    };

                    let total_time = prev_time + compute_time + comm_time;

                    // Update DP if this is better
                    if dp[end_layer][node_idx].is_none()
                        || dp[end_layer][node_idx].as_ref().unwrap().0 > total_time
                    {
                        prev_assignment.push(PartitionAssignment {
                            node_id: nodes[node_idx].node_id,
                            layer_range,
                            required_files: get_required_files(&layer_range, manifest),
                            estimated_memory_bytes: memory_needed,
                            estimated_load_time_secs: 0.0,  // Computed later
                        });

                        dp[end_layer][node_idx] = Some((total_time, prev_assignment));
                    }
                }
            }
        }
    }

    // Extract optimal assignment
    let (optimal_time, optimal_assignment) = dp[L]
        .iter()
        .flatten()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .ok_or(PartitionError::NoFeasiblePartition)?
        .clone();

    info!(
        "Optimal partition computed: latency={:.2}ms, num_partitions={}",
        optimal_time * 1000.0,
        optimal_assignment.len(),
    );

    Ok(optimal_assignment)
}
```

### 5.3 Assignment Distribution

Once computed, assignments are distributed to nodes:

```
Coordinator                              Worker Nodes
     │                                    │     │     │
     │──── BEGIN_ASSIGNMENT_PHASE ───────>│     │     │
     │                                    │     │     │
     │──── PARTITION_ASSIGNMENT ─────────>│     │     │
     │  {node: A, layers: 0-23}           │     │     │
     │                                    │     │     │
     │──── PARTITION_ASSIGNMENT ─────────>┼─────│     │
     │  {node: B, layers: 24-47}          │     │     │
     │                                    │     │     │
     │──── PARTITION_ASSIGNMENT ─────────>┼─────┼─────│
     │  {node: C, layers: 48-70}          │     │     │
     │                                    │     │     │
     │<──── ASSIGNMENT_ACK ───────────────│     │     │
     │<──── ASSIGNMENT_ACK ───────────────┼─────│     │
     │<──── ASSIGNMENT_ACK ───────────────┼─────┼─────│
     │                                    │     │     │
```

---

## 6. Ready State Conditions

### 6.1 Node-Level Readiness

A node is considered READY when all of the following are true:

1. **Configuration Loaded**: Valid configuration parsed
2. **Network Operational**: Can send/receive messages from all peers
3. **Model Loaded**: All assigned weights loaded and validated
4. **Kernels Compiled**: GPU kernels compiled and warmed up
5. **Self-Test Passed**: Successfully completed warmup inference
6. **Health Monitor Active**: Sending and receiving heartbeats
7. **Checkpoint Loaded**: If restarting, loaded previous checkpoint

```rust
#[derive(Debug, Clone)]
pub struct NodeReadiness {
    pub configuration: ReadinessCheck,
    pub network: ReadinessCheck,
    pub model: ReadinessCheck,
    pub kernels: ReadinessCheck,
    pub self_test: ReadinessCheck,
    pub health_monitor: ReadinessCheck,
    pub checkpoint: ReadinessCheck,
}

#[derive(Debug, Clone)]
pub struct ReadinessCheck {
    pub status: CheckStatus,
    pub message: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    Pending,
    InProgress,
    Passed,
    Failed,
    Skipped,
}

impl NodeReadiness {
    pub fn is_ready(&self) -> bool {
        self.configuration.status == CheckStatus::Passed
            && self.network.status == CheckStatus::Passed
            && self.model.status == CheckStatus::Passed
            && self.kernels.status == CheckStatus::Passed
            && self.self_test.status == CheckStatus::Passed
            && self.health_monitor.status == CheckStatus::Passed
            && (self.checkpoint.status == CheckStatus::Passed
                || self.checkpoint.status == CheckStatus::Skipped)
    }

    pub fn blocking_checks(&self) -> Vec<&ReadinessCheck> {
        vec![
            &self.configuration,
            &self.network,
            &self.model,
            &self.kernels,
            &self.self_test,
            &self.health_monitor,
        ]
        .into_iter()
        .filter(|c| c.status != CheckStatus::Passed && c.status != CheckStatus::Skipped)
        .collect()
    }
}
```

### 6.2 Cluster-Level Readiness

The cluster is considered READY when:

1. **Quorum Achieved**: At least 2f+1 nodes operational
2. **Coordinator Elected**: Stable leader elected via Raft
3. **All Partitions Assigned**: Every layer has an owner
4. **Weights Validated**: Merkle tree verification passed
5. **Network Topology Verified**: All inter-node connections tested
6. **Health Checks Passing**: All nodes responding to heartbeats
7. **No Outstanding Failures**: No nodes in FAILED or RECOVERING state

```rust
#[derive(Debug, Clone)]
pub struct ClusterReadiness {
    pub quorum: ReadinessCheck,
    pub coordinator: ReadinessCheck,
    pub partitions: ReadinessCheck,
    pub weights: ReadinessCheck,
    pub network_topology: ReadinessCheck,
    pub health_checks: ReadinessCheck,
    pub failure_state: ReadinessCheck,
}

impl ClusterReadiness {
    pub fn is_ready(&self) -> bool {
        vec![
            &self.quorum,
            &self.coordinator,
            &self.partitions,
            &self.weights,
            &self.network_topology,
            &self.health_checks,
            &self.failure_state,
        ]
        .iter()
        .all(|check| check.status == CheckStatus::Passed)
    }
}
```

### 6.3 Readiness Probes

Health check endpoints for monitoring systems:

```rust
// HTTP endpoint: GET /readiness
async fn readiness_probe(
    cluster: &ClusterState,
) -> Result<ReadinessResponse, StatusCode> {
    let node_readiness = cluster.get_node_readiness().await;
    let cluster_readiness = cluster.get_cluster_readiness().await;

    if node_readiness.is_ready() && cluster_readiness.is_ready() {
        Ok(ReadinessResponse {
            status: "ready",
            checks: serde_json::to_value(&node_readiness)?,
        })
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// HTTP endpoint: GET /health
async fn liveness_probe(
    cluster: &ClusterState,
) -> Result<HealthResponse, StatusCode> {
    // Liveness only checks if process is alive
    if cluster.is_process_alive() {
        Ok(HealthResponse { status: "alive" })
    } else {
        Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}
```

---

## 7. Failure Scenarios and Recovery

### 7.1 Failure Taxonomy

| Failure Type | Detection | Recovery | Impact |
|--------------|-----------|----------|--------|
| **Node Crash During Init** | Timeout | Restart node | Delays cluster formation |
| **Coordinator Crash** | Heartbeat timeout | Re-elect leader | <1s downtime |
| **Network Partition** | Heartbeat timeouts | Wait for heal | Cluster unavailable if quorum lost |
| **Corrupted Weights** | Checksum mismatch | Re-download weights | Fails that node |
| **Insufficient Memory** | OOM error | Repartition model | Requires reconfiguration |
| **Slow Node** | Progress monitoring | Speculative execution | Minimal latency impact |

### 7.2 Node Crash During Initialization

**Scenario**: A worker node crashes while loading model weights.

**Detection**:
- Coordinator monitors load progress reports
- If no progress report received within timeout, node is suspected failed
- Coordinator queries other nodes for observations

**Recovery**:
1. Coordinator marks node as FAILED
2. If cluster still has quorum, proceed without failed node
3. Recompute partition assignment excluding failed node
4. Redistribute failed node's layers to remaining nodes
5. Affected nodes reload new partition assignments
6. Restart validation phase

**Timeline**:
```
T=0:    Node A starts loading weights
T=30s:  Node A sends progress report (25%)
T=60s:  Node A sends progress report (50%)
T=90s:  [Node A crashes]
T=150s: Coordinator timeout for progress report
T=151s: Coordinator queries peers about Node A
T=152s: Peers report Node A unresponsive
T=153s: Coordinator marks Node A as FAILED
T=154s: Coordinator recomputes partitions (B, C, D, E)
T=155s: Coordinator sends new assignments
T=200s: Nodes complete reloading
T=210s: Cluster ready (total delay: 120s)
```

### 7.3 Coordinator Crash During Election

**Scenario**: The coordinator crashes while distributing partition assignments.

**Detection**:
- Workers monitoring heartbeats from coordinator
- If 3 consecutive heartbeats missed (300ms), suspect failure
- Workers initiate new election

**Recovery**:
1. Raft election timeout fires on followers
2. Candidate with most up-to-date log becomes leader
3. New coordinator resumes partition assignment
4. Workers acknowledge new leader
5. Continue from last committed log entry

**Timeline**:
```
T=0:    Coordinator crashes after assigning to nodes A, B
        (nodes C, D not yet assigned)
T=100ms: Heartbeat #1 missed
T=200ms: Heartbeat #2 missed
T=300ms: Heartbeat #3 missed, election triggered
T=350ms: Node B wins election (random timeout fired first)
T=360ms: Node B sends heartbeats, acknowledged
T=361ms: Node B examines log, sees incomplete assignment
T=362ms: Node B resends assignments to C, D
T=400ms: All nodes acknowledged, continue initialization
Total recovery time: 400ms
```

**Proof of Correctness**:
- Raft guarantees at most one leader per term
- New leader has all committed entries (election restriction)
- Uncommitted entries may be replayed or discarded safely
- Workers acknowledge assignments idempotently

### 7.4 Network Partition During Cluster Formation

**Scenario**: Network partition splits cluster into two groups during initialization.

**Detection**:
- Nodes unable to communicate across partition
- Each partition suspects the other side failed
- Quorum checks fail on minority partition

**Behavior**:
```
Initial cluster: [A, B, C, D, E]  (quorum = 3)
Partition occurs: [A, B] | [C, D, E]

Majority partition [C, D, E]:
  - Has quorum (3 ≥ 3)
  - Continues initialization
  - Suspects A, B failed
  - Proceeds with 3-node cluster

Minority partition [A, B]:
  - No quorum (2 < 3)
  - Cannot proceed
  - Waits for network heal or manual intervention
  - Logs errors
```

**Recovery After Partition Heals**:
1. Nodes in minority detect majority coordinator
2. Minority nodes reconcile logs with majority
3. Minority nodes reset to last committed state
4. Minority nodes rejoin cluster as new members
5. Coordinator may reassign partitions to utilize recovered nodes

### 7.5 Corrupted Model Weights

**Scenario**: A node downloads corrupted weights from storage.

**Detection**:
- Checksum verification fails during load
- OR: Cross-validation Merkle tree mismatch detected

**Recovery**:
```rust
async fn handle_corrupted_weights(
    node_id: NodeId,
    file_path: &str,
) -> Result<()> {
    warn!(
        node = ?node_id,
        file = file_path,
        "Corrupted weights detected"
    );

    // 1. Retry download from source
    for attempt in 1..=3 {
        info!("Retry attempt {} for {}", attempt, file_path);

        match retry_download(file_path).await {
            Ok(data) => {
                if verify_checksum(&data) {
                    return Ok(());
                }
            }
            Err(e) => {
                warn!("Retry {} failed: {}", attempt, e);
            }
        }

        tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
    }

    // 2. If retries fail, try alternate sources
    if let Some(peer) = find_peer_with_same_partition(node_id).await? {
        info!("Fetching weights from peer: {:?}", peer);
        match fetch_from_peer(peer, file_path).await {
            Ok(data) if verify_checksum(&data) => return Ok(()),
            _ => {}
        }
    }

    // 3. If all sources fail, mark node as failed
    Err(LoadError::CorruptedWeights {
        file: file_path.to_string(),
        exhausted_sources: true,
    })
}
```

### 7.6 Insufficient Resources

**Scenario**: Partition assignment requests more memory than node has available.

**Detection**:
- Feasibility check fails before assignment
- OR: OOM error during weight loading

**Recovery**:
```rust
async fn handle_insufficient_resources(
    cluster: &ClusterState,
    manifest: &ModelManifest,
) -> Result<RecoveryAction> {
    let feasibility = check_partition_feasibility(manifest, &cluster.nodes).await?;

    if !feasibility.feasible {
        warn!("Cluster cannot support model: {}", feasibility.reason);

        // Option 1: Add more nodes
        if can_add_nodes(&cluster) {
            return Ok(RecoveryAction::RequestAdditionalNodes {
                required_memory: feasibility.memory_deficit,
            });
        }

        // Option 2: Use model quantization
        if supports_quantization(manifest) {
            return Ok(RecoveryAction::QuantizeModel {
                from_precision: "fp16",
                to_precision: "int8",
            });
        }

        // Option 3: Use offloading
        if supports_disk_offload(&cluster) {
            return Ok(RecoveryAction::EnableDiskOffload {
                offload_percentage: 30,
            });
        }

        // Option 4: Reject startup
        return Err(StartupError::InsufficientResources {
            reason: feasibility.reason,
        });
    }

    Ok(RecoveryAction::Proceed)
}
```

### 7.7 Recovery State Machine

Detailed state machine for failure recovery during initialization:

```
Normal Flow:                 Failure Recovery Flow:

COLD                         COLD
  │                            │
  ▼                            ▼
BOOTSTRAP                    BOOTSTRAP
  │                            │
  │                            │ [Failure detected]
  ▼                            ▼
JOINING ─────────────────> RECOVERY_ASSESS
  │                            │
  │                            ├─> [Retriable] ─────> RETRY
  │                            │                        │
  │                            │                        └─> back to failed state
  │                            │
  │                            └─> [Catastrophic] ───> FAILED
  │                                                      │
  ▼                                                      └─> ABORT_STARTUP
LOADING
  │
  ▼
VALIDATING
  │
  ▼
READY
```

---

## 8. Timing Constraints

### 8.1 Timeout Configuration

Comprehensive timeout parameters:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    // Node-level timeouts
    pub config_load_timeout: Duration,           // 10s
    pub hardware_detect_timeout: Duration,       // 5s
    pub network_init_timeout: Duration,          // 10s
    pub peer_discovery_timeout: Duration,        // 30s
    pub join_request_timeout: Duration,          // 10s
    pub model_load_timeout: Duration,            // 300s (5 min)
    pub kernel_compile_timeout: Duration,        // 60s
    pub warmup_timeout: Duration,                // 30s

    // Cluster-level timeouts
    pub election_timeout: Range<Duration>,       // 150-300ms
    pub heartbeat_interval: Duration,            // 100ms
    pub partition_assign_timeout: Duration,      // 60s
    pub weight_validation_timeout: Duration,     // 60s

    // Retry configuration
    pub max_join_retries: usize,                 // 3
    pub max_download_retries: usize,             // 5
    pub retry_backoff_base: Duration,            // 2s (exponential)
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            config_load_timeout: Duration::from_secs(10),
            hardware_detect_timeout: Duration::from_secs(5),
            network_init_timeout: Duration::from_secs(10),
            peer_discovery_timeout: Duration::from_secs(30),
            join_request_timeout: Duration::from_secs(10),
            model_load_timeout: Duration::from_secs(300),
            kernel_compile_timeout: Duration::from_secs(60),
            warmup_timeout: Duration::from_secs(30),

            election_timeout: Duration::from_millis(150)..Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(100),
            partition_assign_timeout: Duration::from_secs(60),
            weight_validation_timeout: Duration::from_secs(60),

            max_join_retries: 3,
            max_download_retries: 5,
            retry_backoff_base: Duration::from_secs(2),
        }
    }
}
```

### 8.2 Performance Targets

Expected initialization times for different scenarios:

| Scenario | Model Size | Nodes | Expected Time | Max Time |
|----------|------------|-------|---------------|----------|
| **Hot Start** | 7B params | 2 | 15s | 30s |
| **Cold Start (Local)** | 7B params | 2 | 45s | 90s |
| **Cold Start (S3)** | 7B params | 2 | 90s | 180s |
| **Hot Start** | 70B params | 8 | 30s | 60s |
| **Cold Start (Local)** | 70B params | 8 | 120s | 300s |
| **Cold Start (S3)** | 70B params | 8 | 240s | 600s |
| **Hot Start** | 405B params | 32 | 60s | 120s |
| **Cold Start (Local)** | 405B params | 32 | 300s | 720s |

**Hot Start**: Model weights already in filesystem cache
**Cold Start**: Model weights must be fetched from storage

### 8.3 Latency Budget Breakdown

For a 70B model cold start from S3 (240s target):

| Phase | Operation | Target | Budget |
|-------|-----------|--------|--------|
| 1 | Configuration load | 5s | 2% |
| 2 | Hardware detection | 3s | 1% |
| 3 | Network initialization | 10s | 4% |
| 4 | Peer discovery | 15s | 6% |
| 5 | Cluster formation | 20s | 8% |
| 6 | Raft election | 2s | 1% |
| 7 | Partition computation | 5s | 2% |
| 8 | Assignment distribution | 5s | 2% |
| 9 | Model download (S3) | 120s | 50% |
| 10 | Weight verification | 20s | 8% |
| 11 | Kernel compilation | 25s | 10% |
| 12 | Warmup inference | 10s | 4% |
| **Total** | | **240s** | **100%** |

### 8.4 Adaptive Timeouts

Timeouts adapt based on observed performance:

```rust
pub struct AdaptiveTimeout {
    base_timeout: Duration,
    observations: RingBuffer<Duration>,
    percentile: f64,  // e.g., 99th percentile
}

impl AdaptiveTimeout {
    pub fn new(base: Duration, percentile: f64) -> Self {
        Self {
            base_timeout: base,
            observations: RingBuffer::new(100),
            percentile,
        }
    }

    pub fn record_observation(&mut self, duration: Duration) {
        self.observations.push(duration);
    }

    pub fn current_timeout(&self) -> Duration {
        if self.observations.len() < 10 {
            // Not enough data, use base timeout
            return self.base_timeout;
        }

        let mut sorted = self.observations.to_vec();
        sorted.sort();

        let idx = (sorted.len() as f64 * self.percentile) as usize;
        let observed = sorted[idx.min(sorted.len() - 1)];

        // Timeout = observed * 3 (generous margin)
        // But clamp between [base, base * 10]
        let timeout = observed * 3;
        timeout.clamp(self.base_timeout, self.base_timeout * 10)
    }
}
```

---

## 9. Security Considerations

### 9.1 Authentication

Nodes authenticate to the cluster:

```rust
#[derive(Debug, Clone)]
pub struct NodeIdentity {
    pub node_id: NodeId,
    pub public_key: PublicKey,  // Ed25519
    pub certificate: Option<X509Certificate>,  // Optional TLS cert
}

async fn authenticate_join_request(
    request: &JoinRequest,
    trusted_keys: &KeyStore,
) -> Result<NodeIdentity> {
    // 1. Verify signature on request
    let public_key = PublicKey::from_bytes(&request.public_key_bytes)?;

    if !verify_signature(
        &request.signed_data(),
        &request.signature,
        &public_key,
    ) {
        return Err(AuthError::InvalidSignature);
    }

    // 2. Check if key is trusted (future: PKI or trust-on-first-use)
    if !trusted_keys.is_trusted(&public_key) {
        return Err(AuthError::UntrustedKey);
    }

    // 3. Verify certificate if provided
    if let Some(cert_bytes) = &request.certificate {
        let cert = X509Certificate::from_der(cert_bytes)?;
        verify_certificate(&cert, &trusted_keys.ca_cert)?;
    }

    Ok(NodeIdentity {
        node_id: request.node_id,
        public_key,
        certificate: request.certificate.as_ref().map(|c| parse_cert(c)),
    })
}
```

### 9.2 Encrypted Communication

All inter-node communication is encrypted:

```rust
// TLS configuration for gRPC
fn tls_config() -> ServerConfig {
    let cert = load_certificate("node.crt")?;
    let key = load_private_key("node.key")?;

    ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()  // Future: mutual TLS
        .with_single_cert(vec![cert], key)?
}

// QUIC configuration (TLS 1.3 built-in)
fn quic_config() -> quinn::ServerConfig {
    let cert = load_certificate("node.crt")?;
    let key = load_private_key("node.key")?;

    quinn::ServerConfig::with_crypto(Arc::new(
        rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)?
    ))
}
```

### 9.3 Byzantine Behavior Prevention

Prevent malicious nodes during initialization:

```rust
async fn validate_join_request(
    request: &JoinRequest,
    cluster: &ClusterState,
) -> Result<()> {
    // 1. Check protocol version compatibility
    if !is_compatible_version(&request.protocol_version) {
        return Err(ValidationError::IncompatibleProtocol {
            theirs: request.protocol_version,
            ours: CURRENT_PROTOCOL_VERSION,
        });
    }

    // 2. Validate capabilities make sense
    if !request.capabilities.is_plausible() {
        return Err(ValidationError::ImplausibleCapabilities);
    }

    // 3. Check for duplicate node IDs
    if cluster.has_node(request.node_id) {
        return Err(ValidationError::DuplicateNodeId(request.node_id));
    }

    // 4. Rate limit join requests (prevent DoS)
    if !cluster.join_rate_limiter.allow(request.source_addr) {
        return Err(ValidationError::RateLimited);
    }

    // 5. Verify claimed resources (future: challenge-response)
    // Could require node to perform small computation to prove GPU exists

    Ok(())
}

impl NodeCapabilities {
    fn is_plausible(&self) -> bool {
        // Sanity checks on reported capabilities

        // CPU cores should be reasonable (1-1024)
        if self.hardware.cpu_cores == 0 || self.hardware.cpu_cores > 1024 {
            return false;
        }

        // RAM should be reasonable (1GB - 16TB)
        let gb = self.hardware.ram_bytes / (1024 * 1024 * 1024);
        if gb == 0 || gb > 16 * 1024 {
            return false;
        }

        // Network bandwidth should match known hardware (1-1000 Gbps)
        if self.hardware.network_bandwidth_gbps < 1.0
            || self.hardware.network_bandwidth_gbps > 1000.0 {
            return false;
        }

        // TFLOPS should correlate with GPU type
        for gpu in &self.hardware.gpu_devices {
            let expected_tflops = match gpu.name.as_str() {
                name if name.contains("H100") => 1000.0..=2000.0,
                name if name.contains("A100") => 300.0..=600.0,
                name if name.contains("V100") => 100.0..=200.0,
                _ => 1.0..=3000.0,  // Unknown GPU, be generous
            };

            if !expected_tflops.contains(&gpu.flops_fp16) {
                return false;
            }
        }

        true
    }
}
```

### 9.4 Secure Model Distribution

Verify model integrity:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedModelManifest {
    pub manifest: ModelManifest,
    pub signature: Vec<u8>,
    pub signer: PublicKey,
}

async fn verify_model_manifest(
    signed: &SignedModelManifest,
    trusted_publishers: &[PublicKey],
) -> Result<()> {
    // 1. Verify signature
    let manifest_bytes = serde_json::to_vec(&signed.manifest)?;

    if !verify_signature(&manifest_bytes, &signed.signature, &signed.signer) {
        return Err(SecurityError::InvalidManifestSignature);
    }

    // 2. Check if signer is trusted
    if !trusted_publishers.contains(&signed.signer) {
        return Err(SecurityError::UntrustedPublisher {
            public_key: signed.signer.clone(),
        });
    }

    // 3. Verify all file checksums in manifest
    for file in &signed.manifest.weight_files {
        // Checksums will be verified during download
        // Manifest signature ensures checksums weren't tampered with
    }

    Ok(())
}
```

---

## 10. Implementation Roadmap

### Phase 1: Basic Initialization (Milestone 1)

**Goal**: Single-node cold start to ready state

**Tasks**:
- [ ] Configuration loading from TOML/YAML
- [ ] Hardware capability detection
- [ ] Model manifest parsing
- [ ] Weight loading from filesystem
- [ ] Checksum verification
- [ ] Basic readiness checks

**Deliverable**: Single node can load and validate a model

### Phase 2: Cluster Formation (Milestone 2)

**Goal**: Multi-node cluster with static discovery

**Tasks**:
- [ ] Static peer discovery
- [ ] Join request/response protocol
- [ ] Basic Raft leader election
- [ ] Heartbeat mechanism
- [ ] Cluster state management

**Deliverable**: Nodes can form a cluster and elect coordinator

### Phase 3: Distributed Model Loading (Milestone 3)

**Goal**: Partition model across nodes

**Tasks**:
- [ ] Partition assignment algorithm
- [ ] Distributed weight loading
- [ ] Cross-validation with Merkle trees
- [ ] Progress monitoring
- [ ] Failure detection during load

**Deliverable**: Cluster can partition and load a model across multiple nodes

### Phase 4: Failure Recovery (Milestone 4)

**Goal**: Handle failures during initialization

**Tasks**:
- [ ] Timeout handling
- [ ] Retry logic with exponential backoff
- [ ] Coordinator failover
- [ ] Partition reassignment on failure
- [ ] Checkpoint/restart support

**Deliverable**: Cluster can recover from node failures during initialization

### Phase 5: Advanced Features (Milestone 5)

**Goal**: Production-ready initialization

**Tasks**:
- [ ] DNS/Consul discovery
- [ ] S3/object storage support
- [ ] Adaptive timeouts
- [ ] Authentication and TLS
- [ ] Byzantine behavior detection
- [ ] Graceful shutdown and restart

**Deliverable**: Production-ready initialization system

### Phase 6: Optimization (Milestone 6)

**Goal**: Minimize initialization time

**Tasks**:
- [ ] Parallel weight loading
- [ ] Weight prefetching
- [ ] Incremental model updates
- [ ] Hot reload support
- [ ] Kernel caching

**Deliverable**: Sub-minute initialization for common scenarios

---

## Appendix A: State Transition Diagrams

### Full Node State Machine

```
                    ┌──────────────────┐
                    │   COLD           │
                    │  (Process start)  │
                    └────────┬──────────┘
                             │
                     Load configuration
                     Initialize logging
                             │
                             ▼
                    ┌──────────────────┐
                    │   BOOTSTRAP      │
                    │ (Detect hardware) │
                    └────────┬──────────┘
                             │
                      Initialize network
                      Discover peers
                             │
                             ▼
                    ┌──────────────────┐
                    │   JOINING        │
                    │ (Join cluster)    │
                    └────────┬──────────┘
                             │
                      Send JOIN_REQUEST
                      Receive assignment
                             │
                             ▼
                    ┌──────────────────┐
                    │   LOADING        │
                    │ (Load model)      │
                    └────────┬──────────┘
                             │
                      Fetch weight files
                      Verify checksums
                      Allocate GPU memory
                             │
                             ▼
                    ┌──────────────────┐
                    │   VALIDATING     │
                    │ (Run self-test)   │
                    └────────┬──────────┘
                             │
                      Compile kernels
                      Warmup inference
                      Cross-validate
                             │
                             ▼
                    ┌──────────────────┐
                    │   READY          │
                    │ (Awaiting work)   │
                    └────────┬──────────┘
                             │
                      Receive first request
                             │
                             ▼
                    ┌──────────────────┐
                    │   OPERATIONAL    │
                    │ (Serving traffic) │
                    └──────────────────┘

Error Paths:
  * ──timeout──> RECOVERY_ASSESS ──retry──> (previous state)
  * ──critical──> FAILED ──manual──> COLD
```

---

## Appendix B: Message Sequence Diagrams

### Successful Cluster Formation

```
Node A       Node B       Node C       Coordinator Election
  │            │            │
  │ COLD       │ COLD       │ COLD
  ├──config───>│            │
  ├──network──>│            │
  │            ├──config───>│
  │            ├──network──>│
  │            │            ├──config─────────>
  │            │            ├──network────────>
  │            │            │
  │<─discover──┼──discover─>│
  │            │            │
  │────────────┼─REQUEST_VOTE─────────────────>
  │            │            │
  │<───────────┼──VOTE_GRANTED───────────────>
  │            │            │
  │ (A becomes coordinator)  │
  │            │            │
  │─HEARTBEAT─>│            │
  │────────────┼─HEARTBEAT─>│
  │            │            │
  │<─ACK───────│            │
  │<───────────┼───ACK──────│
  │            │            │
  │─ASSIGN────>│            │
  │────────────┼─ASSIGN────>│
  │            │            │
  │            │  [load weights]
  │            │            │  [load weights]
  │            │            │
  │<─PROGRESS──│            │
  │<───────────┼─PROGRESS───│
  │            │            │
  │─VALIDATE──>│            │
  │────────────┼─VALIDATE──>│
  │            │            │
  │<─CHECKSUM──│            │
  │<───────────┼─CHECKSUM───│
  │            │            │
  │ (compute Merkle root)    │
  │            │            │
  │─READY─────>│            │
  │────────────┼─READY─────>│
  │            │            │
  ALL NODES OPERATIONAL      │
```

---

## Appendix C: Configuration Examples

### Minimal Configuration (Development)

```toml
[node]
id = "dev-node-0"
role = "worker"

[cluster]
discovery_method = "static"
seed_nodes = ["localhost:7000"]
cluster_name = "dev-cluster"
quorum_size = 1

[model]
source_type = "filesystem"
source_path = "./models/tiny-llama"

[network]
bind_address = "127.0.0.1:7000"
```

### Production Configuration

```toml
[node]
id = "${HOSTNAME}"
role = "${NODE_ROLE}"  # From environment

[cluster]
discovery_method = "consul"
consul_address = "consul.internal:8500"
cluster_name = "butterfly-prod"
quorum_size = 5

[model]
source_type = "s3"
source_path = "s3://ml-models/llama-70b-v2/"
manifest_hash = "sha256:abcdef123456..."
partition_strategy = "layer_range"

[network]
bind_address = "0.0.0.0:7000"
external_address = "${EXTERNAL_IP}:7000"
max_message_size = "2GB"
compression = "zstd"

[resources]
max_memory_gb = 80
max_gpu_memory_gb = 40

[security]
tls_cert = "/etc/butterfly/tls/node.crt"
tls_key = "/etc/butterfly/tls/node.key"
ca_cert = "/etc/butterfly/tls/ca.crt"
trusted_keys_file = "/etc/butterfly/trusted_keys.json"

[timeouts]
model_load_timeout_secs = 600
join_timeout_secs = 120

[observability]
metrics_port = 9090
tracing_endpoint = "jaeger.internal:6831"
log_level = "info"
```

---

## Appendix D: Error Codes and Troubleshooting

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `INIT_001` | Configuration file not found | Check file path, ensure readable |
| `INIT_002` | Invalid configuration syntax | Validate TOML/YAML syntax |
| `INIT_003` | No GPU detected | Check GPU drivers, CUDA/ROCm installation |
| `INIT_004` | Insufficient memory | Add RAM or reduce model size |
| `NET_001` | Cannot bind to address | Check port not in use, firewall rules |
| `NET_002` | No seed nodes reachable | Check network connectivity, DNS |
| `NET_003` | Connection refused | Ensure seed nodes are running |
| `CLUSTER_001` | Quorum not reached | Start more nodes, check network |
| `CLUSTER_002` | Election timeout | Check network latency, increase timeout |
| `CLUSTER_003` | Duplicate node ID | Ensure unique IDs, regenerate if needed |
| `MODEL_001` | Manifest not found | Check model path, permissions |
| `MODEL_002` | Checksum mismatch | Re-download weights, check storage integrity |
| `MODEL_003` | Unsupported model format | Convert model to supported format |
| `MODEL_004` | Model too large | Add nodes or use quantization |
| `AUTH_001` | Invalid signature | Check node keys, regenerate if corrupted |
| `AUTH_002` | Untrusted public key | Add key to trusted store |
| `AUTH_003` | Certificate expired | Renew TLS certificates |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Authors**: Butterfly Distributed Systems Team
**Status**: Design Specification (Implementation Pending)
