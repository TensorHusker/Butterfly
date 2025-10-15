# Butterfly Bootstrap Protocol Specification

## Document Information

**Version**: 1.0
**Status**: Design Specification
**Last Updated**: 2025-10-11
**Protocol Version**: `butterfly-bootstrap/1.0`

## Table of Contents

1. [Protocol Overview](#1-protocol-overview)
2. [Message Format Specification](#2-message-format-specification)
3. [Handshake Sequences](#3-handshake-sequences)
4. [Cluster Discovery Protocol](#4-cluster-discovery-protocol)
5. [Raft-Based Leader Election](#5-raft-based-leader-election)
6. [Model Distribution Protocol](#6-model-distribution-protocol)
7. [Version Compatibility](#7-version-compatibility)
8. [Error Handling](#8-error-handling)
9. [Security Protocol](#9-security-protocol)
10. [Wire Format](#10-wire-format)

---

## 1. Protocol Overview

### 1.1 Design Principles

The Bootstrap Protocol follows these principles:

1. **Determinism**: Same inputs always produce same outputs
2. **Idempotency**: Repeated messages have same effect as single message
3. **Backward Compatibility**: New versions can communicate with old versions
4. **Fail-Safe**: Protocol failures lead to safe shutdown, not undefined behavior
5. **Minimalism**: Smallest message set that achieves correctness

### 1.2 Protocol Layers

```
┌─────────────────────────────────────────┐
│   Application Layer (Bootstrap Logic)   │
├─────────────────────────────────────────┤
│   Message Layer (Structured Messages)   │
├─────────────────────────────────────────┤
│   Transport Layer (QUIC/gRPC/SHM)      │
├─────────────────────────────────────────┤
│   Security Layer (TLS 1.3, Ed25519)    │
└─────────────────────────────────────────┘
```

### 1.3 Message Categories

| Category | Purpose | Reliability | Ordering |
|----------|---------|-------------|----------|
| **Discovery** | Find peers | Best-effort | Unordered |
| **Join** | Cluster membership | Reliable | Ordered |
| **Election** | Leader election | Reliable | Ordered |
| **Assignment** | Work distribution | Reliable | Ordered |
| **Model** | Weight transfer | Reliable | Ordered |
| **Validation** | Consistency checks | Reliable | Unordered |
| **Health** | Heartbeats | Best-effort | Unordered |

---

## 2. Message Format Specification

### 2.1 Base Message Structure

All messages share this envelope:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapMessage {
    /// Protocol version (semantic versioning)
    pub protocol_version: Version,

    /// Message type discriminator
    pub message_type: MessageType,

    /// Unique message ID for deduplication
    pub message_id: MessageId,

    /// Sender node ID
    pub sender: NodeId,

    /// Optional recipient (broadcast if None)
    pub recipient: Option<NodeId>,

    /// Message timestamp (Unix milliseconds)
    pub timestamp: i64,

    /// Message payload
    pub payload: MessagePayload,

    /// Cryptographic signature (Ed25519)
    pub signature: Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

pub const CURRENT_PROTOCOL_VERSION: Version = Version {
    major: 1,
    minor: 0,
    patch: 0,
};
```

### 2.2 Message Type Enumeration

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum MessageType {
    // Discovery (0x0000 - 0x00FF)
    DiscoveryRequest = 0x0001,
    DiscoveryResponse = 0x0002,
    DiscoveryAnnounce = 0x0003,

    // Join (0x0100 - 0x01FF)
    JoinRequest = 0x0101,
    JoinResponse = 0x0102,
    JoinAck = 0x0103,
    LeaveNotification = 0x0104,

    // Election (0x0200 - 0x02FF)
    RequestVote = 0x0201,
    VoteResponse = 0x0202,
    AppendEntries = 0x0203,
    AppendEntriesResponse = 0x0204,
    LeaderHeartbeat = 0x0205,

    // Assignment (0x0300 - 0x03FF)
    PartitionAssignment = 0x0301,
    AssignmentAck = 0x0302,
    AssignmentReject = 0x0303,

    // Model (0x0400 - 0x04FF)
    ModelManifestRequest = 0x0401,
    ModelManifestResponse = 0x0402,
    WeightChunkRequest = 0x0403,
    WeightChunkResponse = 0x0404,
    LoadProgress = 0x0405,
    LoadComplete = 0x0406,

    // Validation (0x0500 - 0x05FF)
    ChecksumRequest = 0x0501,
    ChecksumReport = 0x0502,
    ValidationResult = 0x0503,

    // Health (0x0600 - 0x06FF)
    Heartbeat = 0x0601,
    HealthStatus = 0x0602,

    // Error (0xFF00 - 0xFFFF)
    Error = 0xFF01,
}
```

### 2.3 Payload Definitions

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    // Discovery
    DiscoveryRequest(DiscoveryRequestPayload),
    DiscoveryResponse(DiscoveryResponsePayload),
    DiscoveryAnnounce(DiscoveryAnnouncePayload),

    // Join
    JoinRequest(JoinRequestPayload),
    JoinResponse(JoinResponsePayload),
    JoinAck(JoinAckPayload),
    LeaveNotification(LeaveNotificationPayload),

    // Election
    RequestVote(RequestVotePayload),
    VoteResponse(VoteResponsePayload),
    AppendEntries(AppendEntriesPayload),
    AppendEntriesResponse(AppendEntriesResponsePayload),
    LeaderHeartbeat(LeaderHeartbeatPayload),

    // Assignment
    PartitionAssignment(PartitionAssignmentPayload),
    AssignmentAck(AssignmentAckPayload),
    AssignmentReject(AssignmentRejectPayload),

    // Model
    ModelManifestRequest(ModelManifestRequestPayload),
    ModelManifestResponse(ModelManifestResponsePayload),
    WeightChunkRequest(WeightChunkRequestPayload),
    WeightChunkResponse(WeightChunkResponsePayload),
    LoadProgress(LoadProgressPayload),
    LoadComplete(LoadCompletePayload),

    // Validation
    ChecksumRequest(ChecksumRequestPayload),
    ChecksumReport(ChecksumReportPayload),
    ValidationResult(ValidationResultPayload),

    // Health
    Heartbeat(HeartbeatPayload),
    HealthStatus(HealthStatusPayload),

    // Error
    Error(ErrorPayload),
}
```

---

## 3. Handshake Sequences

### 3.1 Initial Connection Handshake

When two nodes connect for the first time:

```
Node A                                    Node B
  │                                         │
  │──────── TCP/QUIC Connect ──────────────>│
  │<─────── Connection Accepted ────────────│
  │                                         │
  │──────── DiscoveryRequest ──────────────>│
  │  {                                      │
  │    protocol_version: 1.0.0,            │
  │    node_id: A,                         │
  │    capabilities: {...},                │
  │    cluster_name: "butterfly-prod"      │
  │  }                                      │
  │                                         │
  │<─────── DiscoveryResponse ──────────────│
  │  {                                      │
  │    protocol_version: 1.0.0,            │
  │    node_id: B,                         │
  │    capabilities: {...},                │
  │    coordinator_id: Some(C),            │
  │    peer_list: [B, C, D]                │
  │  }                                      │
  │                                         │
  │  [Connection established]               │
```

**DiscoveryRequestPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryRequestPayload {
    /// Cluster name this node wants to join
    pub cluster_name: String,

    /// Node's reported capabilities
    pub capabilities: NodeCapabilities,

    /// List of other peers this node knows about
    pub known_peers: Vec<PeerInfo>,
}
```

**DiscoveryResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResponsePayload {
    /// Whether this node accepts the connection
    pub accepted: bool,

    /// Reason if rejected
    pub rejection_reason: Option<String>,

    /// Responder's capabilities
    pub capabilities: NodeCapabilities,

    /// Current coordinator (if one exists)
    pub coordinator_id: Option<NodeId>,

    /// List of peers in the cluster
    pub peer_list: Vec<PeerInfo>,

    /// Current cluster configuration
    pub cluster_config: Option<ClusterConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub address: SocketAddr,
    pub capabilities: NodeCapabilities,
    pub joined_at: i64,  // Unix timestamp
}
```

### 3.2 Cluster Join Handshake

After discovery, node joins the cluster formally:

```
Worker Node A                             Coordinator C
  │                                         │
  │──────── JoinRequest ───────────────────>│
  │  {                                      │
  │    node_id: A,                         │
  │    capabilities: {...},                │
  │    protocol_version: 1.0.0,            │
  │    public_key: <Ed25519 key>,          │
  │    signature: <signed request>         │
  │  }                                      │
  │                                         │
  │                                   [Validate signature]
  │                                   [Check capacity]
  │                                   [Allocate partition]
  │                                         │
  │<─────── JoinResponse ───────────────────│
  │  {                                      │
  │    status: Accepted,                   │
  │    node_id: A,                         │
  │    cluster_config: {...},              │
  │    model_manifest: {...},              │
  │    assignment_pending: true            │
  │  }                                      │
  │                                         │
  │──────── JoinAck ───────────────────────>│
  │  {                                      │
  │    node_id: A,                         │
  │    ready_for_assignment: true          │
  │  }                                      │
  │                                         │
  │  [Node enters LOADING state]            │
```

**JoinRequestPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinRequestPayload {
    /// Node's self-assigned ID (UUID)
    pub node_id: NodeId,

    /// Hardware and performance capabilities
    pub capabilities: NodeCapabilities,

    /// Protocol version node implements
    pub protocol_version: Version,

    /// Public key for authentication (Ed25519)
    pub public_key: [u8; 32],

    /// Optional: TLS certificate
    pub certificate: Option<Vec<u8>>,

    /// Reason for joining (first start, rejoin after failure, etc.)
    pub join_reason: JoinReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinReason {
    InitialStart,
    RejoinAfterFailure,
    RejoinAfterRestart,
    RejoinAfterNetworkPartition,
}
```

**JoinResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResponsePayload {
    /// Whether join was accepted
    pub status: JoinStatus,

    /// Confirmed node ID (may differ from request if collision)
    pub node_id: NodeId,

    /// Current cluster configuration
    pub cluster_config: ClusterConfig,

    /// Model manifest for loading
    pub model_manifest: ModelManifest,

    /// Epoch number
    pub epoch: u64,

    /// Whether partition assignment will follow
    pub assignment_pending: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinStatus {
    Accepted,
    Rejected { reason: String },
    Pending { retry_after_ms: u64 },
}
```

---

## 4. Cluster Discovery Protocol

### 4.1 Multicast Discovery (Optional)

For local network discovery without seed nodes:

```
Node A                                    Multicast Group
  │                                         │
  │──────── DiscoveryAnnounce ─────────────>│
  │  {                                      │
  │    node_id: A,                         │
  │    address: 10.0.1.10:7000,            │
  │    cluster_name: "butterfly-prod",     │
  │    capabilities: {...}                 │
  │  }                                      │
  │                                         │
  │<─────── DiscoveryAnnounce (from B) ─────│
  │<─────── DiscoveryAnnounce (from C) ─────│
  │                                         │
  │  [Connect to discovered peers]          │
```

**DiscoveryAnnouncePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryAnnouncePayload {
    /// Node announcing itself
    pub node_id: NodeId,

    /// Address to connect to
    pub address: SocketAddr,

    /// Cluster this node belongs to
    pub cluster_name: String,

    /// Basic capabilities
    pub capabilities: NodeCapabilities,

    /// Whether this node is coordinator
    pub is_coordinator: bool,

    /// How long this announcement is valid (seconds)
    pub ttl: u32,
}
```

### 4.2 Consul/etcd Integration

Discovery via external service:

```rust
// Register with Consul
pub async fn register_with_consul(
    consul_addr: &str,
    node_info: &NodeInfo,
) -> Result<()> {
    let client = consul::Client::new(consul_addr)?;

    let registration = ServiceRegistration {
        id: node_info.node_id.to_string(),
        name: "butterfly-node".to_string(),
        address: node_info.external_address.ip().to_string(),
        port: node_info.external_address.port(),
        tags: vec![
            format!("cluster:{}", node_info.cluster_name),
            format!("version:{}", node_info.protocol_version),
        ],
        meta: hashmap! {
            "node_id" => node_info.node_id.to_string(),
            "capabilities" => serde_json::to_string(&node_info.capabilities)?,
        },
        check: Some(ServiceCheck {
            http: format!("http://{}:{}/health", node_info.external_address.ip(), 8080),
            interval: "10s".to_string(),
            timeout: "2s".to_string(),
        }),
    };

    client.agent().service_register(&registration).await?;

    Ok(())
}

// Discover peers from Consul
pub async fn discover_from_consul(
    consul_addr: &str,
    cluster_name: &str,
) -> Result<Vec<PeerInfo>> {
    let client = consul::Client::new(consul_addr)?;

    let services = client
        .catalog()
        .service("butterfly-node", None)
        .await?;

    let peers = services
        .iter()
        .filter(|s| s.tags.contains(&format!("cluster:{}", cluster_name)))
        .map(|s| {
            let node_id = NodeId::from_str(s.meta.get("node_id").unwrap())?;
            let capabilities: NodeCapabilities = serde_json::from_str(
                s.meta.get("capabilities").unwrap()
            )?;

            Ok(PeerInfo {
                node_id,
                address: format!("{}:{}", s.address, s.port).parse()?,
                capabilities,
                joined_at: 0,  // Not available from Consul
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(peers)
}
```

---

## 5. Raft-Based Leader Election

### 5.1 Election Initiation

When no leader exists or heartbeat timeout occurs:

```
Follower A                   Follower B                  Follower C
    │                            │                           │
    │  [Election timeout]        │                           │
    │  term = 1                  │                           │
    │  vote for self             │                           │
    │                            │                           │
    │──── RequestVote ──────────>│                           │
    │  {                         │                           │
    │    term: 1,                │                           │
    │    candidate_id: A,        │                           │
    │    last_log_index: 10,     │                           │
    │    last_log_term: 0        │                           │
    │  }                         │                           │
    │                            │                           │
    │────────────────────────────┼──── RequestVote ─────────>│
    │                            │                           │
    │                            │  [Check log up-to-date]   │
    │                            │  [Grant vote]             │
    │                            │                           │
    │<──── VoteResponse ─────────│                           │
    │  {                         │                           │
    │    term: 1,                │                           │
    │    vote_granted: true      │                           │
    │  }                         │                           │
    │                            │                           │
    │<───────────────────────────┼──── VoteResponse ─────────│
    │                            │                           │
    │  [Received majority votes] │                           │
    │  [Become leader]           │                           │
    │                            │                           │
    │──── LeaderHeartbeat ──────>│                           │
    │────────────────────────────┼──── LeaderHeartbeat ─────>│
```

**RequestVotePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVotePayload {
    /// Candidate's term
    pub term: u64,

    /// Candidate requesting vote
    pub candidate_id: NodeId,

    /// Index of candidate's last log entry
    pub last_log_index: u64,

    /// Term of candidate's last log entry
    pub last_log_term: u64,

    /// Whether this is a pre-vote (optional optimization)
    pub pre_vote: bool,
}
```

**VoteResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponsePayload {
    /// Current term for candidate to update itself
    pub term: u64,

    /// True means candidate received vote
    pub vote_granted: bool,

    /// Reason if vote not granted
    pub rejection_reason: Option<VoteRejectionReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteRejectionReason {
    /// Already voted for another candidate this term
    AlreadyVoted { voted_for: NodeId },

    /// Candidate's log is not up-to-date
    LogNotUpToDate {
        our_last_index: u64,
        our_last_term: u64,
    },

    /// Our term is higher
    StaleTerm { our_term: u64 },
}
```

### 5.2 Leader Heartbeats

After election, leader maintains authority via heartbeats:

```
Leader A                     Follower B                  Follower C
    │                            │                           │
    │──── LeaderHeartbeat ──────>│                           │
    │  {                         │                           │
    │    term: 1,                │                           │
    │    leader_id: A,           │                           │
    │    commit_index: 5,        │                           │
    │    timestamp: T            │                           │
    │  }                         │                           │
    │                            │                           │
    │────────────────────────────┼──── LeaderHeartbeat ─────>│
    │                            │                           │
    │                            │  [Reset election timer]   │
    │                            │                           │
    │  [100ms interval]          │                           │
    │                            │                           │
    │──── LeaderHeartbeat ──────>│                           │
    │────────────────────────────┼──── LeaderHeartbeat ─────>│
```

**LeaderHeartbeatPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderHeartbeatPayload {
    /// Leader's term
    pub term: u64,

    /// Leader's node ID
    pub leader_id: NodeId,

    /// Index of highest log entry known to be committed
    pub commit_index: u64,

    /// Leader's current timestamp (for clock skew detection)
    pub timestamp: i64,

    /// Leader's current cluster view
    pub cluster_members: Vec<NodeId>,
}
```

### 5.3 Log Replication (for Configuration Changes)

When coordinator makes configuration changes:

```
Leader A                     Follower B                  Follower C
    │                            │                           │
    │──── AppendEntries ────────>│                           │
    │  {                         │                           │
    │    term: 1,                │                           │
    │    leader_id: A,           │                           │
    │    prev_log_index: 5,      │                           │
    │    prev_log_term: 1,       │                           │
    │    entries: [              │                           │
    │      LogEntry::ConfigChange│                           │
    │    ],                      │                           │
    │    leader_commit: 5        │                           │
    │  }                         │                           │
    │                            │                           │
    │────────────────────────────┼──── AppendEntries ───────>│
    │                            │                           │
    │                            │  [Append to log]          │
    │                            │                           │
    │<──── AppendEntriesResponse ─│                           │
    │  {                         │                           │
    │    term: 1,                │                           │
    │    success: true,          │                           │
    │    match_index: 6          │                           │
    │  }                         │                           │
    │                            │                           │
    │<───────────────────────────┼──── AppendEntriesResponse │
    │                            │                           │
    │  [Majority replicated]     │                           │
    │  [Advance commit_index]    │                           │
```

**AppendEntriesPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesPayload {
    /// Leader's term
    pub term: u64,

    /// Leader's node ID
    pub leader_id: NodeId,

    /// Index of log entry immediately preceding new ones
    pub prev_log_index: u64,

    /// Term of prev_log_index entry
    pub prev_log_term: u64,

    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,

    /// Leader's commit index
    pub leader_commit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    /// Cluster configuration change
    ConfigChange(ClusterConfig),

    /// Partition assignment
    PartitionAssignment {
        epoch: u64,
        assignments: Vec<PartitionAssignment>,
    },

    /// Model manifest update
    ModelManifestUpdate {
        manifest: ModelManifest,
        checksum: [u8; 32],
    },

    /// Checkpoint metadata
    Checkpoint {
        epoch: u64,
        merkle_root: [u8; 32],
    },

    /// No-op (for leader initialization)
    NoOp,
}
```

**AppendEntriesResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponsePayload {
    /// Current term, for leader to update itself
    pub term: u64,

    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,

    /// Follower's last log index (for leader to update next_index)
    pub match_index: u64,

    /// Hint for faster log backtracking on conflict
    pub conflict_index: Option<u64>,
    pub conflict_term: Option<u64>,
}
```

---

## 6. Model Distribution Protocol

### 6.1 Manifest Distribution

Coordinator distributes model manifest to all workers:

```
Coordinator C                            Worker A
    │                                        │
    │──── ModelManifestResponse ────────────>│
    │  {                                     │
    │    manifest: {                         │
    │      model_id: "llama-70b-v2",        │
    │      architecture: "llama",           │
    │      total_layers: 80,                │
    │      weight_files: [                  │
    │        {                              │
    │          path: "layer_00-09.safetensors",
    │          size_bytes: 8589934592,     │
    │          checksum: [0xab, ...],      │
    │          layer_range: {start: 0, end: 10}
    │        },                             │
    │        ...                            │
    │      ],                               │
    │    },                                  │
    │    signature: <Ed25519 sig>,          │
    │    source_config: {                   │
    │      type: "s3",                      │
    │      bucket: "ml-models",             │
    │      prefix: "llama-70b-v2/"          │
    │    }                                   │
    │  }                                     │
```

**ModelManifestResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifestResponsePayload {
    /// Model manifest
    pub manifest: ModelManifest,

    /// Cryptographic signature of manifest
    pub signature: Vec<u8>,

    /// Public key of signer
    pub signer_public_key: [u8; 32],

    /// Where to fetch weights from
    pub source_config: ModelSourceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSourceConfig {
    Filesystem {
        base_path: String,
    },
    S3 {
        bucket: String,
        prefix: String,
        region: String,
    },
    HTTP {
        base_url: String,
        auth_token: Option<String>,
    },
    Coordinator {
        chunk_size_bytes: u64,
    },
    P2P {
        tracker_nodes: Vec<NodeId>,
    },
}
```

### 6.2 Partition Assignment Distribution

```
Coordinator C                            Worker A
    │                                        │
    │──── PartitionAssignment ──────────────>│
    │  {                                     │
    │    epoch: 1,                           │
    │    node_id: A,                         │
    │    layer_range: {start: 0, end: 20},  │
    │    required_files: [                   │
    │      "layer_00-09.safetensors",       │
    │      "layer_10-19.safetensors"        │
    │    ],                                  │
    │    estimated_memory_bytes: 17179869184,│
    │    dependencies: {                     │
    │      predecessors: [],                 │
    │      successors: [B]                   │
    │    }                                   │
    │  }                                     │
    │                                        │
    │                                   [Validate assignment]
    │                                   [Check resources]
    │                                        │
    │<──── AssignmentAck ─────────────────────│
    │  {                                     │
    │    node_id: A,                         │
    │    epoch: 1,                           │
    │    accepted: true,                     │
    │    estimated_load_time_secs: 120       │
    │  }                                     │
```

**PartitionAssignmentPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignmentPayload {
    /// Assignment epoch
    pub epoch: u64,

    /// Node receiving this assignment
    pub node_id: NodeId,

    /// Layer range to compute
    pub layer_range: LayerRange,

    /// Weight files needed for this partition
    pub required_files: Vec<String>,

    /// Estimated memory consumption
    pub estimated_memory_bytes: u64,

    /// Dependencies on other nodes
    pub dependencies: PartitionDependencies,

    /// Deadline to complete loading (Unix timestamp)
    pub load_deadline: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionDependencies {
    /// Nodes this partition receives input from
    pub predecessors: Vec<NodeId>,

    /// Nodes this partition sends output to
    pub successors: Vec<NodeId>,
}
```

**AssignmentAckPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentAckPayload {
    /// Node acknowledging
    pub node_id: NodeId,

    /// Assignment epoch
    pub epoch: u64,

    /// Whether assignment is accepted
    pub accepted: bool,

    /// Reason if rejected
    pub rejection_reason: Option<String>,

    /// Estimated time to complete loading (seconds)
    pub estimated_load_time_secs: f64,
}
```

### 6.3 Weight Chunk Transfer (Coordinator Distribution)

If weights are distributed via coordinator:

```
Worker A                                 Coordinator C
    │                                        │
    │──── WeightChunkRequest ───────────────>│
    │  {                                     │
    │    file_id: "layer_00-09.safetensors", │
    │    chunk_index: 0,                     │
    │    chunk_size: 134217728  // 128 MB   │
    │  }                                     │
    │                                        │
    │                                   [Fetch chunk]
    │                                   [Compress]
    │                                        │
    │<──── WeightChunkResponse ───────────────│
    │  {                                     │
    │    file_id: "layer_00-09.safetensors", │
    │    chunk_index: 0,                     │
    │    total_chunks: 64,                   │
    │    data: <compressed bytes>,           │
    │    checksum: [0x12, ...],              │
    │    compressed: true                    │
    │  }                                     │
    │                                        │
    │  [Verify checksum]                     │
    │  [Decompress]                          │
    │  [Append to file]                      │
    │                                        │
    │──── WeightChunkRequest ───────────────>│
    │  {                                     │
    │    file_id: "layer_00-09.safetensors", │
    │    chunk_index: 1,                     │
    │    chunk_size: 134217728               │
    │  }                                     │
    │                                        │
    │  ... (repeat for all chunks)           │
```

**WeightChunkRequestPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightChunkRequestPayload {
    /// File identifier from manifest
    pub file_id: String,

    /// Chunk index (0-based)
    pub chunk_index: u64,

    /// Preferred chunk size in bytes
    pub chunk_size: u64,
}
```

**WeightChunkResponsePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightChunkResponsePayload {
    /// File identifier
    pub file_id: String,

    /// This chunk's index
    pub chunk_index: u64,

    /// Total number of chunks for this file
    pub total_chunks: u64,

    /// Chunk data (possibly compressed)
    pub data: Vec<u8>,

    /// Checksum of this chunk (before compression)
    pub checksum: [u8; 32],

    /// Whether data is compressed
    pub compressed: bool,

    /// Compression algorithm if compressed
    pub compression: Option<CompressionAlgorithm>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Zstd,
    Lz4,
    None,
}
```

### 6.4 Load Progress Reporting

Workers report progress periodically:

```
Worker A                                 Coordinator C
    │                                        │
    │──── LoadProgress ─────────────────────>│
    │  {                                     │
    │    node_id: A,                         │
    │    epoch: 1,                           │
    │    progress_percent: 25.0,             │
    │    bytes_loaded: 4294967296,           │
    │    total_bytes: 17179869184,           │
    │    estimated_completion_secs: 90,      │
    │    current_file: "layer_00-09.safetensors"
    │  }                                     │
    │                                        │
    │  [Continue loading...]                 │
    │                                        │
    │──── LoadProgress ─────────────────────>│
    │  {                                     │
    │    node_id: A,                         │
    │    epoch: 1,                           │
    │    progress_percent: 50.0,             │
    │    bytes_loaded: 8589934592,           │
    │    ...                                 │
    │  }                                     │
    │                                        │
    │  ... (continue until 100%)             │
    │                                        │
    │──── LoadComplete ─────────────────────>│
    │  {                                     │
    │    node_id: A,                         │
    │    epoch: 1,                           │
    │    total_load_time_secs: 115.3,        │
    │    memory_used_bytes: 17179869184,     │
    │    partition_checksum: [0xab, ...]     │
    │  }                                     │
```

**LoadProgressPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadProgressPayload {
    /// Reporting node
    pub node_id: NodeId,

    /// Assignment epoch
    pub epoch: u64,

    /// Progress percentage (0.0 - 100.0)
    pub progress_percent: f64,

    /// Bytes loaded so far
    pub bytes_loaded: u64,

    /// Total bytes to load
    pub total_bytes: u64,

    /// Estimated time to completion (seconds)
    pub estimated_completion_secs: f64,

    /// Current file being loaded
    pub current_file: String,

    /// Whether warmup/compilation has started
    pub warmup_started: bool,
}
```

**LoadCompletePayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCompletePayload {
    /// Node reporting completion
    pub node_id: NodeId,

    /// Assignment epoch
    pub epoch: u64,

    /// Total time taken to load (seconds)
    pub total_load_time_secs: f64,

    /// Actual memory used (bytes)
    pub memory_used_bytes: u64,

    /// Checksum of loaded partition
    pub partition_checksum: [u8; 32],

    /// Whether warmup inference completed successfully
    pub warmup_success: bool,

    /// Warmup inference latency (milliseconds)
    pub warmup_latency_ms: f64,
}
```

### 6.5 Cross-Validation Protocol

After all nodes load, coordinator validates consistency:

```
Coordinator C                            All Workers
    │                                    │   │   │
    │──── ChecksumRequest ──────────────>│   │   │
    │  {                                 │   │   │
    │    epoch: 1                        │   │   │
    │  }                                 │   │   │
    │                                    │   │   │
    │<──── ChecksumReport ───────────────│   │   │
    │  {                                 │   │   │
    │    node_id: A,                     │   │   │
    │    epoch: 1,                       │   │   │
    │    layer_range: {0, 20},           │   │   │
    │    weight_hash: [0xab, ...],       │   │   │
    │    timestamp: T1                   │   │   │
    │  }                                 │   │   │
    │                                    │   │   │
    │<──── ChecksumReport ───────────────┼───│   │
    │<──── ChecksumReport ───────────────┼───┼───│
    │                                    │   │   │
    │  [Build Merkle tree]               │   │   │
    │  [Compute root hash]               │   │   │
    │                                    │   │   │
    │──── ValidationResult ─────────────>│   │   │
    │  {                                 │   │   │
    │    epoch: 1,                       │   │   │
    │    merkle_root: [0x12, ...],       │   │   │
    │    status: Valid,                  │   │   │
    │    cluster_ready: true             │   │   │
    │  }                                 │   │   │
```

**ChecksumRequestPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumRequestPayload {
    /// Epoch to report checksums for
    pub epoch: u64,
}
```

**ChecksumReportPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumReportPayload {
    /// Reporting node
    pub node_id: NodeId,

    /// Assignment epoch
    pub epoch: u64,

    /// Layer range this node loaded
    pub layer_range: LayerRange,

    /// Hash of all weights for this partition
    pub weight_hash: [u8; 32],

    /// Timestamp of checksum computation
    pub timestamp: i64,

    /// Optional: Merkle tree of individual tensors
    pub tensor_checksums: Option<Vec<[u8; 32]>>,
}
```

**ValidationResultPayload**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResultPayload {
    /// Epoch being validated
    pub epoch: u64,

    /// Merkle root of all partition checksums
    pub merkle_root: [u8; 32],

    /// Validation status
    pub status: ValidationStatus,

    /// Whether cluster is ready for inference
    pub cluster_ready: bool,

    /// List of nodes that passed validation
    pub validated_nodes: Vec<NodeId>,

    /// Nodes that failed (if any)
    pub failed_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Invalid { mismatch_count: usize },
    Incomplete { missing_count: usize },
}
```

---

## 7. Version Compatibility

### 7.1 Version Negotiation

Nodes negotiate protocol version during discovery:

```rust
pub fn are_versions_compatible(v1: &Version, v2: &Version) -> bool {
    // Major version must match
    if v1.major != v2.major {
        return false;
    }

    // Minor version: backward compatible
    // Node with higher minor can talk to node with lower minor
    // (higher minor must implement compatibility layer)

    true
}

pub fn select_protocol_version(v1: &Version, v2: &Version) -> Version {
    // Use the lower minor version for communication
    Version {
        major: v1.major.min(v2.major),
        minor: v1.minor.min(v2.minor),
        patch: 0,  // Patch version doesn't affect protocol
    }
}
```

### 7.2 Feature Flags

For gradual feature rollout:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolFeatures {
    /// Bitmap of supported features
    pub features: u64,
}

pub mod feature_flags {
    pub const COMPRESSION: u64 = 1 << 0;
    pub const P2P_DISTRIBUTION: u64 = 1 << 1;
    pub const ENCRYPTION: u64 = 1 << 2;
    pub const ADAPTIVE_TIMEOUTS: u64 = 1 << 3;
    pub const SPECULATIVE_EXECUTION: u64 = 1 << 4;
}

impl ProtocolFeatures {
    pub fn supports(&self, feature: u64) -> bool {
        (self.features & feature) != 0
    }

    pub fn common_features(&self, other: &Self) -> Self {
        Self {
            features: self.features & other.features,
        }
    }
}
```

---

## 8. Error Handling

### 8.1 Error Message Format

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    /// Error code (see error code registry)
    pub code: ErrorCode,

    /// Human-readable error message
    pub message: String,

    /// Original message that caused error (if applicable)
    pub caused_by_message_id: Option<MessageId>,

    /// Additional context
    pub context: HashMap<String, String>,

    /// Whether error is recoverable
    pub recoverable: bool,

    /// Suggested retry delay (if recoverable)
    pub retry_after_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    // Protocol errors (0x1000 - 0x1FFF)
    InvalidMessage = 0x1001,
    UnsupportedVersion = 0x1002,
    InvalidSignature = 0x1003,
    MessageTooLarge = 0x1004,

    // Discovery errors (0x2000 - 0x2FFF)
    ClusterNameMismatch = 0x2001,
    NoPeersAvailable = 0x2002,

    // Join errors (0x3000 - 0x3FFF)
    ClusterFull = 0x3001,
    InsufficientCapabilities = 0x3002,
    DuplicateNodeId = 0x3003,
    JoinRateLimited = 0x3004,

    // Election errors (0x4000 - 0x4FFF)
    StaleTerm = 0x4001,
    LogInconsistency = 0x4002,
    NoQuorum = 0x4003,

    // Model errors (0x5000 - 0x5FFF)
    ManifestNotFound = 0x5001,
    InvalidManifest = 0x5002,
    ChecksumMismatch = 0x5003,
    WeightFileNotFound = 0x5004,
    InsufficientMemory = 0x5005,

    // Validation errors (0x6000 - 0x6FFF)
    ValidationFailed = 0x6001,
    MerkleRootMismatch = 0x6002,

    // Internal errors (0xF000 - 0xFFFF)
    InternalError = 0xF001,
    Timeout = 0xF002,
}
```

### 8.2 Error Recovery Strategies

```rust
pub fn should_retry_on_error(error: &ErrorPayload) -> bool {
    match error.code {
        // Network/transient errors - retry
        ErrorCode::Timeout => true,
        ErrorCode::NoPeersAvailable => true,
        ErrorCode::JoinRateLimited => true,

        // Protocol/semantic errors - don't retry
        ErrorCode::InvalidMessage => false,
        ErrorCode::UnsupportedVersion => false,
        ErrorCode::ClusterNameMismatch => false,

        // Resource errors - may become available
        ErrorCode::ClusterFull => true,
        ErrorCode::InsufficientMemory => false,  // Need human intervention

        _ => error.recoverable,
    }
}

pub fn get_retry_delay(error: &ErrorPayload, attempt: u32) -> Duration {
    // Use exponential backoff
    let base_delay = error.retry_after_ms.unwrap_or(1000);
    let backoff = base_delay * 2u64.pow(attempt.min(5));
    let jitter = rand::random::<u64>() % (backoff / 2);

    Duration::from_millis(backoff + jitter)
}
```

---

## 9. Security Protocol

### 9.1 Message Authentication

All messages must be signed:

```rust
pub fn sign_message(
    message: &BootstrapMessage,
    private_key: &SecretKey,
) -> Signature {
    use ed25519_dalek::Signer;

    // Compute message hash (excluding signature field)
    let message_without_sig = BootstrapMessage {
        signature: Signature([0u8; 64]),
        ..message.clone()
    };

    let message_bytes = bincode::serialize(&message_without_sig).unwrap();
    private_key.sign(&message_bytes)
}

pub fn verify_message_signature(
    message: &BootstrapMessage,
    public_key: &PublicKey,
) -> bool {
    use ed25519_dalek::Verifier;

    let message_without_sig = BootstrapMessage {
        signature: Signature([0u8; 64]),
        ..message.clone()
    };

    let message_bytes = bincode::serialize(&message_without_sig).unwrap();

    public_key
        .verify(&message_bytes, &message.signature)
        .is_ok()
}
```

### 9.2 Replay Attack Prevention

```rust
#[derive(Debug)]
pub struct MessageDeduplicator {
    seen_messages: DashMap<MessageId, Instant>,
    window: Duration,
}

impl MessageDeduplicator {
    pub fn new(window: Duration) -> Self {
        Self {
            seen_messages: DashMap::new(),
            window,
        }
    }

    pub fn is_duplicate(&self, message_id: MessageId, timestamp: i64) -> bool {
        // Check if we've seen this message ID recently
        if let Some(seen_at) = self.seen_messages.get(&message_id) {
            return true;
        }

        // Check if timestamp is within acceptable window
        let now = chrono::Utc::now().timestamp_millis();
        let age = (now - timestamp).abs() as u64;

        if age > self.window.as_millis() as u64 {
            // Message too old or too far in future - reject
            return true;
        }

        // Record this message ID
        self.seen_messages
            .insert(message_id, Instant::now());

        // Periodically clean old entries
        self.cleanup_old_entries();

        false
    }

    fn cleanup_old_entries(&self) {
        let cutoff = Instant::now() - self.window;
        self.seen_messages
            .retain(|_, &mut seen_at| seen_at > cutoff);
    }
}
```

---

## 10. Wire Format

### 10.1 Binary Serialization

Messages are serialized using bincode for efficiency:

```rust
pub fn serialize_message(message: &BootstrapMessage) -> Result<Vec<u8>> {
    bincode::serialize(message).map_err(Into::into)
}

pub fn deserialize_message(bytes: &[u8]) -> Result<BootstrapMessage> {
    bincode::deserialize(bytes).map_err(Into::into)
}
```

### 10.2 Message Framing (Over QUIC)

```
┌─────────────────────────────────────────────────────┐
│  Magic Bytes (4)  │  Version (2)  │  Length (4)     │
├─────────────────────────────────────────────────────┤
│  Message Type (2)  │  Flags (2)  │  Checksum (4)   │
├─────────────────────────────────────────────────────┤
│                                                     │
│              Serialized Payload                     │
│              (bincode-encoded)                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

```rust
pub const MAGIC_BYTES: [u8; 4] = *b"BFLY";

#[derive(Debug, Clone)]
pub struct MessageFrame {
    pub magic: [u8; 4],
    pub version: u16,
    pub length: u32,
    pub message_type: MessageType,
    pub flags: FrameFlags,
    pub checksum: u32,
    pub payload: Vec<u8>,
}

bitflags! {
    pub struct FrameFlags: u16 {
        const COMPRESSED = 1 << 0;
        const ENCRYPTED = 1 << 1;
        const PRIORITY_HIGH = 1 << 2;
        const REQUIRES_ACK = 1 << 3;
    }
}

impl MessageFrame {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_be_bytes());
        buf.extend_from_slice(&self.length.to_be_bytes());
        buf.extend_from_slice(&(self.message_type as u16).to_be_bytes());
        buf.extend_from_slice(&self.flags.bits().to_be_bytes());
        buf.extend_from_slice(&self.checksum.to_be_bytes());
        buf.extend_from_slice(&self.payload);

        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(DecodeError::MessageTooShort);
        }

        let magic: [u8; 4] = bytes[0..4].try_into()?;
        if magic != MAGIC_BYTES {
            return Err(DecodeError::InvalidMagic);
        }

        let version = u16::from_be_bytes(bytes[4..6].try_into()?);
        let length = u32::from_be_bytes(bytes[6..10].try_into()?);
        let message_type = u16::from_be_bytes(bytes[10..12].try_into()?);
        let flags = FrameFlags::from_bits_truncate(u16::from_be_bytes(bytes[12..14].try_into()?));
        let checksum = u32::from_be_bytes(bytes[14..18].try_into()?);

        let payload = bytes[18..].to_vec();

        // Verify checksum
        let computed_checksum = crc32(&payload);
        if computed_checksum != checksum {
            return Err(DecodeError::ChecksumMismatch);
        }

        Ok(Self {
            magic,
            version,
            length,
            message_type: MessageType::try_from(message_type)?,
            flags,
            checksum,
            payload,
        })
    }
}
```

---

## Appendix A: Complete Message Flow Example

### Scenario: 3-Node Cluster Cold Start

```
T=0: All nodes start simultaneously

Node A                  Node B                  Node C
  │                       │                       │
  │ COLD                  │ COLD                  │ COLD
  │                       │                       │
T=5s: Config loaded, network initialized
  │                       │                       │
  │────DiscoveryAnnounce─>│                       │
  │<───DiscoveryAnnounce──│                       │
  │                       │                       │
  │────DiscoveryAnnounce──┼──────────────────────>│
  │<──────────────────────┼───DiscoveryAnnounce───│
  │                       │                       │
  │<──DiscoveryAnnounce───┼──────────────────────>│
  │                       │<──DiscoveryAnnounce───│
  │                       │                       │
T=10s: All nodes discovered each other
  │                       │                       │
  │ [Election timeout: 150ms]                     │
  │ [Election timeout: 200ms]                     │
  │                       │ [Election timeout: 180ms]
  │                       │                       │
T=10.15s: A's timeout fires first
  │                       │                       │
  │────RequestVote───────>│                       │
  │────RequestVote────────┼──────────────────────>│
  │  {term: 1}            │                       │
  │                       │                       │
  │<───VoteResponse───────│                       │
  │  {granted: true}      │                       │
  │                       │                       │
  │<──────────────────────┼───VoteResponse────────│
  │                       │  {granted: true}      │
  │                       │                       │
T=10.16s: A elected leader
  │                       │                       │
  │────LeaderHeartbeat───>│                       │
  │────LeaderHeartbeat────┼──────────────────────>│
  │                       │                       │
T=10.20s: Assign partitions
  │                       │                       │
  │────PartitionAssignment>│                       │
  │  {layers: 0-26}       │                       │
  │                       │                       │
  │────PartitionAssignment─┼──────────────────────>│
  │  {layers: 27-53}      │  {layers: 54-79}      │
  │                       │                       │
  │<───AssignmentAck──────│                       │
  │<──────────────────────┼───AssignmentAck───────│
  │                       │                       │
T=10.25s: Start loading weights
  │                       │                       │
  │  [Load layer_00-26]   │  [Load layer_27-53]   │  [Load layer_54-79]
  │                       │                       │
T=15s: Progress reports
  │────LoadProgress──────>│                       │
  │  {progress: 25%}      │                       │
  │<──────────────────────┼───LoadProgress────────│
  │                       │  {progress: 30%}      │
  │                       │<──LoadProgress────────│
  │                       │  {progress: 20%}      │
  │                       │                       │
T=90s: All loads complete
  │────LoadComplete──────>│                       │
  │<──────────────────────┼───LoadComplete────────│
  │                       │<──LoadComplete────────│
  │                       │                       │
T=91s: Cross-validation
  │────ChecksumRequest───>│                       │
  │────ChecksumRequest────┼──────────────────────>│
  │                       │                       │
  │<───ChecksumReport─────│                       │
  │<──────────────────────┼───ChecksumReport──────│
  │                       │                       │
T=92s: Validation complete
  │                       │                       │
  │  [Compute Merkle root: 0xabcd1234]            │
  │                       │                       │
  │────ValidationResult──>│                       │
  │────ValidationResult───┼──────────────────────>│
  │  {status: Valid, cluster_ready: true}         │
  │                       │                       │
T=93s: CLUSTER READY
  │                       │                       │
  │ OPERATIONAL           │ OPERATIONAL           │ OPERATIONAL
```

---

**Protocol Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Design Specification
**Authors**: Butterfly Distributed Systems Team
