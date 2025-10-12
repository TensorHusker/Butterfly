# Butterfly Configuration System Design

## Executive Summary

The Butterfly configuration system provides a type-safe, hierarchical, and Byzantine-resilient framework for managing distributed inference system parameters. It balances flexibility for operators with strong safety guarantees required in a fault-tolerant distributed system.

**Core Principles:**
1. **Type Safety**: All configuration values validated at parse time using Rust's type system
2. **Hierarchy**: Clear precedence from defaults → files → environment → runtime updates
3. **Byzantine Resilience**: Configuration changes require consensus among honest nodes
4. **Hot Reload**: Safe runtime reconfiguration for non-critical parameters
5. **Auditability**: All configuration changes logged and versioned

**Key Innovation**: Configuration updates treated as distributed consensus operations, ensuring all nodes converge to identical configurations even under Byzantine failures.

---

## 1. Configuration Architecture

### 1.1 Layered Design

```
┌────────────────────────────────────────────────────────┐
│              Application Layer                          │
│  Components access config via strongly-typed API        │
├────────────────────────────────────────────────────────┤
│              Configuration Manager                      │
│  • Hot reload coordination                             │
│  • Validation and type checking                        │
│  • Watch for changes                                   │
├────────────────────────────────────────────────────────┤
│              Source Layer                              │
│  Defaults → File → Environment → Runtime Updates       │
├────────────────────────────────────────────────────────┤
│              Consensus Layer                           │
│  Byzantine agreement on cluster-wide config changes    │
└────────────────────────────────────────────────────────┘
```

### 1.2 Configuration Hierarchy and Precedence

Configuration values are resolved through a hierarchy with clear precedence rules:

**Precedence Order** (highest to lowest):
1. **Runtime Updates**: Dynamic changes via API (requires consensus)
2. **Environment Variables**: `BUTTERFLY_*` prefixed variables
3. **Configuration File**: TOML file specified via CLI
4. **Included Files**: Referenced via `include` directive in main config
5. **Defaults**: Hardcoded sensible defaults in code

**Precedence Rules:**
- Higher precedence sources override lower ones
- Within same source level, last value wins
- Validation occurs after all sources merged
- Invalid configurations rejected entirely (no partial application)

**Scope of Configuration:**
- **Node-Local**: Applied only to single node (e.g., local resource limits)
- **Cluster-Wide**: Must be identical across all nodes (e.g., Byzantine f value)
- **Role-Specific**: Applied to nodes with specific roles (e.g., coordinator settings)

### 1.3 Configuration File Format: TOML

**Rationale for TOML:**
- Native Rust ecosystem support via `serde` and `toml` crates
- Human-readable and editable
- Strong typing with clear syntax
- Native support for nested structures
- Comments for documentation
- No ambiguity (unlike YAML)

**Alternative Formats Considered:**
- YAML: Rejected due to parsing ambiguities and security concerns
- JSON: Rejected due to lack of comments and poor human ergonomics
- RON: Rejected due to limited tooling ecosystem

---

## 2. Configuration Schema

### 2.1 Top-Level Structure

```toml
# Butterfly Configuration File
# Version: 1.0

[metadata]
config_version = "1.0.0"
environment = "production"  # production, staging, development
cluster_name = "butterfly-main"

[include]
# Additional configuration files merged into this config
files = [
    "./network.toml",
    "./resources.toml",
]

[node]
# Node-specific configuration
node_id = "node-001"  # Unique identifier (auto-generated if omitted)
role = "compute"  # coordinator, compute, observer
data_dir = "/var/lib/butterfly"
log_dir = "/var/log/butterfly"

[network]
# Network communication settings

[byzantine]
# Byzantine fault tolerance parameters

[partition]
# Model partitioning strategy configuration

[scheduling]
# Workload scheduling policies

[resources]
# Resource limits and allocation

[observability]
# Metrics, logging, and tracing

[security]
# Authentication and encryption
```

### 2.2 Network Configuration

```toml
[network]
# Cluster-wide: Must be identical across all nodes

# Primary communication protocol
protocol = "quic"  # quic, grpc, tcp

# Coordinator endpoint (all nodes must know this)
coordinator_addr = "10.0.1.10:7890"

# Node listening configuration (node-local)
listen_addr = "0.0.0.0:7890"
external_addr = "10.0.1.11:7890"  # Address other nodes use to reach this node

# Connection management
[network.connection]
max_connections = 1000
connection_timeout_ms = 5000
idle_timeout_ms = 30000
keepalive_interval_ms = 1000

# Buffer sizes (affects memory usage and throughput)
[network.buffers]
send_buffer_bytes = 4194304      # 4MB
recv_buffer_bytes = 4194304      # 4MB
max_message_size_bytes = 104857600  # 100MB (for large tensor transfers)

# Compression for network traffic
[network.compression]
enabled = true
algorithm = "zstd"  # none, lz4, zstd, snappy
level = 3  # Compression level (1-22 for zstd)
min_size_bytes = 1024  # Don't compress messages smaller than this

# Retry and backoff policies
[network.retry]
max_retries = 3
initial_backoff_ms = 100
max_backoff_ms = 5000
backoff_multiplier = 2.0

# Network topology hints (optional, for optimization)
[network.topology]
datacenter = "us-west-2a"
rack = "rack-7"
availability_zone = "az1"
```

### 2.3 Byzantine Coordination Configuration

```toml
[byzantine]
# Cluster-wide: CRITICAL - Must be identical across all nodes

# Byzantine fault tolerance parameter
# System tolerates up to f Byzantine failures with 2f+1 nodes
f_value = 1  # Tolerates 1 Byzantine failure, requires 3+ nodes

# Failure detection parameters (Phi Accrual)
[byzantine.failure_detection]
heartbeat_interval_ms = 100
phi_suspect_threshold = 8.0
phi_failed_threshold = 12.0
max_heartbeat_history = 1000

# Byzantine agreement protocol settings
[byzantine.agreement]
protocol = "optimistic_pbft"  # optimistic_pbft, standard_pbft
pre_prepare_timeout_ms = 1000
prepare_timeout_ms = 2000
commit_timeout_ms = 3000
max_in_flight_agreements = 10

# Checkpoint configuration
[byzantine.checkpoint]
interval_tokens = 10  # Checkpoint every N tokens
retention_count = 100  # Keep last N checkpoints
compression = true
sync_to_disk = true

# Reputation and isolation
[byzantine.reputation]
enabled = true
initial_score = 100.0
suspect_threshold = 50.0
isolate_threshold = 10.0
recovery_rate = 1.0  # Points recovered per successful operation

# View change (leader re-election)
[byzantine.view_change]
timeout_ms = 10000
max_view_change_attempts = 5
```

### 2.4 Partition Configuration

```toml
[partition]
# Strategy for partitioning models across nodes

# Partitioning algorithm selection
strategy = "layer_affinity"  # layer_affinity, min_communication, load_balanced, custom

# Strategy-specific parameters
[partition.layer_affinity]
# Group consecutive layers to minimize inter-node communication
min_layers_per_partition = 2
max_layers_per_partition = 10
prefer_attention_boundaries = true  # Partition at attention layer boundaries

[partition.min_communication]
# Minimize communication volume via graph cut algorithms
algorithm = "kernighan_lin"  # kernighan_lin, spectral, metis
balance_tolerance = 0.15  # Allow 15% load imbalance for better cuts

[partition.load_balanced]
# Evenly distribute compute across nodes
metric = "flops"  # flops, memory, latency
rebalance_threshold = 0.20  # Rebalance if imbalance > 20%

# Model-specific hints (optional, learned over time)
[partition.model_hints]
# Override automatic partitioning for specific models
# "llama-70b" = { strategy = "layer_affinity", layers_per_node = 8 }

# Dynamic repartitioning
[partition.dynamic]
enabled = true
rebalance_interval_secs = 300  # Check for rebalancing every 5 minutes
cost_threshold = 0.10  # Only rebalance if improvement > 10%
max_migrations_per_interval = 2
```

### 2.5 Scheduling Configuration

```toml
[scheduling]
# Workload scheduling and load balancing

# Scheduling policy
policy = "predictive"  # round_robin, least_loaded, predictive, custom

# Queue management
[scheduling.queues]
max_queue_depth = 100
priority_levels = 3
queue_timeout_ms = 60000
backpressure_threshold = 80  # Start backpressure at 80% capacity

# Straggler mitigation
[scheduling.stragglers]
detection_enabled = true
speculative_execution = true
straggler_threshold = 0.5  # Consider straggler if < 50% expected progress
speculation_trigger_ms = 2000

# Pipeline parallelism
[scheduling.pipeline]
enabled = true
max_concurrent_batches = 4
batch_size = 8
prefetch_inputs = true

# Work assignment
[scheduling.assignment]
affinity_enabled = true  # Prefer assigning to nodes that already have model
locality_weight = 0.7  # Balance between locality and load
shuffle_on_failure = true
```

### 2.6 Resource Configuration

```toml
[resources]
# Resource limits and allocation (node-local)

# Memory limits
[resources.memory]
max_total_bytes = 34359738368  # 32GB total limit
max_model_bytes = 26843545600   # 25GB for model weights
max_activation_bytes = 4294967296  # 4GB for activations
max_checkpoint_bytes = 3221225472  # 3GB for checkpoints
oom_reserve_bytes = 536870912      # 512MB reserve for OOM handling

# CPU allocation
[resources.cpu]
worker_threads = 0  # 0 = auto-detect physical cores
max_cpu_percent = 90.0
affinity_enabled = true  # Pin threads to cores
numa_aware = true

# GPU configuration (if available)
[resources.gpu]
enabled = false
device_ids = [0, 1]  # Which GPUs to use
memory_fraction = 0.9  # Use 90% of GPU memory
tensor_cores = true
mixed_precision = true

# Disk I/O
[resources.disk]
max_read_mbps = 1000
max_write_mbps = 500
cache_size_bytes = 10737418240  # 10GB disk cache
prefetch_enabled = true

# Rate limiting
[resources.rate_limits]
max_requests_per_sec = 100
max_tokens_per_sec = 10000
burst_multiplier = 2.0
```

### 2.7 Observability Configuration

```toml
[observability]
# Metrics, logging, and distributed tracing

# Logging configuration
[observability.logging]
level = "info"  # trace, debug, info, warn, error
format = "json"  # json, text, pretty
output = "file"  # stdout, stderr, file, syslog
file_path = "/var/log/butterfly/butterfly.log"
rotation = "daily"  # daily, size, never
max_file_size_mb = 100
max_files = 30
include_source_location = true

# Log filtering
[observability.logging.filters]
# Override log level for specific modules
"butterfly_comm" = "debug"
"butterfly_coordination" = "trace"

# Metrics collection
[observability.metrics]
enabled = true
exporter = "prometheus"  # prometheus, influxdb, custom
export_interval_secs = 15
listen_addr = "0.0.0.0:9090"

# Histogram buckets for latency metrics (milliseconds)
[observability.metrics.histograms]
request_latency_ms = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]

# Specific metrics to collect
[observability.metrics.collection]
system_resources = true
network_io = true
inference_latency = true
queue_depths = true
byzantine_events = true

# Distributed tracing
[observability.tracing]
enabled = true
exporter = "jaeger"  # jaeger, zipkin, otlp
endpoint = "http://localhost:14268/api/traces"
sampling_rate = 0.1  # Sample 10% of requests
include_baggage = true

# Profiling (development/debugging)
[observability.profiling]
enabled = false
cpu_profiling = false
memory_profiling = false
flamegraph_output = "/tmp/butterfly-profile.svg"
```

### 2.8 Security Configuration

```toml
[security]
# Authentication, encryption, and access control

# TLS/mTLS configuration
[security.tls]
enabled = true
cert_file = "/etc/butterfly/certs/node.crt"
key_file = "/etc/butterfly/certs/node.key"
ca_file = "/etc/butterfly/certs/ca.crt"
verify_client = true  # Require client certificates (mTLS)
min_tls_version = "1.3"

# Cryptographic signing
[security.signing]
algorithm = "ed25519"  # ed25519, ecdsa_p256
private_key_file = "/etc/butterfly/keys/signing.key"
public_key_file = "/etc/butterfly/keys/signing.pub"

# Access control
[security.access_control]
enabled = true
policy = "rbac"  # rbac, acl, none
policy_file = "/etc/butterfly/policies/rbac.toml"

# API authentication
[security.api]
require_auth = true
auth_method = "jwt"  # jwt, api_key, mtls
jwt_secret_file = "/etc/butterfly/secrets/jwt.key"
token_expiry_secs = 3600

# Rate limiting for security
[security.rate_limiting]
enabled = true
max_failed_auth_attempts = 5
lockout_duration_secs = 300
```

---

## 3. Configuration Categories and Mutability

### 3.1 Static vs Dynamic Configuration

Configuration parameters are classified by their mutability:

| Category | Can Change at Runtime? | Requires Restart? | Requires Consensus? |
|----------|------------------------|-------------------|---------------------|
| **Static-Critical** | ❌ No | ✅ Yes | N/A |
| **Static-Local** | ❌ No | ✅ Yes | N/A |
| **Dynamic-Consensus** | ✅ Yes | ❌ No | ✅ Yes |
| **Dynamic-Local** | ✅ Yes | ❌ No | ❌ No |

**Static-Critical** (Node restart required, must be identical cluster-wide):
- `byzantine.f_value`: Changes cluster size requirements
- `network.coordinator_addr`: All nodes must agree on coordinator
- `metadata.config_version`: Schema version compatibility
- `security.tls.enabled`: Affects connection establishment

**Static-Local** (Node restart required, node-specific):
- `node.node_id`: Identity cannot change while running
- `node.data_dir`: File system paths fixed at startup
- `resources.gpu.device_ids`: GPU allocation requires restart
- `network.listen_addr`: Binding address fixed at startup

**Dynamic-Consensus** (Can change at runtime, requires cluster consensus):
- `partition.strategy`: Affects all nodes' work assignment
- `scheduling.policy`: Coordination algorithm change
- `byzantine.failure_detection.phi_*_threshold`: Failure detection sensitivity
- `network.compression.*`: Communication protocol changes
- `observability.metrics.export_interval_secs`: Cluster-wide metrics timing

**Dynamic-Local** (Can change at runtime, node-local only):
- `observability.logging.level`: Per-node log verbosity
- `resources.rate_limits.*`: Local rate limiting
- `scheduling.queues.max_queue_depth`: Local queue sizing
- `resources.memory.max_activation_bytes`: Local memory management

### 3.2 Validation Rules

Each configuration section has specific validation rules:

**Network Validation:**
- `listen_addr` must be valid IP:port
- `max_message_size_bytes` ≥ 1MB (tensor size constraint)
- `connection_timeout_ms` < `idle_timeout_ms`
- `compression.level` within algorithm-specific range

**Byzantine Validation:**
- `f_value` ≥ 1 (require fault tolerance)
- Cluster must have ≥ `2*f_value + 1` nodes
- `phi_failed_threshold` > `phi_suspect_threshold`
- `checkpoint.interval_tokens` > 0

**Partition Validation:**
- `min_layers_per_partition` ≤ `max_layers_per_partition`
- `balance_tolerance` ∈ [0.0, 1.0]
- Strategy must be one of supported algorithms

**Resources Validation:**
- Sum of memory limits ≤ `max_total_bytes`
- `max_cpu_percent` ∈ [0.0, 100.0]
- `worker_threads` ≥ 0 (0 = auto)
- GPU `device_ids` must exist on system

**Cross-Section Validation:**
- If `resources.gpu.enabled = false`, GPU-related settings ignored
- If `byzantine.reputation.enabled = false`, reputation thresholds unused
- `scheduling.pipeline.max_concurrent_batches` limited by `resources.memory.max_activation_bytes`

---

## 4. Dynamic Reconfiguration Protocol

### 4.1 Hot Reload Mechanism

**Design Goals:**
1. **Safety**: Never leave cluster in inconsistent state
2. **Atomicity**: Configuration changes apply completely or not at all
3. **Consensus**: All nodes agree on configuration version
4. **Rollback**: Failed changes automatically reverted

**Hot Reload State Machine:**

```
STABLE → PROPOSE → PREPARE → COMMIT → STABLE
   ↓                            ↓
   ←────── ROLLBACK ←───────────┘
```

### 4.2 Configuration Update Protocol

**Phase 1: PROPOSE (Coordinator)**
```
1. Coordinator receives configuration update request via API
2. Validate new configuration:
   a. Parse and type-check
   b. Run validation rules
   c. Check compatibility with current state
3. Classify changes:
   a. Identify static vs dynamic parameters
   b. Identify consensus vs local parameters
   c. Reject if static-critical parameters changed
4. Compute configuration diff
5. Broadcast PROPOSE_CONFIG(new_config, diff, version)
```

**Phase 2: PREPARE (All Nodes)**
```
1. Each node receives PROPOSE_CONFIG
2. Validate locally:
   a. Check version compatibility
   b. Validate against local constraints
   c. Simulate applying configuration (dry run)
3. If valid:
   a. Enter PREPARING state
   b. Respond PREPARE_OK(config_hash)
4. If invalid:
   a. Respond PREPARE_REJECT(reason)
   b. Remain in STABLE state
```

**Phase 3: COMMIT (Byzantine Agreement)**
```
1. Coordinator collects PREPARE responses
2. If ≥ 2f+1 PREPARE_OK with matching hashes:
   a. Broadcast COMMIT_CONFIG(config, version)
   b. Enter COMMITTING state
3. If any PREPARE_REJECT or timeout:
   a. Broadcast ROLLBACK_CONFIG
   b. Abort change, remain STABLE
```

**Phase 4: APPLY (All Nodes)**
```
1. Each node receives COMMIT_CONFIG
2. Atomically apply configuration:
   a. Update in-memory config structure
   b. Persist to disk (atomic write)
   c. Notify affected components
3. Enter STABLE state with new version
4. Respond CONFIG_APPLIED(version)
```

**Phase 5: FINALIZE (Coordinator)**
```
1. Coordinator collects CONFIG_APPLIED responses
2. If ≥ 2f+1 confirmations:
   a. Configuration change successful
   b. Log to audit trail
   c. Broadcast CONFIG_FINALIZED
3. If timeout or failures:
   a. Initiate ROLLBACK protocol
   b. Force nodes to revert to previous config
```

### 4.3 Rollback Protocol

**Automatic Rollback Triggers:**
- Any node fails to PREPARE
- Timeout waiting for PREPARE responses
- Configuration hash mismatch during COMMIT
- Application failure during APPLY phase
- Node crashes during configuration change

**Rollback Procedure:**
```
1. Coordinator broadcasts ROLLBACK_CONFIG(previous_version)
2. All nodes in PREPARING/COMMITTING:
   a. Discard proposed configuration
   b. Revert to previous stable config
   c. Restore previous config version
   d. Respond ROLLBACK_COMPLETE
3. Coordinator waits for ≥ 2f+1 ROLLBACK_COMPLETE
4. System returns to STABLE state
5. Log rollback event with reason
```

### 4.4 Partial Failure Handling

**Scenario: Node Unreachable During Update**
```
1. Coordinator detects node N_i non-responsive during PROPOSE
2. If cluster still has ≥ 2f+1 responsive nodes:
   a. Proceed with configuration update
   b. N_i marked as outdated when it recovers
3. When N_i recovers:
   a. Detects configuration version mismatch
   b. Fetches current configuration from peers
   c. Validates and applies configuration
   d. Rejoins cluster with updated config
```

**Scenario: Byzantine Node During Update**
```
1. Node N_b sends PREPARE_OK but with different config_hash
2. Coordinator detects hash mismatch (Byzantine behavior)
3. Initiates Byzantine isolation protocol:
   a. Mark N_b as suspected Byzantine
   b. Exclude N_b from configuration consensus
   c. Continue update with remaining honest nodes
4. N_b isolated until behavior explained/corrected
```

### 4.5 Configuration Versioning

**Version Format:**
```
config_version = "MAJOR.MINOR.PATCH-COMMIT"
Example: "1.2.5-a3f4e9c"

MAJOR: Incompatible schema changes
MINOR: Backward-compatible additions
PATCH: Bug fixes, clarifications
COMMIT: Git commit hash for auditability
```

**Version Compatibility Rules:**
- Nodes with different MAJOR versions cannot join cluster
- MINOR version differences allowed (downgrade to lowest common)
- PATCH differences purely informational
- COMMIT hash stored for audit trail

**Configuration History:**
```
Maintain last N configuration versions:
- Current active version
- Previous N-1 versions for rollback
- Stored in: {data_dir}/config/history/

Each history entry includes:
- Full configuration file
- Timestamp of application
- Node that proposed change
- Consensus signatures from all nodes
```

---

## 5. Configuration Manager Implementation

### 5.1 Crate Structure: `butterfly-config`

**Module Organization:**
```
butterfly-config/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API
│   ├── schema.rs                 # Configuration structs
│   ├── validation.rs             # Validation logic
│   ├── loader.rs                 # File/env loading
│   ├── manager.rs                # Hot reload coordination
│   ├── consensus.rs              # Byzantine-safe updates
│   ├── versioning.rs             # Version management
│   └── watchers.rs               # File change detection
├── tests/
│   ├── integration_tests.rs
│   ├── validation_tests.rs
│   └── hot_reload_tests.rs
└── benches/
    └── config_benchmarks.rs
```

### 5.2 Core Types

**Main Configuration Struct:**
```rust
/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ButterflyConfig {
    pub metadata: MetadataConfig,
    pub node: NodeConfig,
    pub network: NetworkConfig,
    pub byzantine: ByzantineConfig,
    pub partition: PartitionConfig,
    pub scheduling: SchedulingConfig,
    pub resources: ResourcesConfig,
    pub observability: ObservabilityConfig,
    pub security: SecurityConfig,

    /// Configuration version for compatibility checking
    #[serde(skip)]
    pub version: ConfigVersion,

    /// Hash of configuration for consensus
    #[serde(skip)]
    pub hash: ConfigHash,
}

impl ButterflyConfig {
    /// Load configuration from multiple sources with precedence
    pub fn load() -> Result<Self, ConfigError>;

    /// Validate configuration consistency and constraints
    pub fn validate(&self) -> Result<(), ValidationError>;

    /// Compute cryptographic hash of configuration
    pub fn compute_hash(&self) -> ConfigHash;

    /// Check if another config is compatible (can coexist in cluster)
    pub fn is_compatible_with(&self, other: &Self) -> bool;

    /// Merge configuration from higher precedence source
    pub fn merge(&mut self, other: Self, source: ConfigSource);
}
```

**Configuration Manager:**
```rust
/// Manages configuration lifecycle with hot reload support
pub struct ConfigManager {
    /// Current active configuration
    current: Arc<RwLock<ButterflyConfig>>,

    /// Configuration history for rollback
    history: VecDeque<(ConfigVersion, ButterflyConfig)>,

    /// Watchers for file changes
    watchers: Vec<FileWatcher>,

    /// Subscribers notified on config changes
    subscribers: Vec<ConfigSubscriber>,

    /// State machine for configuration updates
    update_state: Arc<Mutex<UpdateState>>,

    /// Consensus protocol for cluster-wide updates
    consensus: Arc<dyn ConfigConsensus>,
}

impl ConfigManager {
    /// Create new configuration manager
    pub fn new(config: ButterflyConfig) -> Self;

    /// Get current configuration (cheap clone via Arc)
    pub fn get_config(&self) -> Arc<ButterflyConfig>;

    /// Subscribe to configuration changes
    pub fn subscribe(&mut self, subscriber: ConfigSubscriber);

    /// Propose configuration change (initiates consensus)
    pub async fn propose_update(&self, new_config: ButterflyConfig)
        -> Result<ConfigVersion, UpdateError>;

    /// Watch configuration file for changes
    pub async fn watch_file(&mut self, path: PathBuf) -> Result<(), WatchError>;

    /// Manually reload configuration from sources
    pub async fn reload(&self) -> Result<(), ReloadError>;

    /// Rollback to previous configuration version
    pub async fn rollback(&self, version: ConfigVersion)
        -> Result<(), RollbackError>;

    /// Get configuration change history
    pub fn get_history(&self) -> Vec<ConfigHistoryEntry>;
}
```

**Configuration Loader:**
```rust
/// Loads configuration from multiple sources
pub struct ConfigLoader {
    /// Default configuration
    defaults: ButterflyConfig,

    /// File paths to load
    file_paths: Vec<PathBuf>,

    /// Environment variable prefix
    env_prefix: String,

    /// Runtime overrides
    overrides: HashMap<String, Value>,
}

impl ConfigLoader {
    /// Load configuration with precedence resolution
    pub fn load(&self) -> Result<ButterflyConfig, LoadError>;

    /// Load from TOML file
    fn load_from_file(&self, path: &Path) -> Result<ButterflyConfig, LoadError>;

    /// Load from environment variables
    fn load_from_env(&self) -> Result<ButterflyConfig, LoadError>;

    /// Apply runtime overrides
    fn apply_overrides(&self, config: &mut ButterflyConfig) -> Result<(), LoadError>;

    /// Resolve includes and merge
    fn resolve_includes(&self, config: &mut ButterflyConfig) -> Result<(), LoadError>;
}
```

**Validation Engine:**
```rust
/// Validates configuration constraints
pub struct ConfigValidator {
    /// Validation rules registry
    rules: Vec<Box<dyn ValidationRule>>,
}

pub trait ValidationRule: Send + Sync {
    /// Validate configuration section
    fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationError>;

    /// Get rule name for error reporting
    fn name(&self) -> &'static str;
}

impl ConfigValidator {
    /// Register validation rule
    pub fn register<R: ValidationRule + 'static>(&mut self, rule: R);

    /// Validate entire configuration
    pub fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationErrors>;

    /// Validate specific section
    pub fn validate_section<T>(&self, section: &T) -> Result<(), ValidationError>
    where
        T: Validate;
}

// Example validation rules
pub struct NetworkValidationRule;
pub struct ByzantineValidationRule;
pub struct ResourceValidationRule;
pub struct CrossSectionValidationRule;
```

**Hot Reload State Machine:**
```rust
/// State machine for configuration updates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateState {
    Stable,
    Proposing { version: ConfigVersion },
    Preparing { version: ConfigVersion },
    Committing { version: ConfigVersion },
    Applying { version: ConfigVersion },
    RollingBack { from_version: ConfigVersion },
}

/// Protocol for Byzantine-safe configuration consensus
#[async_trait]
pub trait ConfigConsensus: Send + Sync {
    /// Propose configuration change
    async fn propose(&self, config: ButterflyConfig)
        -> Result<ConfigVersion, ConsensusError>;

    /// Handle PREPARE phase
    async fn prepare(&self, config: ButterflyConfig, version: ConfigVersion)
        -> Result<ConfigHash, ConsensusError>;

    /// Handle COMMIT phase
    async fn commit(&self, config: ButterflyConfig, version: ConfigVersion)
        -> Result<(), ConsensusError>;

    /// Handle ROLLBACK phase
    async fn rollback(&self, version: ConfigVersion)
        -> Result<(), ConsensusError>;
}
```

### 5.3 Integration with Components

**Component Configuration Access:**
```rust
/// Components access configuration via typed interfaces
pub trait ConfigurableComponent {
    /// Configuration type for this component
    type Config: Clone + Send + Sync;

    /// Apply configuration change
    fn apply_config(&mut self, config: Self::Config) -> Result<(), ApplyError>;

    /// Get current configuration
    fn get_config(&self) -> Self::Config;
}

/// Example: Network component accesses network config
impl ConfigurableComponent for NetworkManager {
    type Config = NetworkConfig;

    fn apply_config(&mut self, config: NetworkConfig) -> Result<(), ApplyError> {
        // Update connection pools, timeouts, etc.
        self.update_connection_limits(config.connection.max_connections)?;
        self.update_buffers(config.buffers)?;
        Ok(())
    }

    fn get_config(&self) -> NetworkConfig {
        self.config.clone()
    }
}
```

**Configuration Change Notifications:**
```rust
/// Subscriber pattern for configuration changes
pub trait ConfigSubscriber: Send + Sync {
    /// Called when configuration changes
    fn on_config_changed(&self, old: &ButterflyConfig, new: &ButterflyConfig);

    /// Called before configuration change (can reject)
    fn before_config_change(&self, proposed: &ButterflyConfig)
        -> Result<(), String> {
        Ok(()) // Default: accept all changes
    }
}

/// Example subscriber
pub struct MetricsConfigSubscriber {
    metrics: Arc<MetricsCollector>,
}

impl ConfigSubscriber for MetricsConfigSubscriber {
    fn on_config_changed(&self, old: &ButterflyConfig, new: &ButterflyConfig) {
        if old.observability.metrics != new.observability.metrics {
            self.metrics.update_config(&new.observability.metrics);
        }
    }
}
```

---

## 6. CLI Integration

### 6.1 Command-Line Interface

**Configuration-Related Commands:**
```bash
# Start with configuration file
butterfly start --config /etc/butterfly/config.toml

# Override specific values via CLI flags
butterfly start --config config.toml \
  --node-id node-007 \
  --listen-addr 0.0.0.0:7890 \
  --log-level debug

# Validate configuration without starting
butterfly config validate --config config.toml

# Show effective configuration (after merging all sources)
butterfly config show --config config.toml

# Show default configuration
butterfly config defaults --output default-config.toml

# Generate configuration template
butterfly config init --output config.toml

# Live configuration updates
butterfly config update --key network.compression.enabled --value false
butterfly config update --file new-config.toml

# Configuration history
butterfly config history --limit 10
butterfly config rollback --version 1.2.4

# Export configuration
butterfly config export --output current-config.toml
```

**CLI Flags Mapping:**
```rust
#[derive(Parser, Debug)]
#[command(name = "butterfly")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Start(StartCmd),
    Config(ConfigCmd),
    // ... other commands
}

#[derive(Args, Debug)]
pub struct StartCmd {
    /// Path to configuration file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Node ID override
    #[arg(long)]
    node_id: Option<String>,

    /// Listen address override
    #[arg(long)]
    listen_addr: Option<SocketAddr>,

    /// Log level override
    #[arg(long, value_enum)]
    log_level: Option<LogLevel>,

    /// Environment variables file
    #[arg(long, value_name = "FILE")]
    env_file: Option<PathBuf>,
}

impl StartCmd {
    /// Build configuration from CLI args
    pub fn build_config(&self) -> Result<ButterflyConfig, ConfigError> {
        let mut loader = ConfigLoader::new();

        // Load from file if specified
        if let Some(config_path) = &self.config {
            loader.add_file(config_path);
        }

        // Apply CLI overrides
        if let Some(node_id) = &self.node_id {
            loader.override_value("node.node_id", node_id);
        }
        if let Some(listen_addr) = &self.listen_addr {
            loader.override_value("network.listen_addr", listen_addr);
        }
        if let Some(log_level) = &self.log_level {
            loader.override_value("observability.logging.level", log_level);
        }

        loader.load()
    }
}
```

### 6.2 Environment Variable Mapping

**Environment Variable Convention:**
```bash
# Format: BUTTERFLY_SECTION_SUBSECTION_KEY=value

# Examples:
export BUTTERFLY_NODE_NODE_ID="node-007"
export BUTTERFLY_NETWORK_LISTEN_ADDR="0.0.0.0:7890"
export BUTTERFLY_BYZANTINE_F_VALUE=2
export BUTTERFLY_OBSERVABILITY_LOGGING_LEVEL="debug"
export BUTTERFLY_RESOURCES_MEMORY_MAX_TOTAL_BYTES=34359738368

# Nested structures use double underscore
export BUTTERFLY_NETWORK_CONNECTION__MAX_CONNECTIONS=1000
export BUTTERFLY_NETWORK_BUFFERS__SEND_BUFFER_BYTES=4194304

# Arrays use comma separation
export BUTTERFLY_RESOURCES_GPU__DEVICE_IDS="0,1,2"
```

**Environment Variable Loader:**
```rust
impl ConfigLoader {
    fn load_from_env(&self) -> Result<ButterflyConfig, LoadError> {
        let mut config = ButterflyConfig::default();

        for (key, value) in env::vars() {
            if !key.starts_with(&self.env_prefix) {
                continue;
            }

            // Parse key path: BUTTERFLY_NODE_NODE_ID -> node.node_id
            let key_path = key
                .trim_start_matches(&self.env_prefix)
                .trim_start_matches('_')
                .replace("__", ".")
                .replace('_', ".")
                .to_lowercase();

            // Set value in configuration
            self.set_value_by_path(&mut config, &key_path, &value)?;
        }

        Ok(config)
    }
}
```

---

## 7. API Integration

### 7.1 Runtime Configuration Updates

**HTTP/gRPC API Endpoints:**
```
POST   /api/v1/config                    # Propose new configuration
GET    /api/v1/config                    # Get current configuration
GET    /api/v1/config/history            # Get configuration history
POST   /api/v1/config/rollback/{version} # Rollback to version
GET    /api/v1/config/validate           # Validate configuration
PATCH  /api/v1/config                    # Partial configuration update
```

**API Request/Response Types:**
```rust
/// API for runtime configuration management
#[async_trait]
pub trait ConfigApi {
    /// Get current configuration
    async fn get_config(&self) -> Result<ButterflyConfig, ApiError>;

    /// Propose configuration update
    async fn update_config(&self, new_config: ButterflyConfig)
        -> Result<UpdateResponse, ApiError>;

    /// Partial configuration update
    async fn patch_config(&self, patch: ConfigPatch)
        -> Result<UpdateResponse, ApiError>;

    /// Validate configuration
    async fn validate_config(&self, config: ButterflyConfig)
        -> Result<ValidationResult, ApiError>;

    /// Get configuration history
    async fn get_history(&self, limit: usize)
        -> Result<Vec<ConfigHistoryEntry>, ApiError>;

    /// Rollback to previous version
    async fn rollback(&self, version: ConfigVersion)
        -> Result<RollbackResponse, ApiError>;
}

#[derive(Serialize, Deserialize)]
pub struct UpdateResponse {
    pub success: bool,
    pub version: ConfigVersion,
    pub applied_at: DateTime<Utc>,
    pub rollback_available: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ConfigPatch {
    /// JSON Patch operations
    pub operations: Vec<PatchOperation>,
}

#[derive(Serialize, Deserialize)]
pub enum PatchOperation {
    Set { path: String, value: Value },
    Delete { path: String },
    Merge { path: String, value: Value },
}
```

**API Authentication & Authorization:**
```rust
/// API requires authentication for configuration changes
pub struct ConfigApiHandler {
    config_manager: Arc<ConfigManager>,
    auth: Arc<dyn AuthProvider>,
    rbac: Arc<RbacPolicy>,
}

impl ConfigApiHandler {
    async fn update_config_endpoint(
        &self,
        req: UpdateConfigRequest,
        auth_token: AuthToken,
    ) -> Result<UpdateResponse, ApiError> {
        // Authenticate request
        let identity = self.auth.verify_token(&auth_token).await?;

        // Authorize action
        self.rbac.check_permission(&identity, Permission::UpdateConfig)?;

        // Validate configuration
        req.config.validate()?;

        // Propose update via consensus
        let version = self.config_manager
            .propose_update(req.config)
            .await?;

        Ok(UpdateResponse {
            success: true,
            version,
            applied_at: Utc::now(),
            rollback_available: true,
        })
    }
}
```

### 7.2 Configuration Observability

**Metrics for Configuration:**
```rust
/// Prometheus metrics for configuration system
pub struct ConfigMetrics {
    /// Total configuration updates attempted
    updates_total: Counter,

    /// Configuration updates by result (success/failure)
    updates_by_result: CounterVec,

    /// Configuration validation failures
    validation_failures: Counter,

    /// Time spent in each update phase
    update_phase_duration: HistogramVec,

    /// Current configuration version
    current_version: Gauge,

    /// Number of configuration rollbacks
    rollbacks_total: Counter,
}
```

**Logging Configuration Changes:**
```rust
/// Audit log for configuration changes
#[derive(Serialize, Deserialize)]
pub struct ConfigAuditEntry {
    pub timestamp: DateTime<Utc>,
    pub version: ConfigVersion,
    pub action: ConfigAction,
    pub initiator: NodeId,
    pub changes: Vec<ConfigChange>,
    pub result: ConfigResult,
}

#[derive(Serialize, Deserialize)]
pub enum ConfigAction {
    Update,
    Rollback,
    Reload,
}

#[derive(Serialize, Deserialize)]
pub struct ConfigChange {
    pub path: String,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
}
```

---

## 8. Safety Guarantees

### 8.1 Type Safety

**Compile-Time Guarantees:**
- All configuration values strongly typed via Rust's type system
- Invalid configurations rejected at parse time, not runtime
- No `stringly-typed` configuration keys in application code
- Exhaustive matching on enums prevents missing cases

**Runtime Validation:**
- Cross-field constraints validated before application
- Resource limits checked against system capabilities
- Network addresses validated for reachability
- File paths checked for existence and permissions

### 8.2 Byzantine Resilience

**Consensus-Based Updates:**
- Configuration changes require Byzantine agreement (2f+1 nodes)
- Malicious nodes cannot force invalid configurations
- Configuration hashes prevent tampering
- Cryptographic signatures authenticate changes

**Divergence Detection:**
- Nodes periodically exchange configuration hashes
- Mismatch detected triggers reconciliation
- Outdated nodes fetch current configuration
- Byzantine nodes isolated if persistent divergence

### 8.3 Atomicity and Consistency

**Atomic Updates:**
- Configuration changes all-or-nothing (no partial updates)
- Rollback guaranteed if any node fails to apply
- In-memory and persistent state updated atomically
- No observable intermediate states

**Consistency Guarantees:**
- All honest nodes converge to identical configuration
- Causal consistency preserved (updates respect causality)
- Linearizability for configuration reads/writes
- No split-brain scenarios even under partitions

---

## 9. Performance Considerations

### 9.1 Configuration Load Time

**Optimization Strategies:**
- Lazy loading of includes (load on-demand)
- Parallel parsing of independent sections
- Caching of parsed configuration
- Memoization of validation results

**Expected Performance:**
- Small config (<10KB): <1ms parse time
- Medium config (<100KB): <10ms parse time
- Large config (<1MB): <100ms parse time
- Validation: <5ms for typical configuration

### 9.2 Hot Reload Overhead

**Consensus Latency:**
- Optimistic case (no failures): 3 RTT (~30ms local network)
- With Byzantine nodes: 5 RTT (~50ms local network)
- Cross-datacenter: Add network latency (50-200ms)

**Minimizing Disruption:**
- Most hot reloads require no service interruption
- Only affected components reinitialized
- Existing requests not cancelled
- Gradual rollout possible for some changes

### 9.3 Memory Overhead

**Configuration Storage:**
- Active config: ~10-50KB in memory
- History (10 versions): ~100-500KB
- Watchers and subscribers: ~10KB per component

**Optimization:**
- Shared Arc pointers avoid deep copies
- History compressed (zstd) for storage
- Periodic garbage collection of old versions

---

## 10. Testing Strategy

### 10.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let toml = r#"
            [node]
            node_id = "test-node"
            role = "compute"
        "#;

        let config: ButterflyConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.node.node_id, "test-node");
    }

    #[test]
    fn test_validation_rules() {
        let mut config = ButterflyConfig::default();
        config.byzantine.f_value = 0; // Invalid

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_precedence() {
        let defaults = ButterflyConfig::default();
        let file_config = load_from_file("test.toml");
        let env_config = load_from_env();

        let final_config = merge_configs(vec![defaults, file_config, env_config]);
        // Verify env overrides file overrides defaults
    }
}
```

### 10.2 Integration Tests

```rust
#[tokio::test]
async fn test_hot_reload_consensus() {
    // Setup cluster with 5 nodes (f=2)
    let cluster = TestCluster::new(5).await;

    // Propose configuration change
    let new_config = modified_config();
    let result = cluster.nodes[0]
        .config_manager
        .propose_update(new_config.clone())
        .await;

    assert!(result.is_ok());

    // Verify all nodes converged to new config
    for node in &cluster.nodes {
        let config = node.config_manager.get_config();
        assert_eq!(config.hash(), new_config.hash());
    }
}

#[tokio::test]
async fn test_byzantine_node_during_update() {
    let cluster = TestCluster::new(5).await;

    // Node 0 behaves Byzantine (sends wrong hash)
    cluster.nodes[0].set_byzantine(true);

    // Propose update
    let result = cluster.nodes[1]
        .config_manager
        .propose_update(new_config())
        .await;

    // Update should succeed despite Byzantine node
    assert!(result.is_ok());

    // Byzantine node should be isolated
    assert!(cluster.nodes[0].is_isolated());
}
```

### 10.3 Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn config_roundtrip(config in arbitrary_config()) {
        // Serialize to TOML and back
        let toml = toml::to_string(&config).unwrap();
        let parsed: ButterflyConfig = toml::from_str(&toml).unwrap();

        // Should be identical
        prop_assert_eq!(config, parsed);
    }

    #[test]
    fn validation_idempotent(config in valid_config()) {
        // Validation should be idempotent
        config.validate().unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn merge_associative(
        c1 in arbitrary_config(),
        c2 in arbitrary_config(),
        c3 in arbitrary_config()
    ) {
        // Merge should be associative
        let result1 = merge(merge(c1.clone(), c2.clone()), c3.clone());
        let result2 = merge(c1.clone(), merge(c2.clone(), c3.clone()));

        prop_assert_eq!(result1, result2);
    }
}
```

---

## 11. Migration and Compatibility

### 11.1 Configuration Schema Versioning

**Backward Compatibility:**
- New fields added as optional with defaults
- Deprecated fields maintained for 2 major versions
- Clear migration guides for breaking changes

**Forward Compatibility:**
- Unknown fields logged as warnings but not rejected
- Allows gradual rollout of new versions
- Downgrade path documented for each version

### 11.2 Configuration Migration Tools

```bash
# Migrate old configuration to new schema
butterfly config migrate --from v1.0 --to v2.0 --input old.toml --output new.toml

# Check compatibility with cluster
butterfly config check-compatibility --config new.toml --cluster-version 1.5.0

# Dry-run migration (show changes without applying)
butterfly config migrate --dry-run --from v1.0 --to v2.0 old.toml
```

---

## 12. Documentation and Examples

### 12.1 Configuration Templates

**Minimal Configuration (Development):**
```toml
[node]
node_id = "dev-node"
role = "compute"

[network]
coordinator_addr = "127.0.0.1:7890"
listen_addr = "127.0.0.1:7891"

[byzantine]
f_value = 1

[observability.logging]
level = "debug"
format = "pretty"
output = "stdout"
```

**Production Configuration (High Availability):**
```toml
[metadata]
config_version = "1.0.0"
environment = "production"
cluster_name = "butterfly-prod"

[node]
node_id = "prod-node-001"
role = "compute"
data_dir = "/var/lib/butterfly"
log_dir = "/var/log/butterfly"

[network]
protocol = "quic"
coordinator_addr = "10.0.1.10:7890"
listen_addr = "0.0.0.0:7890"
external_addr = "10.0.1.11:7890"

[network.compression]
enabled = true
algorithm = "zstd"
level = 6

[byzantine]
f_value = 2  # Tolerates 2 failures with 5 nodes

[security.tls]
enabled = true
cert_file = "/etc/butterfly/certs/node.crt"
key_file = "/etc/butterfly/certs/node.key"
ca_file = "/etc/butterfly/certs/ca.crt"
verify_client = true

[observability.logging]
level = "info"
format = "json"
output = "file"
file_path = "/var/log/butterfly/butterfly.log"
rotation = "daily"
max_files = 30

[observability.metrics]
enabled = true
exporter = "prometheus"
listen_addr = "0.0.0.0:9090"
```

---

## 13. Future Enhancements

### 13.1 Advanced Features (Phase 2)

- **Configuration Profiles**: Named sets of configurations (dev, staging, prod)
- **Conditional Configuration**: Apply settings based on node characteristics
- **Configuration Templating**: Jinja2-style templating for dynamic values
- **Remote Configuration Store**: etcd/Consul integration for centralized config
- **Configuration Encryption**: Encrypt sensitive values at rest
- **Configuration Drift Detection**: Alert on unauthorized changes
- **A/B Testing**: Canary deployments for configuration changes

### 13.2 Operational Tools (Phase 2)

- **Configuration Diff Tool**: Visual diff between configurations
- **Configuration Playground**: Web UI for testing configurations
- **Configuration Recommendations**: AI-powered tuning suggestions
- **Configuration Import/Export**: Terraform/Ansible integration
- **Configuration Compliance**: Policy enforcement (e.g., minimum TLS version)

---

## Appendix A: Configuration Reference

See `config_schema.toml` for complete annotated example with all options.

## Appendix B: Error Messages

Common configuration errors and resolutions:

| Error | Cause | Resolution |
|-------|-------|------------|
| `InvalidByzantineF` | f_value incompatible with cluster size | Ensure cluster has ≥ 2f+1 nodes |
| `ConfigHashMismatch` | Nodes have divergent configurations | Run `butterfly config sync` |
| `ValidationFailed` | Configuration violates constraints | Check validation error details |
| `ConsensusTimeout` | Unable to reach agreement | Check network connectivity |
| `IncompatibleVersion` | Configuration schema version mismatch | Migrate configuration to current version |

## Appendix C: Performance Tuning

Configuration parameters most impactful for performance:

1. **Network Buffers**: Increase for high-bandwidth networks
2. **Compression Level**: Balance CPU vs network (3-6 sweet spot)
3. **Checkpoint Interval**: More frequent = faster recovery, higher overhead
4. **Pipeline Batches**: More batches = better throughput, higher memory

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Authors:** Butterfly Contributors
**Status:** Design Specification (Implementation Pending)
