# butterfly-config Crate Specification

## Overview

The `butterfly-config` crate provides a type-safe, Byzantine-resilient configuration management system for the Butterfly distributed inference system. It handles configuration loading, validation, hot reloading, and consensus-based updates across the cluster.

**Crate Location:** `/crates/butterfly-config/`

**Key Responsibilities:**
- Load configuration from multiple sources (files, environment, CLI, runtime updates)
- Validate configuration constraints and cross-dependencies
- Manage configuration versioning and compatibility
- Coordinate Byzantine-safe configuration updates across cluster
- Notify components of configuration changes
- Maintain configuration history for rollback

---

## Module Structure

```
butterfly-config/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                    # Public API and re-exports
│   ├── schema/
│   │   ├── mod.rs                # Configuration type definitions
│   │   ├── metadata.rs           # MetadataConfig
│   │   ├── node.rs               # NodeConfig
│   │   ├── network.rs            # NetworkConfig and sub-configs
│   │   ├── byzantine.rs          # ByzantineConfig and sub-configs
│   │   ├── partition.rs          # PartitionConfig and sub-configs
│   │   ├── scheduling.rs         # SchedulingConfig and sub-configs
│   │   ├── resources.rs          # ResourcesConfig and sub-configs
│   │   ├── observability.rs      # ObservabilityConfig and sub-configs
│   │   └── security.rs           # SecurityConfig and sub-configs
│   ├── loader/
│   │   ├── mod.rs                # ConfigLoader main implementation
│   │   ├── file.rs               # TOML file loading
│   │   ├── env.rs                # Environment variable loading
│   │   ├── include.rs            # Include directive resolution
│   │   └── merge.rs              # Configuration merging logic
│   ├── validation/
│   │   ├── mod.rs                # Validation framework
│   │   ├── rules.rs              # Validation rule trait and registry
│   │   ├── network_rules.rs      # Network-specific validation
│   │   ├── byzantine_rules.rs    # Byzantine-specific validation
│   │   ├── resource_rules.rs     # Resource-specific validation
│   │   └── cross_section_rules.rs # Cross-section validation
│   ├── manager/
│   │   ├── mod.rs                # ConfigManager implementation
│   │   ├── state_machine.rs      # Hot reload state machine
│   │   ├── subscriber.rs         # Change notification system
│   │   └── history.rs            # Configuration history tracking
│   ├── consensus/
│   │   ├── mod.rs                # ConfigConsensus trait
│   │   ├── protocol.rs           # Byzantine agreement protocol
│   │   ├── messages.rs           # Consensus message types
│   │   └── phases.rs             # PROPOSE/PREPARE/COMMIT phases
│   ├── versioning/
│   │   ├── mod.rs                # Version management
│   │   ├── version.rs            # ConfigVersion type
│   │   ├── compatibility.rs      # Version compatibility checking
│   │   └── migration.rs          # Configuration migration utilities
│   ├── watchers/
│   │   ├── mod.rs                # File watching infrastructure
│   │   ├── file_watcher.rs       # File change detection
│   │   └── debounce.rs           # Event debouncing
│   ├── error.rs                  # Error types
│   └── hash.rs                   # Configuration hashing utilities
├── tests/
│   ├── integration/
│   │   ├── mod.rs
│   │   ├── loading_tests.rs      # Configuration loading tests
│   │   ├── validation_tests.rs   # Validation tests
│   │   ├── hot_reload_tests.rs   # Hot reload tests
│   │   ├── consensus_tests.rs    # Byzantine consensus tests
│   │   └── rollback_tests.rs     # Rollback tests
│   └── fixtures/
│       ├── valid_configs/        # Valid configuration examples
│       ├── invalid_configs/      # Invalid configurations for testing
│       └── migration/            # Migration test cases
├── benches/
│   ├── config_benchmarks.rs      # Loading and validation benchmarks
│   └── consensus_benchmarks.rs   # Consensus protocol benchmarks
└── examples/
    ├── basic_usage.rs            # Basic configuration loading
    ├── hot_reload.rs             # Hot reload example
    └── custom_validation.rs      # Custom validation rules
```

---

## Core Types and Traits

### 1. Configuration Schema Types (`src/schema/`)

#### Root Configuration (`schema/mod.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Root configuration structure for Butterfly distributed inference system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

    /// Configuration version (not serialized, computed at load time)
    #[serde(skip)]
    pub version: ConfigVersion,

    /// Cryptographic hash of configuration (not serialized)
    #[serde(skip)]
    pub hash: ConfigHash,
}

impl ButterflyConfig {
    /// Create configuration with default values
    pub fn default() -> Self;

    /// Load configuration from all sources
    pub fn load() -> Result<Self, ConfigError>;

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<(), ValidationErrors>;

    /// Compute cryptographic hash of this configuration
    pub fn compute_hash(&self) -> ConfigHash;

    /// Check if this config is compatible with another (can coexist in cluster)
    pub fn is_compatible_with(&self, other: &Self) -> bool;

    /// Get configuration value by dot-separated path
    pub fn get_value(&self, path: &str) -> Option<ConfigValue>;

    /// Set configuration value by dot-separated path
    pub fn set_value(&mut self, path: &str, value: ConfigValue) -> Result<(), ConfigError>;

    /// Merge another configuration into this one
    pub fn merge(&mut self, other: Self, source: ConfigSource);

    /// Clone this configuration efficiently (uses Arc internally)
    pub fn cheap_clone(&self) -> Arc<Self>;
}

/// Configuration value variant
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

/// Configuration source for precedence tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigSource {
    Default,
    IncludedFile,
    ConfigFile,
    Environment,
    CommandLine,
    RuntimeUpdate,
}
```

#### Network Configuration (`schema/network.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkConfig {
    pub protocol: NetworkProtocol,
    pub coordinator_addr: SocketAddr,
    pub listen_addr: SocketAddr,
    pub external_addr: SocketAddr,
    pub ipv6_enabled: bool,
    pub interface: Option<String>,
    pub connection: ConnectionConfig,
    pub buffers: BufferConfig,
    pub compression: CompressionConfig,
    pub retry: RetryConfig,
    pub topology: Option<TopologyConfig>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NetworkProtocol {
    Quic,
    Grpc,
    Tcp,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionConfig {
    pub max_connections: usize,
    pub max_connections_per_peer: usize,
    #[serde(with = "duration_ms")]
    pub connection_timeout: Duration,
    #[serde(with = "duration_ms")]
    pub idle_timeout: Duration,
    #[serde(with = "duration_ms")]
    pub keepalive_interval: Duration,
    pub tcp_nodelay: bool,
    pub connection_pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferConfig {
    pub send_buffer_bytes: usize,
    pub recv_buffer_bytes: usize,
    pub max_message_size_bytes: usize,
    pub stream_buffer_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: u8,
    pub min_size_bytes: usize,
    pub compress_tensors: bool,
    pub compress_checkpoints: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Snappy,
}

// ... (RetryConfig, TopologyConfig similarly defined)

// Helper module for Duration serialization as milliseconds
mod duration_ms {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ms = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(ms))
    }
}
```

#### Byzantine Configuration (`schema/byzantine.rs`)

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ByzantineConfig {
    pub f_value: usize,
    pub timeout_multiplier: f64,
    pub failure_detection: FailureDetectionConfig,
    pub agreement: AgreementConfig,
    pub checkpoint: CheckpointConfig,
    pub reputation: ReputationConfig,
    pub view_change: ViewChangeConfig,
}

impl ByzantineConfig {
    /// Calculate minimum cluster size for this f_value
    pub fn min_cluster_size(&self) -> usize {
        2 * self.f_value + 1
    }

    /// Calculate quorum size for consensus
    pub fn quorum_size(&self) -> usize {
        2 * self.f_value + 1
    }

    /// Check if cluster size is sufficient
    pub fn is_cluster_size_valid(&self, cluster_size: usize) -> bool {
        cluster_size >= self.min_cluster_size()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureDetectionConfig {
    #[serde(with = "duration_ms")]
    pub heartbeat_interval: Duration,
    pub phi_suspect_threshold: f64,
    pub phi_failed_threshold: f64,
    pub max_heartbeat_history: usize,
    pub adaptive_heartbeat: bool,
    #[serde(with = "duration_ms")]
    pub min_heartbeat_interval: Duration,
    #[serde(with = "duration_ms")]
    pub max_heartbeat_interval: Duration,
}

// ... (Other sub-configs similarly defined)
```

**Similar patterns for:**
- `schema/partition.rs`: PartitionConfig with strategy-specific sub-configs
- `schema/scheduling.rs`: SchedulingConfig with queue, straggler, pipeline settings
- `schema/resources.rs`: ResourcesConfig with memory, CPU, GPU, disk settings
- `schema/observability.rs`: ObservabilityConfig with logging, metrics, tracing
- `schema/security.rs`: SecurityConfig with TLS, signing, access control

### 2. Configuration Loader (`src/loader/`)

#### Main Loader (`loader/mod.rs`)

```rust
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// Loads configuration from multiple sources with precedence resolution
pub struct ConfigLoader {
    /// Default configuration values
    defaults: ButterflyConfig,

    /// Configuration file paths
    file_paths: Vec<PathBuf>,

    /// Environment variable prefix
    env_prefix: String,

    /// Runtime overrides
    overrides: HashMap<String, ConfigValue>,

    /// Configuration cache
    cache: Option<Arc<ButterflyConfig>>,
}

impl ConfigLoader {
    /// Create new loader with default configuration
    pub fn new() -> Self {
        Self {
            defaults: ButterflyConfig::default(),
            file_paths: Vec::new(),
            env_prefix: "BUTTERFLY".to_string(),
            overrides: HashMap::new(),
            cache: None,
        }
    }

    /// Add configuration file to load
    pub fn add_file<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.file_paths.push(path.as_ref().to_path_buf());
        self.cache = None; // Invalidate cache
        self
    }

    /// Set environment variable prefix
    pub fn set_env_prefix(&mut self, prefix: impl Into<String>) -> &mut Self {
        self.env_prefix = prefix.into();
        self.cache = None;
        self
    }

    /// Add runtime override
    pub fn override_value(
        &mut self,
        path: impl Into<String>,
        value: impl Into<ConfigValue>,
    ) -> &mut Self {
        self.overrides.insert(path.into(), value.into());
        self.cache = None;
        self
    }

    /// Load configuration from all sources
    pub fn load(&mut self) -> Result<ButterflyConfig, ConfigError> {
        // Check cache
        if let Some(cached) = &self.cache {
            return Ok((**cached).clone());
        }

        // Start with defaults
        let mut config = self.defaults.clone();

        // Load from files (in order)
        for path in &self.file_paths {
            let file_config = self.load_from_file(path)?;
            config.merge(file_config, ConfigSource::ConfigFile);
        }

        // Resolve includes
        self.resolve_includes(&mut config)?;

        // Load from environment variables
        let env_config = self.load_from_env()?;
        config.merge(env_config, ConfigSource::Environment);

        // Apply runtime overrides
        self.apply_overrides(&mut config)?;

        // Compute version and hash
        config.version = ConfigVersion::current();
        config.hash = config.compute_hash();

        // Validate
        config.validate()?;

        // Cache result
        self.cache = Some(Arc::new(config.clone()));

        Ok(config)
    }

    /// Load configuration from TOML file
    fn load_from_file(&self, path: &Path) -> Result<ButterflyConfig, ConfigError> {
        // Implementation in loader/file.rs
    }

    /// Load configuration from environment variables
    fn load_from_env(&self) -> Result<ButterflyConfig, ConfigError> {
        // Implementation in loader/env.rs
    }

    /// Resolve include directives recursively
    fn resolve_includes(&self, config: &mut ButterflyConfig) -> Result<(), ConfigError> {
        // Implementation in loader/include.rs
    }

    /// Apply runtime overrides
    fn apply_overrides(&self, config: &mut ButterflyConfig) -> Result<(), ConfigError> {
        for (path, value) in &self.overrides {
            config.set_value(path, value.clone())?;
        }
        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}
```

#### File Loading (`loader/file.rs`)

```rust
use std::fs;
use std::path::Path;

/// Load configuration from TOML file
pub fn load_from_file(path: &Path) -> Result<ButterflyConfig, ConfigError> {
    // Read file
    let contents = fs::read_to_string(path)
        .map_err(|e| ConfigError::FileRead {
            path: path.to_path_buf(),
            source: e,
        })?;

    // Parse TOML
    let config: ButterflyConfig = toml::from_str(&contents)
        .map_err(|e| ConfigError::ParseError {
            path: path.to_path_buf(),
            source: e.into(),
        })?;

    Ok(config)
}

/// Load configuration from multiple files and merge
pub fn load_from_files<P: AsRef<Path>>(
    paths: &[P],
) -> Result<ButterflyConfig, ConfigError> {
    let mut config = ButterflyConfig::default();

    for path in paths {
        let file_config = load_from_file(path.as_ref())?;
        config.merge(file_config, ConfigSource::ConfigFile);
    }

    Ok(config)
}
```

#### Environment Loading (`loader/env.rs`)

```rust
use std::env;

/// Load configuration from environment variables
pub fn load_from_env(prefix: &str) -> Result<ButterflyConfig, ConfigError> {
    let mut config = ButterflyConfig::default();

    // Iterate environment variables
    for (key, value) in env::vars() {
        if !key.starts_with(prefix) {
            continue;
        }

        // Parse key path: BUTTERFLY_NODE_NODE_ID -> node.node_id
        let key_path = parse_env_key(&key, prefix)?;

        // Parse value
        let config_value = parse_env_value(&value)?;

        // Set value in configuration
        config.set_value(&key_path, config_value)?;
    }

    Ok(config)
}

/// Parse environment variable key to configuration path
fn parse_env_key(key: &str, prefix: &str) -> Result<String, ConfigError> {
    // Remove prefix: BUTTERFLY_NODE_NODE_ID -> NODE_NODE_ID
    let key = key.trim_start_matches(prefix).trim_start_matches('_');

    // Convert to lowercase and replace double underscores with dots
    // NODE__NODE_ID -> node.node_id
    let path = key
        .replace("__", ".")
        .replace('_', ".")
        .to_lowercase();

    Ok(path)
}

/// Parse environment variable value
fn parse_env_value(value: &str) -> Result<ConfigValue, ConfigError> {
    // Try parsing as various types
    if let Ok(b) = value.parse::<bool>() {
        return Ok(ConfigValue::Boolean(b));
    }

    if let Ok(i) = value.parse::<i64>() {
        return Ok(ConfigValue::Integer(i));
    }

    if let Ok(f) = value.parse::<f64>() {
        return Ok(ConfigValue::Float(f));
    }

    // Check for array (comma-separated)
    if value.contains(',') {
        let elements: Result<Vec<_>, _> = value
            .split(',')
            .map(|s| parse_env_value(s.trim()))
            .collect();
        return Ok(ConfigValue::Array(elements?));
    }

    // Default to string
    Ok(ConfigValue::String(value.to_string()))
}
```

### 3. Configuration Validation (`src/validation/`)

#### Validation Framework (`validation/mod.rs`)

```rust
/// Validation rule trait
pub trait ValidationRule: Send + Sync {
    /// Validate configuration
    fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationError>;

    /// Get rule name for error reporting
    fn name(&self) -> &'static str;

    /// Get rule description
    fn description(&self) -> &'static str {
        ""
    }

    /// Check if rule is critical (failure prevents startup)
    fn is_critical(&self) -> bool {
        true
    }
}

/// Validation error with context
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub rule_name: String,
    pub message: String,
    pub path: Option<String>,
    pub hint: Option<String>,
}

/// Multiple validation errors
#[derive(Debug, Clone)]
pub struct ValidationErrors {
    pub errors: Vec<ValidationError>,
}

impl ValidationErrors {
    pub fn has_critical_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn format_errors(&self) -> String {
        // Format all errors into human-readable message
    }
}

/// Configuration validator with rule registry
pub struct ConfigValidator {
    rules: Vec<Box<dyn ValidationRule>>,
}

impl ConfigValidator {
    /// Create new validator with default rules
    pub fn new() -> Self {
        let mut validator = Self { rules: Vec::new() };

        // Register default rules
        validator.register(NetworkValidationRule);
        validator.register(ByzantineValidationRule);
        validator.register(PartitionValidationRule);
        validator.register(SchedulingValidationRule);
        validator.register(ResourceValidationRule);
        validator.register(ObservabilityValidationRule);
        validator.register(SecurityValidationRule);
        validator.register(CrossSectionValidationRule);

        validator
    }

    /// Register validation rule
    pub fn register<R: ValidationRule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }

    /// Validate entire configuration
    pub fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationErrors> {
        let mut errors = Vec::new();

        for rule in &self.rules {
            if let Err(e) = rule.validate(config) {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrors { errors })
        }
    }

    /// Validate and collect all errors (don't stop on first error)
    pub fn validate_all(&self, config: &ButterflyConfig) -> ValidationErrors {
        let errors: Vec<_> = self.rules
            .iter()
            .filter_map(|rule| rule.validate(config).err())
            .collect();

        ValidationErrors { errors }
    }
}
```

#### Byzantine Validation (`validation/byzantine_rules.rs`)

```rust
pub struct ByzantineValidationRule;

impl ValidationRule for ByzantineValidationRule {
    fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationError> {
        let byz = &config.byzantine;

        // Validate f_value
        if byz.f_value < 1 {
            return Err(ValidationError {
                rule_name: self.name().to_string(),
                message: "f_value must be at least 1 for fault tolerance".to_string(),
                path: Some("byzantine.f_value".to_string()),
                hint: Some("Set f_value to 1 or higher".to_string()),
            });
        }

        // Validate phi thresholds
        if byz.failure_detection.phi_failed_threshold
            <= byz.failure_detection.phi_suspect_threshold
        {
            return Err(ValidationError {
                rule_name: self.name().to_string(),
                message: "phi_failed_threshold must be greater than phi_suspect_threshold"
                    .to_string(),
                path: Some("byzantine.failure_detection".to_string()),
                hint: Some("Typical values: phi_suspect_threshold=8.0, phi_failed_threshold=12.0"
                    .to_string()),
            });
        }

        // Validate checkpoint interval
        if byz.checkpoint.interval_tokens == 0 {
            return Err(ValidationError {
                rule_name: self.name().to_string(),
                message: "checkpoint interval_tokens must be positive".to_string(),
                path: Some("byzantine.checkpoint.interval_tokens".to_string()),
                hint: Some("Typical value: 10".to_string()),
            });
        }

        // Validate reputation thresholds
        if byz.reputation.enabled {
            if byz.reputation.isolate_threshold >= byz.reputation.suspect_threshold {
                return Err(ValidationError {
                    rule_name: self.name().to_string(),
                    message: "isolate_threshold must be less than suspect_threshold".to_string(),
                    path: Some("byzantine.reputation".to_string()),
                    hint: None,
                });
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Byzantine Configuration Validation"
    }

    fn description(&self) -> &'static str {
        "Validates Byzantine fault tolerance parameters"
    }
}
```

**Similar validation rules for:**
- `network_rules.rs`: Network configuration validation
- `resource_rules.rs`: Resource limit validation
- `cross_section_rules.rs`: Cross-section constraints (e.g., memory limits sum correctly)

### 4. Configuration Manager (`src/manager/`)

#### Configuration Manager (`manager/mod.rs`)

```rust
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};

/// Manages configuration lifecycle with hot reload support
pub struct ConfigManager {
    /// Current active configuration (Arc for cheap clones)
    current: Arc<RwLock<ButterflyConfig>>,

    /// Configuration history for rollback (bounded queue)
    history: Arc<Mutex<VecDeque<(ConfigVersion, ButterflyConfig)>>>,

    /// Maximum history size
    max_history: usize,

    /// File watchers for configuration changes
    watchers: Arc<Mutex<Vec<FileWatcher>>>,

    /// Subscribers notified on config changes
    subscribers: Arc<Mutex<Vec<Arc<dyn ConfigSubscriber>>>>,

    /// State machine for configuration updates
    update_state: Arc<Mutex<UpdateState>>,

    /// Consensus protocol for cluster-wide updates
    consensus: Arc<dyn ConfigConsensus>,

    /// Configuration validator
    validator: Arc<ConfigValidator>,
}

impl ConfigManager {
    /// Create new configuration manager
    pub fn new(
        config: ButterflyConfig,
        consensus: Arc<dyn ConfigConsensus>,
    ) -> Self {
        Self {
            current: Arc::new(RwLock::new(config)),
            history: Arc::new(Mutex::new(VecDeque::new())),
            max_history: 100,
            watchers: Arc::new(Mutex::new(Vec::new())),
            subscribers: Arc::new(Mutex::new(Vec::new())),
            update_state: Arc::new(Mutex::new(UpdateState::Stable)),
            consensus,
            validator: Arc::new(ConfigValidator::new()),
        }
    }

    /// Get current configuration (cheap Arc clone)
    pub async fn get_config(&self) -> Arc<ButterflyConfig> {
        let config = self.current.read().await;
        Arc::new((*config).clone())
    }

    /// Subscribe to configuration changes
    pub async fn subscribe(&self, subscriber: Arc<dyn ConfigSubscriber>) {
        let mut subs = self.subscribers.lock().await;
        subs.push(subscriber);
    }

    /// Propose configuration update (initiates consensus)
    pub async fn propose_update(
        &self,
        new_config: ButterflyConfig,
    ) -> Result<ConfigVersion, UpdateError> {
        // Check current state
        let mut state = self.update_state.lock().await;
        if !matches!(*state, UpdateState::Stable) {
            return Err(UpdateError::UpdateInProgress);
        }

        // Validate configuration
        self.validator.validate(&new_config)?;

        // Check compatibility with current config
        let current = self.current.read().await;
        if !new_config.is_compatible_with(&*current) {
            return Err(UpdateError::IncompatibleConfig);
        }
        drop(current);

        // Notify subscribers (before_config_change)
        self.notify_before_change(&new_config).await?;

        // Enter PROPOSING state
        let version = ConfigVersion::next();
        *state = UpdateState::Proposing { version };
        drop(state);

        // Initiate Byzantine consensus
        let result = self.consensus.propose(new_config.clone()).await;

        match result {
            Ok(version) => {
                // Success: configuration applied
                self.finalize_update(new_config, version).await?;
                Ok(version)
            }
            Err(e) => {
                // Failure: rollback
                self.rollback_update().await?;
                Err(UpdateError::ConsensusFailed(e))
            }
        }
    }

    /// Watch configuration file for changes
    pub async fn watch_file(&self, path: PathBuf) -> Result<(), WatchError> {
        let watcher = FileWatcher::new(path)?;

        // Setup callback for file changes
        let manager = Arc::new(self.clone());
        watcher.on_change(move || {
            let manager = manager.clone();
            tokio::spawn(async move {
                if let Err(e) = manager.reload().await {
                    tracing::error!("Failed to reload configuration: {}", e);
                }
            });
        });

        let mut watchers = self.watchers.lock().await;
        watchers.push(watcher);

        Ok(())
    }

    /// Manually reload configuration from sources
    pub async fn reload(&self) -> Result<(), ReloadError> {
        let mut loader = ConfigLoader::new();
        // ... setup loader with original sources
        let new_config = loader.load()?;

        self.propose_update(new_config).await?;

        Ok(())
    }

    /// Rollback to previous configuration version
    pub async fn rollback(
        &self,
        version: ConfigVersion,
    ) -> Result<(), RollbackError> {
        // Find configuration in history
        let history = self.history.lock().await;
        let config = history
            .iter()
            .find(|(v, _)| *v == version)
            .map(|(_, c)| c.clone())
            .ok_or(RollbackError::VersionNotFound(version))?;
        drop(history);

        // Propose rollback (goes through consensus)
        self.propose_update(config).await?;

        Ok(())
    }

    /// Get configuration change history
    pub async fn get_history(&self) -> Vec<ConfigHistoryEntry> {
        let history = self.history.lock().await;
        history
            .iter()
            .map(|(version, config)| ConfigHistoryEntry {
                version: *version,
                timestamp: version.timestamp(),
                hash: config.hash,
            })
            .collect()
    }

    /// Finalize configuration update
    async fn finalize_update(
        &self,
        new_config: ButterflyConfig,
        version: ConfigVersion,
    ) -> Result<(), UpdateError> {
        // Store old config in history
        let old_config = {
            let config = self.current.read().await;
            (*config).clone()
        };

        {
            let mut history = self.history.lock().await;
            history.push_front((old_config.version, old_config.clone()));

            // Trim history if needed
            while history.len() > self.max_history {
                history.pop_back();
            }
        }

        // Update current config
        {
            let mut current = self.current.write().await;
            *current = new_config.clone();
        }

        // Notify subscribers
        self.notify_after_change(&old_config, &new_config).await;

        // Return to stable state
        let mut state = self.update_state.lock().await;
        *state = UpdateState::Stable;

        Ok(())
    }

    /// Rollback update in progress
    async fn rollback_update(&self) -> Result<(), UpdateError> {
        let mut state = self.update_state.lock().await;
        *state = UpdateState::Stable;
        Ok(())
    }

    /// Notify subscribers before config change
    async fn notify_before_change(
        &self,
        proposed: &ButterflyConfig,
    ) -> Result<(), UpdateError> {
        let subscribers = self.subscribers.lock().await;

        for subscriber in subscribers.iter() {
            subscriber.before_config_change(proposed)?;
        }

        Ok(())
    }

    /// Notify subscribers after config change
    async fn notify_after_change(
        &self,
        old: &ButterflyConfig,
        new: &ButterflyConfig,
    ) {
        let subscribers = self.subscribers.lock().await;

        for subscriber in subscribers.iter() {
            subscriber.on_config_changed(old, new);
        }
    }
}
```

#### Configuration Subscriber (`manager/subscriber.rs`)

```rust
/// Trait for components that need to be notified of configuration changes
pub trait ConfigSubscriber: Send + Sync {
    /// Called after configuration successfully changed
    fn on_config_changed(&self, old: &ButterflyConfig, new: &ButterflyConfig);

    /// Called before configuration change (can reject)
    fn before_config_change(
        &self,
        proposed: &ButterflyConfig,
    ) -> Result<(), String> {
        Ok(()) // Default: accept all changes
    }

    /// Get subscriber name for logging
    fn name(&self) -> &str;
}

/// Example: Network component subscriber
pub struct NetworkConfigSubscriber {
    network_manager: Arc<NetworkManager>,
}

impl ConfigSubscriber for NetworkConfigSubscriber {
    fn on_config_changed(&self, old: &ButterflyConfig, new: &ButterflyConfig) {
        if old.network != new.network {
            tracing::info!("Network configuration changed, updating...");

            if let Err(e) = self.network_manager.update_config(&new.network) {
                tracing::error!("Failed to apply network config: {}", e);
            }
        }
    }

    fn before_config_change(&self, proposed: &ButterflyConfig) -> Result<(), String> {
        // Validate that network changes are safe
        if proposed.network.listen_addr != self.current_listen_addr() {
            return Err("Cannot change listen_addr at runtime".to_string());
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "NetworkConfigSubscriber"
    }
}
```

### 5. Configuration Consensus (`src/consensus/`)

#### Consensus Trait (`consensus/mod.rs`)

```rust
use async_trait::async_trait;

/// Byzantine-safe configuration consensus protocol
#[async_trait]
pub trait ConfigConsensus: Send + Sync {
    /// Propose configuration change to cluster
    async fn propose(
        &self,
        config: ButterflyConfig,
    ) -> Result<ConfigVersion, ConsensusError>;

    /// Handle PREPARE phase of consensus
    async fn prepare(
        &self,
        config: ButterflyConfig,
        version: ConfigVersion,
    ) -> Result<ConfigHash, ConsensusError>;

    /// Handle COMMIT phase of consensus
    async fn commit(
        &self,
        config: ButterflyConfig,
        version: ConfigVersion,
    ) -> Result<(), ConsensusError>;

    /// Handle ROLLBACK phase if consensus fails
    async fn rollback(&self, version: ConfigVersion) -> Result<(), ConsensusError>;

    /// Get current consensus state
    async fn state(&self) -> ConsensusState;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusState {
    Idle,
    Proposing,
    Preparing,
    Committing,
    RollingBack,
}

#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Timeout waiting for quorum")]
    Timeout,

    #[error("Configuration hash mismatch")]
    HashMismatch,

    #[error("Insufficient nodes for consensus")]
    InsufficientNodes,

    #[error("Byzantine behavior detected from node {0}")]
    ByzantineNode(String),

    #[error("Consensus rejected: {0}")]
    Rejected(String),
}
```

#### Consensus Protocol Implementation (`consensus/protocol.rs`)

```rust
use std::collections::HashMap;
use tokio::time::{timeout, Duration};

/// Byzantine consensus implementation for configuration updates
pub struct ConfigConsensusProtocol {
    /// Node ID
    node_id: String,

    /// Communication channel to other nodes
    comm: Arc<dyn CommunicationChannel>,

    /// Byzantine configuration
    byzantine_config: ByzantineConfig,

    /// Current consensus state
    state: Arc<Mutex<ConsensusState>>,

    /// Pending proposals
    proposals: Arc<Mutex<HashMap<ConfigVersion, ProposalState>>>,
}

struct ProposalState {
    config: ButterflyConfig,
    prepare_votes: HashMap<String, ConfigHash>,
    commit_votes: HashSet<String>,
    timestamp: Instant,
}

#[async_trait]
impl ConfigConsensus for ConfigConsensusProtocol {
    async fn propose(
        &self,
        config: ButterflyConfig,
    ) -> Result<ConfigVersion, ConsensusError> {
        let version = ConfigVersion::next();
        let config_hash = config.compute_hash();

        // Phase 1: PRE-PREPARE (broadcast to all nodes)
        tracing::info!("Proposing config update version {}", version);

        let propose_msg = ConsensusMessage::PrePrepare {
            version,
            config: config.clone(),
            hash: config_hash,
        };

        self.comm.broadcast(propose_msg).await?;

        // Phase 2: PREPARE (collect votes)
        let prepare_timeout = Duration::from_millis(
            self.byzantine_config.agreement.prepare_timeout_ms
        );

        let prepare_votes = timeout(
            prepare_timeout,
            self.collect_prepare_votes(version, config_hash)
        )
        .await
        .map_err(|_| ConsensusError::Timeout)??;

        // Check if we have quorum with matching hashes
        let quorum_size = self.byzantine_config.quorum_size();
        if prepare_votes.len() < quorum_size {
            return Err(ConsensusError::InsufficientNodes);
        }

        // Verify all hashes match
        for (node_id, hash) in &prepare_votes {
            if *hash != config_hash {
                return Err(ConsensusError::ByzantineNode(node_id.clone()));
            }
        }

        // Phase 3: COMMIT (broadcast commit decision)
        let commit_msg = ConsensusMessage::Commit { version };
        self.comm.broadcast(commit_msg).await?;

        // Wait for commit confirmations
        let commit_timeout = Duration::from_millis(
            self.byzantine_config.agreement.commit_timeout_ms
        );

        timeout(commit_timeout, self.wait_for_commits(version, quorum_size))
            .await
            .map_err(|_| ConsensusError::Timeout)??;

        tracing::info!("Config update {} committed successfully", version);

        Ok(version)
    }

    async fn prepare(
        &self,
        config: ButterflyConfig,
        version: ConfigVersion,
    ) -> Result<ConfigHash, ConsensusError> {
        // Validate configuration locally
        config.validate()
            .map_err(|e| ConsensusError::Rejected(format!("{:?}", e)))?;

        // Compute hash
        let hash = config.compute_hash();

        // Send PREPARE message
        let prepare_msg = ConsensusMessage::Prepare {
            version,
            node_id: self.node_id.clone(),
            hash,
        };

        self.comm.broadcast(prepare_msg).await?;

        Ok(hash)
    }

    async fn commit(
        &self,
        config: ButterflyConfig,
        version: ConfigVersion,
    ) -> Result<(), ConsensusError> {
        // Apply configuration locally
        // (This is called by ConfigManager after consensus completes)
        Ok(())
    }

    async fn rollback(&self, version: ConfigVersion) -> Result<(), ConsensusError> {
        // Broadcast rollback message
        let rollback_msg = ConsensusMessage::Rollback { version };
        self.comm.broadcast(rollback_msg).await?;

        // Clean up proposal state
        let mut proposals = self.proposals.lock().await;
        proposals.remove(&version);

        Ok(())
    }

    async fn state(&self) -> ConsensusState {
        *self.state.lock().await
    }
}

impl ConfigConsensusProtocol {
    async fn collect_prepare_votes(
        &self,
        version: ConfigVersion,
        expected_hash: ConfigHash,
    ) -> Result<HashMap<String, ConfigHash>, ConsensusError> {
        // Wait for PREPARE messages from nodes
        // Implementation details...
        todo!()
    }

    async fn wait_for_commits(
        &self,
        version: ConfigVersion,
        quorum_size: usize,
    ) -> Result<(), ConsensusError> {
        // Wait for COMMIT confirmations
        // Implementation details...
        todo!()
    }
}
```

### 6. Error Types (`src/error.rs`)

```rust
use thiserror::Error;
use std::path::PathBuf;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read configuration file {path}: {source}")]
    FileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse configuration file {path}: {source}")]
    ParseError {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Configuration validation failed: {0}")]
    ValidationFailed(#[from] ValidationErrors),

    #[error("Invalid configuration path: {0}")]
    InvalidPath(String),

    #[error("Type mismatch at {path}: expected {expected}, got {actual}")]
    TypeMismatch {
        path: String,
        expected: String,
        actual: String,
    },

    #[error("Configuration update error: {0}")]
    UpdateError(#[from] UpdateError),

    #[error("Configuration reload error: {0}")]
    ReloadError(#[from] ReloadError),

    #[error("Configuration rollback error: {0}")]
    RollbackError(#[from] RollbackError),
}

#[derive(Debug, Error)]
pub enum UpdateError {
    #[error("Configuration update already in progress")]
    UpdateInProgress,

    #[error("Proposed configuration incompatible with current")]
    IncompatibleConfig,

    #[error("Configuration validation failed: {0}")]
    ValidationFailed(#[from] ValidationErrors),

    #[error("Consensus failed: {0}")]
    ConsensusFailed(#[from] ConsensusError),

    #[error("Subscriber rejected configuration: {0}")]
    SubscriberRejected(String),
}

#[derive(Debug, Error)]
pub enum ReloadError {
    #[error("Failed to load configuration: {0}")]
    LoadFailed(#[from] ConfigError),

    #[error("Failed to propose update: {0}")]
    ProposeFailed(#[from] UpdateError),
}

#[derive(Debug, Error)]
pub enum RollbackError {
    #[error("Configuration version {0} not found in history")]
    VersionNotFound(ConfigVersion),

    #[error("Failed to propose rollback: {0}")]
    ProposeFailed(#[from] UpdateError),
}

#[derive(Debug, Error)]
pub enum WatchError {
    #[error("Failed to create file watcher: {0}")]
    CreateFailed(#[source] std::io::Error),

    #[error("File watch error: {0}")]
    WatchFailed(String),
}
```

---

## Dependencies (`Cargo.toml`)

```toml
[package]
name = "butterfly-config"
version = "0.1.0"
edition = "2021"
authors = ["Butterfly Contributors"]
description = "Configuration management for Butterfly distributed inference"
license = "MIT OR Apache-2.0"

[dependencies]
# Serialization
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"

# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Cryptography (for hashing)
blake3 = "1.5"
ed25519-dalek = { version = "2.1", features = ["serde"] }

# Utilities
chrono = { version = "0.4", features = ["serde"] }
parking_lot = "0.12"

# File watching
notify = "6.1"

# Logging
tracing = "0.1"

# Internal dependencies
butterfly-comm = { path = "../butterfly-comm" }
butterfly-coordination = { path = "../butterfly-coordination" }

[dev-dependencies]
# Testing
tokio-test = "0.4"
tempfile = "3.8"
proptest = "1.4"
mockall = "0.12"

# Benchmarking
criterion = "0.5"

[[bench]]
name = "config_benchmarks"
harness = false

[[bench]]
name = "consensus_benchmarks"
harness = false
```

---

## Testing Strategy

### Unit Tests

- Configuration parsing from TOML
- Environment variable loading and mapping
- Configuration merging and precedence
- Validation rule execution
- Configuration hashing and versioning

### Integration Tests

- Full configuration lifecycle (load → validate → apply)
- Hot reload with file watching
- Byzantine consensus protocol
- Rollback functionality
- Multi-node configuration synchronization

### Property-Based Tests

- Configuration serialization round-trips
- Merging is associative and commutative (where applicable)
- Validation is idempotent
- Hash consistency across identical configs

### Chaos Tests

- Random node failures during consensus
- Network partitions during configuration updates
- Conflicting concurrent updates
- Byzantine node behavior injection

---

## Performance Targets

- Configuration loading: <10ms for typical config (<100KB)
- Validation: <5ms for full configuration
- Hot reload consensus: <100ms (local network, no failures)
- Memory overhead: <1MB per ConfigManager instance
- Configuration hash computation: <1ms

---

## Usage Examples

### Basic Configuration Loading

```rust
use butterfly_config::{ConfigLoader, ButterflyConfig};

// Load configuration from file
let mut loader = ConfigLoader::new();
loader.add_file("config.toml");

let config = loader.load()?;

// Validate
config.validate()?;

println!("Node ID: {}", config.node.node_id);
println!("Byzantine f_value: {}", config.byzantine.f_value);
```

### Configuration Manager with Hot Reload

```rust
use butterfly_config::{ConfigManager, ButterflyConfig};
use std::sync::Arc;

// Create configuration manager
let config = ButterflyConfig::load()?;
let consensus = Arc::new(MyConsensusImpl::new());
let manager = ConfigManager::new(config, consensus);

// Subscribe to changes
manager.subscribe(Arc::new(MyConfigSubscriber::new())).await;

// Watch configuration file
manager.watch_file("config.toml".into()).await?;

// Get current config (cheap)
let config = manager.get_config().await;
println!("Current version: {}", config.version);

// Propose update
let new_config = load_new_config()?;
let version = manager.propose_update(new_config).await?;
println!("Updated to version {}", version);
```

### Custom Validation Rule

```rust
use butterfly_config::{ValidationRule, ValidationError, ButterflyConfig};

struct CustomValidationRule;

impl ValidationRule for CustomValidationRule {
    fn validate(&self, config: &ButterflyConfig) -> Result<(), ValidationError> {
        // Custom validation logic
        if config.network.compression.enabled
            && config.network.compression.algorithm == CompressionAlgorithm::None
        {
            return Err(ValidationError {
                rule_name: self.name().to_string(),
                message: "Compression enabled but algorithm is 'none'".to_string(),
                path: Some("network.compression".to_string()),
                hint: Some("Set algorithm to 'zstd' or 'lz4'".to_string()),
            });
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Custom Compression Validation"
    }
}

// Register custom rule
let mut validator = ConfigValidator::new();
validator.register(CustomValidationRule);
```

---

## Implementation Checklist

- [ ] Core schema types (schema/*.rs)
- [ ] Configuration loader (loader/*.rs)
- [ ] Validation framework (validation/*.rs)
- [ ] Configuration manager (manager/*.rs)
- [ ] Byzantine consensus protocol (consensus/*.rs)
- [ ] File watching (watchers/*.rs)
- [ ] Versioning utilities (versioning/*.rs)
- [ ] Error types (error.rs)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Property-based tests
- [ ] Benchmarks
- [ ] Documentation
- [ ] Examples

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Status:** Design Specification (Implementation Pending)
