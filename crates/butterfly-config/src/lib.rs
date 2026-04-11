//! # Butterfly Configuration System
//!
//! Provides TOML-based configuration with validation and environment variable overrides.
//!
//! ## Configuration Hierarchy
//!
//! 1. Hardcoded defaults
//! 2. TOML configuration file
//! 3. Environment variables (`BUTTERFLY_*` prefix)
//!
//! ## Example Configuration
//!
//! ```toml
//! [metadata]
//! version = "0.1.0"
//! environment = "production"
//!
//! [network]
//! protocol = "tcp"
//! bind_address = "0.0.0.0:8080"
//! max_connections = 1000
//!
//! [byzantine]
//! fault_tolerance_level = 1
//! checkpoint_interval_secs = 60
//!
//! [partition]
//! strategy = "load-balanced"
//!
//! [scheduling]
//! policy = "fifo"
//! queue_size = 10000
//! ```

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadError(String),

    #[error("Failed to parse TOML: {0}")]
    ParseError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

/// Complete Butterfly configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Metadata about this configuration
    #[serde(default)]
    pub metadata: MetadataConfig,

    /// Network communication settings
    #[serde(default)]
    pub network: NetworkConfig,

    /// Byzantine fault tolerance settings
    #[serde(default)]
    pub byzantine: ByzantineConfig,

    /// Model partitioning settings
    #[serde(default)]
    pub partition: PartitionConfig,

    /// Task scheduling settings
    #[serde(default)]
    pub scheduling: SchedulingConfig,

    /// Resource limits
    #[serde(default)]
    pub resources: ResourceConfig,

    /// Observability settings
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ReadError(e.to_string()))?;

        let config: Self = toml::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration with environment variable overrides
    pub fn load_with_overrides(base_path: &Path) -> Result<Self, ConfigError> {
        let mut config = Self::load_from_file(base_path)?;

        // Apply environment variable overrides
        if let Ok(val) = std::env::var("BUTTERFLY_BIND_ADDRESS") {
            config.network.bind_address = val;
        }

        if let Ok(val) = std::env::var("BUTTERFLY_FAULT_TOLERANCE") {
            config.byzantine.fault_tolerance_level = val
                .parse()
                .map_err(|_| ConfigError::InvalidValue("Invalid fault tolerance level".to_string()))?;
        }

        if let Ok(val) = std::env::var("BUTTERFLY_LOG_LEVEL") {
            config.observability.log_level = val;
        }

        config.validate()?;
        Ok(config)
    }

    /// Create default configuration
    pub fn default() -> Self {
        Self {
            metadata: MetadataConfig::default(),
            network: NetworkConfig::default(),
            byzantine: ByzantineConfig::default(),
            partition: PartitionConfig::default(),
            scheduling: SchedulingConfig::default(),
            resources: ResourceConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.network.validate()?;
        self.byzantine.validate()?;
        self.partition.validate()?;
        self.scheduling.validate()?;
        self.resources.validate()?;
        self.observability.validate()?;
        Ok(())
    }
}

/// Metadata configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfig {
    pub version: String,
    pub environment: Environment,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            version: "0.1.0".to_string(),
            environment: Environment::Development,
        }
    }
}

/// Deployment environment
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Development,
    Staging,
    Production,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Protocol to use (tcp or quic)
    pub protocol: NetworkProtocol,

    /// Address to bind to (e.g., "0.0.0.0:8080")
    pub bind_address: String,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Compression strategy
    pub compression: CompressionMode,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            protocol: NetworkProtocol::Tcp,
            bind_address: "0.0.0.0:8080".to_string(),
            max_connections: 1000,
            compression: CompressionMode::Adaptive,
            connection_timeout_secs: 30,
        }
    }
}

impl NetworkConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate bind address
        self.bind_address
            .parse::<SocketAddr>()
            .map_err(|_| ConfigError::ValidationError(format!("Invalid bind address: {}", self.bind_address)))?;

        if self.max_connections == 0 {
            return Err(ConfigError::ValidationError(
                "max_connections must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Parse bind address into SocketAddr
    pub fn bind_socket_addr(&self) -> SocketAddr {
        self.bind_address
            .parse()
            .expect("Validated during config load")
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.connection_timeout_secs)
    }
}

/// Network protocol
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NetworkProtocol {
    Tcp,
    Quic,
}

/// Compression mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CompressionMode {
    None,
    Lz4,
    Zstd,
    Adaptive,
}

/// Byzantine fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineConfig {
    /// Number of Byzantine faults to tolerate (f)
    /// Requires 2f+1 total nodes
    pub fault_tolerance_level: usize,

    /// Checkpoint interval in seconds
    pub checkpoint_interval_secs: u64,

    /// Reputation decay factor (0.0-1.0)
    pub reputation_decay: f64,

    /// Maximum checkpoints to retain
    pub max_checkpoints: usize,
}

impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            fault_tolerance_level: 1,
            checkpoint_interval_secs: 60,
            reputation_decay: 0.95,
            max_checkpoints: 10,
        }
    }
}

impl ByzantineConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.reputation_decay < 0.0 || self.reputation_decay > 1.0 {
            return Err(ConfigError::ValidationError(
                "reputation_decay must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.max_checkpoints == 0 {
            return Err(ConfigError::ValidationError(
                "max_checkpoints must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate required quorum size for the given number of nodes
    pub fn quorum_size(&self, total_nodes: usize) -> Result<usize, ConfigError> {
        let required = 2 * self.fault_tolerance_level + 1;

        if total_nodes < required {
            return Err(ConfigError::ValidationError(format!(
                "Need at least {} nodes for fault_tolerance_level={} (have {})",
                required, self.fault_tolerance_level, total_nodes
            )));
        }

        Ok(required)
    }

    /// Get checkpoint interval as Duration
    pub fn checkpoint_interval(&self) -> Duration {
        Duration::from_secs(self.checkpoint_interval_secs)
    }
}

/// Partition strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Partitioning strategy
    pub strategy: PartitionStrategy,

    /// Rebalance interval in seconds (0 = disabled)
    pub rebalance_interval_secs: u64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            strategy: PartitionStrategy::LoadBalanced,
            rebalance_interval_secs: 300,
        }
    }
}

impl PartitionConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }

    /// Get rebalance interval as Option<Duration>
    pub fn rebalance_interval(&self) -> Option<Duration> {
        if self.rebalance_interval_secs > 0 {
            Some(Duration::from_secs(self.rebalance_interval_secs))
        } else {
            None
        }
    }
}

/// Partitioning strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PartitionStrategy {
    Uniform,
    LoadBalanced,
    TopologyAware,
    TensorParallel,
    PipelineParallel,
}

/// Scheduling policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Scheduling policy
    pub policy: SchedulingPolicy,

    /// Maximum queue size
    pub queue_size: usize,

    /// Enable batch processing
    pub enable_batching: bool,

    /// Maximum batch size
    pub max_batch_size: usize,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fifo,
            queue_size: 10000,
            enable_batching: false,
            max_batch_size: 16,
        }
    }
}

impl SchedulingConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.queue_size == 0 {
            return Err(ConfigError::ValidationError(
                "queue_size must be > 0".to_string(),
            ));
        }

        if self.enable_batching && self.max_batch_size == 0 {
            return Err(ConfigError::ValidationError(
                "max_batch_size must be > 0 when batching is enabled".to_string(),
            ));
        }

        Ok(())
    }
}

/// Scheduling policy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulingPolicy {
    Fifo,
    Priority,
    BatchAware,
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Memory limit in GB (0 = unlimited)
    pub memory_limit_gb: usize,

    /// GPU memory fraction to use (0.0-1.0)
    pub gpu_memory_fraction: f64,

    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            memory_limit_gb: 0, // unlimited
            gpu_memory_fraction: 0.9,
            enable_cpu_affinity: false,
        }
    }
}

impl ResourceConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.gpu_memory_fraction < 0.0 || self.gpu_memory_fraction > 1.0 {
            return Err(ConfigError::ValidationError(
                "gpu_memory_fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get memory limit in bytes (None = unlimited)
    pub fn memory_limit_bytes(&self) -> Option<usize> {
        if self.memory_limit_gb > 0 {
            Some(self.memory_limit_gb * 1024 * 1024 * 1024)
        } else {
            None
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,

    /// Distributed tracing sample rate (0.0-1.0)
    pub tracing_sample_rate: f64,

    /// Enable Prometheus metrics
    pub enable_prometheus: bool,

    /// Prometheus endpoint
    pub prometheus_endpoint: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            metrics_interval_secs: 15,
            log_level: "info".to_string(),
            tracing_sample_rate: 0.01,
            enable_prometheus: true,
            prometheus_endpoint: "0.0.0.0:9090".to_string(),
        }
    }
}

impl ObservabilityConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.log_level.as_str()) {
            return Err(ConfigError::ValidationError(format!(
                "Invalid log_level '{}'. Must be one of: {:?}",
                self.log_level, valid_levels
            )));
        }

        if self.tracing_sample_rate < 0.0 || self.tracing_sample_rate > 1.0 {
            return Err(ConfigError::ValidationError(
                "tracing_sample_rate must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get metrics interval as Duration
    pub fn metrics_interval(&self) -> Duration {
        Duration::from_secs(self.metrics_interval_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quorum_size_calculation() {
        let config = ByzantineConfig::default();

        // f=1 requires 2*1+1=3 nodes
        assert_eq!(config.quorum_size(3).unwrap(), 3);
        assert_eq!(config.quorum_size(5).unwrap(), 3);

        // Not enough nodes
        assert!(config.quorum_size(2).is_err());
    }

    #[test]
    fn test_network_address_validation() {
        let mut config = NetworkConfig::default();
        config.bind_address = "invalid".to_string();
        assert!(config.validate().is_err());

        config.bind_address = "127.0.0.1:8080".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_byzantine_reputation_validation() {
        let mut config = ByzantineConfig::default();

        config.reputation_decay = 1.5;
        assert!(config.validate().is_err());

        config.reputation_decay = 0.95;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_observability_log_level_validation() {
        let mut config = ObservabilityConfig::default();

        config.log_level = "invalid".to_string();
        assert!(config.validate().is_err());

        config.log_level = "debug".to_string();
        assert!(config.validate().is_ok());
    }
}
