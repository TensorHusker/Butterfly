# Changelog

All notable enhancements and improvements to the Butterfly distributed inference system.

## [Unreleased] - Enhanced Edition

### Added

#### Node Management (`node.rs`)
- **Display trait** for `NodeId` - Better debugging and logging support
- **Validation in constructor** - `NodeCapability::new()` with comprehensive validation
  - Validates positive memory, compute FLOPS, and network bandwidth
  - Ensures at least one device is present
- **Helper methods** for `NodeCapability`:
  - `memory_per_device()` - Calculate memory available per device
  - `compute_per_device()` - Calculate compute per device
  - `from_uuid()` and `as_uuid()` for `NodeId` conversion
- **Error types** - `NodeError` enum for better error handling

#### Communication Layer (`communication.rs`)
- **Retry logic** - `send_message_with_retry()` with exponential backoff
- **Timeout support** - `receive_message_timeout()` for bounded wait times
- **Configurable parameters** - `with_config()` constructor for custom timeout and retry settings
- **Enhanced error types**:
  - `Timeout` - Operation exceeded time limit
  - `MaxRetriesExceeded` - Retry attempts exhausted

#### Partitioning (`partition.rs`)
- **Constructor validation** - `PartitionManager::new()` returns `Result`
  - Validates positive number of layers and nodes
- **Query methods**:
  - `config()` - Get partition configuration
  - `total_partitioned_layers()` - Count partitioned layers
  - `is_fully_partitioned()` - Check if all layers assigned
- **Enhanced error types**:
  - `InvalidConfig` - Configuration validation errors
  - `PartitionConflict` - Partition assignment conflicts

#### Load Balancing (`load_balancer.rs`)
- **Advanced statistics** in `LoadStatistics`:
  - `std_dev` - Standard deviation of loads
  - `coefficient_of_variation()` - Normalized variability measure
  - `is_well_balanced()` - Threshold-based balance check
- **Node filtering methods**:
  - `get_overloaded_nodes()` - Find nodes above load threshold
  - `get_underloaded_nodes()` - Find nodes below load threshold
- **Improved statistics calculation** - Proper handling of empty node sets

#### Fault Tolerance (`fault_tolerance.rs`)
- **Enhanced health tracking**:
  - `time_since_heartbeat()` - Duration since last contact
  - `is_degraded()` - Check for degraded state
  - `has_failed()` - Check for failed state
- **Cluster-wide monitoring**:
  - `degraded_nodes()` - Get all degraded nodes
  - `failed_nodes()` - Get all failed nodes
  - `get_health_info()` - Query specific node health
  - `total_nodes()` - Count monitored nodes
  - `cluster_health_percentage()` - Overall cluster health metric

#### Testing & Documentation
- **Integration tests** (`tests/integration_tests.rs`):
  - End-to-end node setup workflow
  - Complete partitioning workflow
  - Load balancer workflow
  - Health monitoring
  - Node capability validation
  - Partition manager validation
  - Communication with timeout
  - Load statistics calculations
- **README enhancements**:
  - Key enhancements section
  - Advanced usage examples for load balancing
  - Advanced usage examples for health monitoring
  - Advanced usage examples for communication with retry

#### Library Exports
- **Expanded public API** (`lib.rs`):
  - All error types now exported
  - `NetworkManager` exported
  - `HealthInfo` exported
  - `LoadStatistics` exported

### Changed

#### Breaking Changes
- `PartitionManager::new()` now returns `Result<Self, PartitionError>` instead of `Self`
  - **Migration**: Change `PartitionManager::new(config)` to `PartitionManager::new(config)?` or `.unwrap()`

#### Non-Breaking Changes
- `NodeCapability` can now be created via validated constructor or struct literal
- `CommunicationLayer` can use `new()` or `with_config()` for custom settings
- `LoadStatistics` now includes `std_dev` field (additional data, backward compatible in most cases)

### Improved

#### Error Handling
- Comprehensive validation across all core types
- Type-safe error propagation with `thiserror`
- Detailed error messages for easier debugging

#### Robustness
- Retry logic prevents transient failures
- Timeout support prevents indefinite blocking
- Validation catches configuration errors early

#### Observability
- Enhanced metrics for load distribution
- Cluster-wide health monitoring
- Coefficient of variation for load balance quality
- Standard deviation for load distribution spread

#### Code Quality
- Consistent error handling patterns
- Additional helper methods for common operations
- Improved type safety with validated constructors
- Better separation of concerns

### Technical Details

#### Statistics Improvements
- Standard deviation calculation uses proper variance formula
- Coefficient of variation provides scale-independent variability measure
- Well-balanced check uses CV threshold (typically 0.2 for good balance)

#### Retry Strategy
- Exponential backoff: 100ms, 200ms, 300ms, etc.
- Configurable maximum attempts (default: 3)
- Last error preserved and returned

#### Validation Strategy
- Fail-fast on invalid configurations
- Comprehensive checks in constructors
- Clear error messages indicating the issue

## [0.1.0] - Initial Release

### Added
- Core distributed inference architecture
- Node management and registry
- Layer partitioning strategies (Sequential, Balanced, Custom)
- Asynchronous communication layer
- Distributed attention and feed-forward implementations
- Health monitoring and fault tolerance
- Load balancing for heterogeneous hardware
- 18 comprehensive unit tests
- Demo application

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles.
