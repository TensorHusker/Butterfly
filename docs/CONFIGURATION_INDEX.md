# Butterfly Configuration System Documentation Index

This index provides an overview of the comprehensive configuration system design for Butterfly distributed inference.

## Overview

The Butterfly configuration system provides type-safe, Byzantine-resilient configuration management with hot reload capabilities. It's designed specifically for distributed systems requiring strong consistency guarantees and fault tolerance.

## Core Documents

### 1. Configuration Design Specification
**File:** [`configuration_design.md`](./configuration_design.md)
**Size:** 47KB
**Purpose:** Comprehensive architecture specification

**Contents:**
- Configuration architecture and hierarchy
- Precedence rules (defaults → files → environment → runtime)
- Complete schema for all configuration sections
- Dynamic reconfiguration protocol (Byzantine-safe)
- Hot reload mechanism with consensus
- Safety guarantees and atomicity
- CLI and API integration
- Testing strategy
- Performance characteristics
- Migration and compatibility

**Key Sections:**
1. Configuration Architecture (layered design, hierarchy)
2. Configuration Schema (all sections with full details)
3. Configuration Categories (static vs dynamic, consensus vs local)
4. Dynamic Reconfiguration Protocol (PROPOSE → PREPARE → COMMIT → ROLLBACK)
5. Configuration Manager Implementation (overview)
6. CLI Integration (commands and flags)
7. API Integration (HTTP/gRPC endpoints)
8. Safety Guarantees (type safety, Byzantine resilience, atomicity)
9. Performance Considerations (load time, hot reload overhead)
10. Testing Strategy (unit, integration, property-based, chaos)

### 2. Configuration Schema Example
**File:** [`config_schema.toml`](./config_schema.toml)
**Size:** 38KB
**Purpose:** Complete annotated configuration file template

**Contents:**
- Every configuration option with inline documentation
- Default values and typical ranges
- Cluster-wide vs node-local annotations
- Examples of all configuration sections:
  - Metadata (versioning, environment)
  - Node (identity, directories)
  - Network (protocol, connections, compression, retry)
  - Byzantine (fault tolerance, failure detection, consensus, checkpointing)
  - Partition (strategies, dynamic rebalancing)
  - Scheduling (policies, queues, stragglers, pipeline)
  - Resources (memory, CPU, GPU, disk, rate limits)
  - Observability (logging, metrics, tracing, profiling)
  - Security (TLS, signing, access control, authentication)
  - Advanced (experimental features)

**Use Cases:**
- Reference for all available options
- Template for creating new configurations
- Documentation of defaults and constraints
- Examples of different configuration patterns

### 3. butterfly-config Crate Specification
**File:** [`butterfly-config-crate-spec.md`](./butterfly-config-crate-spec.md)
**Size:** 45KB
**Purpose:** Detailed implementation specification for configuration crate

**Contents:**
- Complete module structure (19 modules)
- Type definitions for all configuration structs
- ConfigLoader implementation (file, env, include, merge)
- Validation framework and rules
- ConfigManager with hot reload state machine
- Byzantine consensus protocol for configuration updates
- Configuration versioning and compatibility
- File watching infrastructure
- Error types and handling
- Testing strategy (unit, integration, property-based, chaos)
- Performance targets
- Usage examples
- Dependencies (Cargo.toml)
- Implementation checklist

**Key Modules:**
1. `schema/` - Configuration type definitions (9 sub-modules)
2. `loader/` - Configuration loading (4 sub-modules)
3. `validation/` - Validation framework (5 sub-modules)
4. `manager/` - Configuration lifecycle management (3 sub-modules)
5. `consensus/` - Byzantine-safe updates (3 sub-modules)
6. `versioning/` - Version management (3 sub-modules)
7. `watchers/` - File change detection (2 sub-modules)

## Quick Start Guide

### 1. Understanding the System

Read in this order:
1. **Configuration Design (Sections 1-3)** - Architecture and schema overview
2. **Configuration Schema (TOML)** - Browse example configurations
3. **Crate Specification (Overview + Module Structure)** - Implementation details

### 2. Creating a Configuration

Start with [`config_schema.toml`](./config_schema.toml):

```bash
# Copy the template
cp docs/config_schema.toml config.toml

# Edit for your environment
# - Set node.node_id
# - Configure network.coordinator_addr
# - Adjust byzantine.f_value for cluster size
# - Set resource limits
# - Configure observability settings

# Validate (once implemented)
butterfly config validate --config config.toml

# Start node
butterfly start --config config.toml
```

### 3. Runtime Configuration Updates

```bash
# Update via API (requires authentication)
curl -X POST http://coordinator:8080/api/v1/config \
  -H "Authorization: Bearer $TOKEN" \
  -d @new-config.toml

# Update via CLI
butterfly config update --file new-config.toml

# View configuration history
butterfly config history --limit 10

# Rollback if needed
butterfly config rollback --version 1.2.3
```

### 4. Implementing Configuration Support

When adding a new component that uses configuration:

```rust
use butterfly_config::{ConfigManager, ConfigSubscriber};

struct MyComponent {
    config: MyComponentConfig,
}

impl ConfigSubscriber for MyComponent {
    fn on_config_changed(&self, old: &ButterflyConfig, new: &ButterflyConfig) {
        if old.my_section != new.my_section {
            self.update_config(&new.my_section);
        }
    }

    fn name(&self) -> &str {
        "MyComponent"
    }
}

// Subscribe to configuration changes
config_manager.subscribe(Arc::new(my_component)).await;
```

## Configuration Sections Reference

### Network Configuration
- **Purpose:** Inter-node communication settings
- **Key Settings:** Protocol, addresses, buffers, compression, retry policies
- **Mutability:** Mostly static (requires restart), some dynamic (compression, timeouts)
- **See:** `configuration_design.md` Section 2.2, `config_schema.toml` Network Section

### Byzantine Configuration
- **Purpose:** Fault tolerance and consensus parameters
- **Key Settings:** f_value, failure detection, Byzantine agreement, checkpointing
- **Mutability:** Mostly static-critical, some dynamic-consensus
- **See:** `configuration_design.md` Section 2.3, `config_schema.toml` Byzantine Section
- **Important:** f_value determines cluster size (2f+1 minimum)

### Partition Configuration
- **Purpose:** Model partitioning strategies
- **Key Settings:** Strategy selection, strategy parameters, dynamic rebalancing
- **Mutability:** Dynamic-consensus (can change at runtime via consensus)
- **See:** `configuration_design.md` Section 2.4, `config_schema.toml` Partition Section

### Scheduling Configuration
- **Purpose:** Workload scheduling and load balancing
- **Key Settings:** Policy, queue management, straggler mitigation, pipeline parallelism
- **Mutability:** Dynamic-consensus (policy changes) and dynamic-local (queue limits)
- **See:** `configuration_design.md` Section 2.5, `config_schema.toml` Scheduling Section

### Resources Configuration
- **Purpose:** Resource limits and allocation
- **Key Settings:** Memory limits, CPU allocation, GPU configuration, rate limits
- **Mutability:** Mostly static-local (requires restart), some dynamic-local (rate limits)
- **See:** `configuration_design.md` Section 2.6, `config_schema.toml` Resources Section

### Observability Configuration
- **Purpose:** Metrics, logging, and distributed tracing
- **Key Settings:** Log level, metrics export, tracing sampling
- **Mutability:** Dynamic-local (can change without coordination)
- **See:** `configuration_design.md` Section 2.7, `config_schema.toml` Observability Section

### Security Configuration
- **Purpose:** Authentication, encryption, access control
- **Key Settings:** TLS/mTLS, signing, access control policies, API authentication
- **Mutability:** Mostly static-critical (TLS settings require restart)
- **See:** `configuration_design.md` Section 2.8, `config_schema.toml` Security Section

## Configuration Mutability Matrix

| Configuration Type | Runtime Update? | Restart Required? | Consensus Required? | Example Settings |
|-------------------|-----------------|-------------------|---------------------|------------------|
| **Static-Critical** | ❌ No | ✅ Yes | N/A | byzantine.f_value, network.coordinator_addr |
| **Static-Local** | ❌ No | ✅ Yes | ❌ No | node.node_id, resources.gpu.device_ids |
| **Dynamic-Consensus** | ✅ Yes | ❌ No | ✅ Yes | partition.strategy, network.compression |
| **Dynamic-Local** | ✅ Yes | ❌ No | ❌ No | observability.logging.level, resources.rate_limits |

**Legend:**
- **Static-Critical:** Cannot change at runtime, affects cluster coordination
- **Static-Local:** Cannot change at runtime, node-specific
- **Dynamic-Consensus:** Can change at runtime, requires cluster agreement
- **Dynamic-Local:** Can change at runtime, no coordination needed

## Byzantine-Safe Configuration Updates

The configuration system uses a four-phase consensus protocol to ensure all nodes converge to identical configurations even under Byzantine failures:

```
┌─────────────────────────────────────────────────┐
│ Phase 1: PROPOSE                                │
│ Coordinator broadcasts new configuration        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ Phase 2: PREPARE                                │
│ Nodes validate and vote (PREPARE_OK/REJECT)    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ Phase 3: COMMIT (if 2f+1 PREPARE_OK)           │
│ Coordinator broadcasts COMMIT                   │
│ Nodes apply configuration atomically            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ Phase 4: FINALIZE                               │
│ Nodes confirm application                       │
│ Configuration history updated                   │
└─────────────────────────────────────────────────┘
```

**Failure Handling:**
- Any PREPARE_REJECT → Automatic ROLLBACK
- Timeout waiting for votes → Automatic ROLLBACK
- Hash mismatch detected → Automatic ROLLBACK, Byzantine node isolated
- Node crash during update → Recovery fetches current config from peers

**See:** `configuration_design.md` Section 4 for detailed protocol specification

## Implementation Roadmap

### Phase 1: Core Configuration (Week 1-2)
- [ ] Implement schema types (`schema/`)
- [ ] Implement configuration loader (`loader/`)
- [ ] Implement validation framework (`validation/`)
- [ ] Unit tests for loading and validation
- [ ] Basic integration tests

### Phase 2: Configuration Manager (Week 3)
- [ ] Implement ConfigManager (`manager/`)
- [ ] Implement subscriber pattern
- [ ] Implement configuration history
- [ ] Integration tests for manager

### Phase 3: Hot Reload (Week 4)
- [ ] Implement file watchers (`watchers/`)
- [ ] Implement hot reload state machine
- [ ] Integration tests for hot reload
- [ ] Performance benchmarks

### Phase 4: Byzantine Consensus (Week 5-6)
- [ ] Implement consensus protocol (`consensus/`)
- [ ] Implement all consensus phases
- [ ] Integration with butterfly-coordination
- [ ] Byzantine failure tests
- [ ] Chaos engineering tests

### Phase 5: CLI/API Integration (Week 7)
- [ ] Integrate with butterfly-cli
- [ ] Implement API endpoints in butterfly-api
- [ ] End-to-end tests
- [ ] Documentation and examples

### Phase 6: Production Hardening (Week 8)
- [ ] Property-based tests
- [ ] Performance optimization
- [ ] Production-ready error messages
- [ ] Operator documentation
- [ ] Migration tools

## Testing Strategy

### Unit Tests
- Configuration parsing (TOML, environment variables)
- Validation rules (individual and cross-section)
- Configuration merging and precedence
- Hashing and versioning
- Target: >90% coverage

### Integration Tests
- Full configuration lifecycle
- Hot reload with file watching
- Byzantine consensus protocol
- Rollback functionality
- Multi-node synchronization
- Target: All critical paths covered

### Property-Based Tests (proptest)
- Configuration serialization round-trips
- Merging properties (associativity, commutativity)
- Validation idempotence
- Hash consistency
- Target: 1000 test cases per property

### Chaos Tests
- Random node failures during consensus
- Network partitions during updates
- Conflicting concurrent updates
- Byzantine behavior injection
- Configuration file corruption
- Target: 100% failure recovery

### Performance Benchmarks
- Configuration loading (<10ms)
- Validation (<5ms)
- Hot reload consensus (<100ms local, <500ms cross-DC)
- Memory overhead (<1MB per manager)
- Hash computation (<1ms)

## Performance Characteristics

### Configuration Loading
- Small config (<10KB): <1ms
- Medium config (<100KB): <10ms
- Large config (<1MB): <100ms
- Includes parsing, validation, hashing

### Hot Reload Latency
- **Optimistic case** (no failures, local network):
  - 3 RTT: ~30ms
  - Phases: PROPOSE (1 RTT) + PREPARE (1 RTT) + COMMIT (1 RTT)

- **With Byzantine nodes** (1 Byzantine, 5 nodes):
  - 5 RTT: ~50ms
  - Additional rounds for Byzantine detection

- **Cross-datacenter** (US-West to US-East):
  - 3 RTT: ~150-300ms
  - Network latency dominates

### Memory Overhead
- Active configuration: 10-50KB
- Configuration history (10 versions): 100-500KB
- Watchers and subscribers: 10KB per component
- Total per node: <1MB

## Troubleshooting Guide

### Configuration Won't Load
1. Check file path is correct and readable
2. Validate TOML syntax (use online validator)
3. Check validation errors in logs
4. Verify environment variables are correctly formatted
5. Check include paths are resolvable

### Configuration Update Fails
1. Check if cluster has sufficient nodes (≥ 2f+1)
2. Verify network connectivity to all nodes
3. Check if update is for static-critical parameter (requires restart)
4. Review validation errors in response
5. Check if subscriber rejected change
6. Verify Byzantine consensus not timing out

### Configuration Divergence
1. Check configuration hashes: `butterfly config show --hash`
2. Force synchronization: `butterfly config sync`
3. Check for Byzantine node: Review reputation scores
4. Verify all nodes on compatible versions
5. Check audit logs for unauthorized changes

### Hot Reload Not Working
1. Verify file watcher is active: Check logs for "watching file"
2. Check file permissions (readable by butterfly process)
3. Verify configuration changes are in dynamic-* category
4. Check if update requires consensus (cluster-wide changes)
5. Review hot reload state in metrics

## Related Documentation

- [`architecture.md`](./architecture.md) - Overall system architecture
- [`coordination_protocol.md`](./coordination_protocol.md) - Byzantine coordination details
- [`algorithms.md`](./algorithms.md) - Partitioning and scheduling algorithms
- [`performance-analysis.md`](./performance-analysis.md) - System performance characteristics

## Contributing

When adding new configuration options:

1. **Update Schema** (`config_schema.toml`)
   - Add parameter with inline documentation
   - Specify default value
   - Annotate as cluster-wide or node-local
   - Document mutability (static vs dynamic)

2. **Update Types** (`butterfly-config/src/schema/*.rs`)
   - Add field to appropriate struct
   - Implement `serde` serialization
   - Add any custom validation logic

3. **Add Validation** (`butterfly-config/src/validation/*_rules.rs`)
   - Implement validation rule if needed
   - Add cross-section validation if affects other parameters

4. **Update Documentation** (`configuration_design.md`)
   - Document in appropriate section
   - Add to mutability matrix if dynamic
   - Update version compatibility notes

5. **Add Tests**
   - Unit tests for new validation rules
   - Integration tests if affects hot reload
   - Update fixtures with new parameter

## Glossary

- **Byzantine Failure:** Arbitrary node behavior (crashes, corruption, malicious)
- **f_value:** Number of Byzantine failures tolerated (cluster needs 2f+1 nodes)
- **Quorum:** Minimum number of nodes for consensus (2f+1)
- **Consensus:** Agreement protocol ensuring all nodes reach same decision
- **Hot Reload:** Updating configuration without restarting process
- **Cluster-Wide:** Configuration must be identical across all nodes
- **Node-Local:** Configuration specific to individual node
- **Static:** Cannot change without restart
- **Dynamic:** Can change at runtime
- **Precedence:** Order in which configuration sources override each other

## Support and Contact

For questions or issues with the configuration system:
- Review this documentation first
- Check troubleshooting guide
- Review configuration examples in [`examples/`](../examples/)
- File issue on GitHub with configuration and logs

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Maintained By:** Butterfly Contributors
