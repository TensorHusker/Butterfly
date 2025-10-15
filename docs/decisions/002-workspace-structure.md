# ADR-002: Cargo Workspace for Multi-Crate Organization

**Status**: Accepted

**Date**: 2025-10-11

## Context

Butterfly consists of multiple logical components:
- **Orchestrator**: Coordinates distributed inference, manages partitioning
- **Worker**: Executes model partitions, communicates with peers
- **Common**: Shared types, protocols, utilities

We need to decide how to organize the codebase structure. Key considerations:

1. **Code Reuse**: Multiple components share common types (tensors, messages, errors)
2. **Independent Development**: Teams should be able to work on orchestrator and worker independently
3. **Testing**: Need to test components in isolation and integration
4. **Versioning**: Components may evolve at different rates
5. **Build Optimization**: Avoid rebuilding unchanged components
6. **Dependency Management**: Shared dependencies should be consistent across components

The organizational structure impacts development velocity, code quality, and long-term maintainability.

## Decision

We will organize Butterfly as a Cargo workspace with three member crates:

```
butterfly/
├── Cargo.toml          # Workspace root
├── crates/
│   ├── orchestrator/   # Orchestrator binary crate
│   ├── worker/         # Worker binary crate
│   └── common/         # Shared library crate
├── docs/               # Documentation
└── tests/              # Integration tests
```

**Workspace Cargo.toml**:
```toml
[workspace]
members = ["crates/orchestrator", "crates/worker", "crates/common"]
resolver = "2"

[workspace.dependencies]
# Shared dependencies with unified versions
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
# ... other shared dependencies
```

**Crate Dependencies**:
- `orchestrator` depends on `common`
- `worker` depends on `common`
- `common` has minimal external dependencies

## Consequences

### Positive Consequences

**Clear Separation of Concerns**
- Each crate has a well-defined responsibility
- Easier to reason about component boundaries
- Natural encapsulation of implementation details
- Public API surfaces are explicit

**Efficient Incremental Builds**
- Cargo only rebuilds changed crates
- Changes to worker don't require rebuilding orchestrator
- Common library changes trigger rebuilds only in dependents
- Parallel compilation of independent crates

**Independent Testing**
- Unit tests live within each crate
- Integration tests in workspace root test cross-component behavior
- Mock/stub implementations easier to create
- Test isolation prevents accidental coupling

**Flexible Versioning**
- Can version crates independently if needed (future)
- SemVer compatibility checks at crate boundaries
- Easier to deprecate or replace components
- Clear changelog per component

**Unified Dependency Management**
- `workspace.dependencies` ensures version consistency
- Single `Cargo.lock` for reproducible builds
- Easier to audit and update dependencies
- Reduces dependency bloat from version conflicts

**Improved Documentation Structure**
- Each crate has its own docs
- `cargo doc --workspace` generates unified documentation
- Examples can live in appropriate crate
- README per component for focused documentation

**Better IDE Experience**
- Rust-analyzer understands workspace structure
- Go-to-definition works across crates
- Refactoring tools respect crate boundaries
- Code completion includes all workspace members

**Facilitates Future Growth**
- Easy to add new components (CLI, monitoring, etc.)
- Can extract specialized crates (e.g., tensor operations)
- Third-party extensions can depend on common crate
- Clear extension points

### Negative Consequences

**Additional Complexity**
- More files and directories to navigate
- Need to understand Cargo workspace mechanics
- Cargo.toml files in multiple locations
- Slightly more boilerplate

**Circular Dependency Prevention**
- Must carefully design crate boundaries
- Cannot have mutual dependencies between crates
- May require moving code to common or refactoring
- Occasionally awkward for tightly coupled components

**Breaking Changes Are More Visible**
- Changes to common crate's public API affect all dependents
- Need to coordinate changes across crates
- May slow down rapid iteration on shared types
- Requires more careful API design

**Testing Complexity**
- Integration tests need to coordinate multiple binaries
- May need additional test harness infrastructure
- Mocking cross-crate dependencies requires careful setup
- End-to-end tests are more complex

**Build Tool Assumptions**
- Developers must use `cargo build` at workspace root
- Some tools may not understand workspace structure
- Need to document workspace-specific commands
- CI/CD must be workspace-aware

### Neutral Consequences

**Feature Flag Management**
- Features must be propagated through dependency chain
- Common crate features need careful consideration
- Can be beneficial (fine-grained control) or annoying (verbosity)

**Path Dependencies**
- Internal dependencies use path references
- Easy to work with in development
- Publishing to crates.io requires version dependencies

**Documentation Standards**
- Must maintain consistent documentation across crates
- Need workspace-wide documentation guidelines
- More surface area to document

## Alternatives Considered

### Alternative 1: Monolithic Single Crate

**Description**: Implement everything in a single crate with multiple binaries and a library.

```
butterfly/
├── Cargo.toml
└── src/
    ├── lib.rs          # Common code
    ├── orchestrator/
    │   └── main.rs     # Orchestrator binary
    └── worker/
        └── main.rs     # Worker binary
```

**Pros**:
- Simplest structure
- Single Cargo.toml
- No cross-crate dependency management
- Fastest initial development

**Cons**:
- Poor encapsulation (all code can access all other code)
- Longer build times (everything rebuilt together)
- Difficult to test components in isolation
- No clear API boundaries
- Hard to extract or reuse components later

**Why Rejected**: While simplest initially, this structure doesn't scale. As the codebase grows, the lack of boundaries makes it harder to reason about and modify. We prioritize long-term maintainability over short-term simplicity.

### Alternative 2: Separate Repositories

**Description**: Each component in its own git repository with independent versioning.

```
butterfly-common/       (separate repo)
butterfly-orchestrator/ (separate repo, depends on common)
butterfly-worker/       (separate repo, depends on common)
```

**Pros**:
- Maximum independence
- True semantic versioning per component
- Clear ownership boundaries
- Can have different release cycles

**Cons**:
- Coordination overhead for changes spanning components
- Need to publish common crate to make it available
- Difficult to make atomic changes across components
- More complex development setup
- Harder to review cross-component changes
- CI/CD more complex

**Why Rejected**: Butterfly is a cohesive system, not a collection of loosely related components. Coordinated changes will be common, especially early in development. Separate repos would create too much friction. This might make sense in the far future for stable APIs.

### Alternative 3: Subdirectories Without Workspace

**Description**: Organize as separate crates but not in a workspace.

```
butterfly/
├── orchestrator/       (independent crate)
├── worker/             (independent crate)
└── common/             (independent crate)
```

**Pros**:
- Clear separation like workspace
- Simple structure
- No workspace-specific concepts

**Cons**:
- No shared dependency versions
- Each crate has its own Cargo.lock
- No unified build command
- Harder to ensure consistency
- More Cargo.lock files to manage
- No cross-crate optimization

**Why Rejected**: This loses most benefits of both monorepo and multi-repo approaches. Workspace provides valuable features (unified dependencies, coordinated builds) without the overhead of multiple repositories.

### Alternative 4: Feature-Flag Based Modularity

**Description**: Single crate with feature flags to enable/disable components.

```toml
[features]
default = ["orchestrator", "worker"]
orchestrator = ["dep:tokio"]
worker = ["dep:ndarray"]
```

**Pros**:
- Single crate, simple structure
- Can build only what you need
- Fine-grained control over dependencies

**Cons**:
- Feature flags can lead to combinatorial complexity
- Harder to test all feature combinations
- Doesn't solve encapsulation problem
- Confusion about what's enabled
- Conditional compilation scattered throughout code

**Why Rejected**: Feature flags are useful for optional functionality, but using them as a primary modularity mechanism is an anti-pattern. They don't provide the clear boundaries we need.

### Alternative 5: Hybrid Workspace (More Granular)

**Description**: More fine-grained crate separation.

```
butterfly/
├── crates/
│   ├── orchestrator/
│   ├── worker/
│   ├── protocol/       # Message definitions
│   ├── tensor/         # Tensor operations
│   ├── partition/      # Partitioning algorithms
│   └── common/         # Utilities
```

**Pros**:
- Maximum modularity
- Very clear responsibilities
- Easier to reuse specific components
- Better dependency hygiene

**Cons**:
- Over-engineering for current scope
- More boilerplate and navigation
- Premature abstraction
- Dependency graph becomes complex
- Harder to make sweeping changes

**Why Rejected**: While we may evolve to this structure, it's premature for the current stage. We'll start with three crates and extract more as patterns emerge. YAGNI (You Aren't Gonna Need It) principle applies.

## Implementation Notes

### Crate Responsibilities

**orchestrator**:
- Request handling and validation
- Model partitioning algorithms
- Worker coordination and scheduling
- Response aggregation
- Health monitoring

**worker**:
- Model partition loading and execution
- Inter-worker communication
- Local resource management
- Orchestrator heartbeat
- Tensor computation

**common**:
- Protocol message definitions
- Tensor representation and serialization
- Error types and handling
- Configuration structures
- Network utilities
- Logging macros

### Public API Design

**common crate** should expose:
- `pub` types used across components (messages, configs)
- Serialization/deserialization implementations
- Error types
- Utility functions

**orchestrator** and **worker** should:
- Keep most code private
- Export only intentional extension points
- Use `pub(crate)` for internal APIs
- Document public interfaces thoroughly

### Development Workflow

**Building**:
```bash
# Build all crates
cargo build --workspace

# Build specific crate
cargo build -p butterfly-orchestrator

# Release builds
cargo build --workspace --release
```

**Testing**:
```bash
# Test all crates
cargo test --workspace

# Test specific crate
cargo test -p butterfly-common

# Integration tests
cargo test --test '*'
```

**Documentation**:
```bash
# Generate docs for all crates
cargo doc --workspace --no-deps --open
```

### Migration Path

As the system evolves, we may extract additional crates:
1. **tensor**: If tensor operations become complex
2. **protocol**: If message definitions grow substantially
3. **partition**: If partitioning algorithms warrant dedicated crate
4. **metrics**: If observability becomes sophisticated
5. **cli**: If we add command-line tools

Extraction criteria:
- Clear, stable API boundary
- Used by multiple components
- Substantial implementation (>1000 LOC)
- Potential for external reuse

### CI/CD Considerations

**Testing**:
- Run tests for all crates
- Check cross-crate compatibility
- Test workspace as a whole

**Linting**:
- `cargo clippy --workspace --all-targets`
- `cargo fmt --all -- --check`

**Coverage**:
- Generate coverage for entire workspace
- Track per-crate coverage separately

**Caching**:
- Cache `target/` directory
- Incremental builds across CI runs

## References

- [Cargo Workspace Documentation](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Rust API Guidelines: Project Structure](https://rust-lang.github.io/api-guidelines/naming.html)
- [The Cargo Book: Workspace](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Rust Patterns: Workspace](https://rust-unofficial.github.io/patterns/patterns/structural/workspace.html)

## Revision History

| Date | Author | Change Description |
|------|--------|--------------------|
| 2025-10-11 | Agent-4 | Initial version |

## Notes

This decision reflects our current understanding of the system's architecture. We chose a workspace structure that's complex enough to provide clear boundaries but simple enough to remain manageable.

The three-crate structure (orchestrator, worker, common) is a starting point. We're prepared to refactor into more crates if natural boundaries emerge, but we're also avoiding premature optimization.

Key principle: **Optimize for reading and understanding, not for writing**. Code is read far more often than it's written, and clear structure aids comprehension.

We commit to reassessing this structure at major milestones (v0.5, v1.0) and refactoring if patterns suggest a better organization.
