# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Butterfly is a distributed inference system for transformers and large language models. The core purpose is solving LLM scaling problems by distributing inference across multiple nodes, enabling efficient computation for models too large for single machines.

**Key Innovation**: Novel algorithms for partitioning transformers and minimizing communication overhead in distributed settings.

## Architecture Status

This is a newly initialized repository. The codebase is in early development phase.

### Expected Architecture (To Be Implemented)

Based on the project goals, the system will likely include:

1. **Distributed Inference Engine**: Core logic for splitting transformer computations across nodes
2. **Communication Layer**: Low-overhead inter-node communication for tensor passing
3. **Model Partitioning**: Algorithms to intelligently divide transformer layers/attention heads
4. **Scheduling System**: Workload distribution and synchronization across compute nodes
5. **API/Interface**: Client-facing API for submitting inference requests

## Development Commands

*To be added once build system is established*

## Code Organization Strategy

### Branch Structure

**Multi-Agent Development Model**: Different branches for different AI agents/developers working on isolated components.

```
main (integration branch)
├── agent/inference-engine    # Core distributed inference logic
├── agent/communication       # Inter-node communication layer
├── agent/partitioning        # Model partitioning algorithms
├── agent/scheduling          # Workload distribution and sync
├── agent/api                 # Client-facing API
├── agent/monitoring          # Observability and metrics
└── feature/*                 # Specific feature branches
```

### Branch Naming Conventions

- `agent/<component-name>`: Long-lived branches for major architectural components
- `feature/<description>`: Short-lived branches for specific features
- `fix/<issue-description>`: Bug fix branches
- `experiment/<idea>`: Experimental/research branches

### Integration Workflow

1. Each agent works independently on their component branch
2. Components are designed with clear interfaces/contracts
3. Regular integration PRs to `main` with comprehensive tests
4. CI/CD validates cross-component compatibility
5. `main` always represents a working integrated system

### Component Isolation Principles

- **Clear APIs**: Each component exposes well-defined interfaces
- **Minimal Coupling**: Components communicate through abstract protocols
- **Independent Testing**: Each branch has its own test suite
- **Documentation**: Interface contracts documented at component boundaries

### When to Merge

Merge agent branches to `main` when:
- Component reaches stable API milestone
- Integration tests pass
- Documentation is complete
- No breaking changes to other components (or coordinated multi-component merge)

### Conflict Resolution

For cross-component dependencies:
1. Define interface contracts in `main` first
2. Agent branches implement against stable interfaces
3. Use feature flags for incompatible changes during transition
4. Coordinate breaking changes through design discussions

## Repository Information

- **License**: Check LICENSE file
- **Git Branch**: `main` (primary development branch, integration point)
