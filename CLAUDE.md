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

## Repository Information

- **License**: Check LICENSE file
- **Git Branch**: `main` (primary development branch)
