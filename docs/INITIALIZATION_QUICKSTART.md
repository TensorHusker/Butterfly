# Butterfly Initialization - Quick Start Guide

## 5-Minute Overview

This guide provides the essential information to understand Butterfly's initialization system without reading the full specifications.

## What Happens When You Start Butterfly?

```
1. COLD START (0-10s)
   - Load configuration file
   - Detect GPUs and hardware
   - Initialize network stack

2. DISCOVERY (10-20s)
   - Find other nodes via DNS/Consul/static list
   - Exchange capabilities (RAM, GPU, network)

3. ELECTION (20-22s)
   - Elect coordinator using Raft consensus
   - ~2 seconds with 3-7 nodes

4. ASSIGNMENT (22-27s)
   - Coordinator computes optimal partition
   - Assigns layers to nodes based on capacity

5. LOADING (27-147s)
   - Download model weights from S3/filesystem
   - Verify checksums
   - Upload to GPU memory
   - Compile CUDA kernels
   - Run warmup inference

6. VALIDATION (147-157s)
   - Compute checksums of loaded weights
   - Build Merkle tree across cluster
   - Verify global consistency

7. READY (157s)
   - Cluster operational
   - Can serve inference requests
```

**Total Time**: ~2.5 minutes for 70B model on 8 nodes with S3.

## Key Concepts

### Node States

- **COLD**: Just started
- **JOINING**: Connecting to cluster
- **LOADING**: Downloading weights
- **READY**: Waiting for work
- **OPERATIONAL**: Serving requests

### Cluster Roles

- **Coordinator**: Elected leader, assigns work
- **Worker**: Executes inference on partition

### Partition

A range of model layers assigned to one node:
```
Model: 80 layers total
Node A: layers 0-26   (27 layers)
Node B: layers 27-53  (27 layers)
Node C: layers 54-79  (26 layers)
```

## Configuration Example

```toml
[cluster]
cluster_name = "my-cluster"
discovery_method = "static"
seed_nodes = ["10.0.1.10:7000", "10.0.1.11:7000"]
quorum_size = 3

[model]
source_type = "s3"
source_path = "s3://ml-models/llama-70b-v2"

[node]
role = "worker"  # or "coordinator"
bind_address = "0.0.0.0:7000"
```

## Starting a Node

```bash
# Coordinator
butterfly-node --config coord.toml --role coordinator

# Workers
butterfly-node --config worker.toml --role worker --seed-nodes coord:7000
```

## Monitoring Initialization

### Check Progress

```bash
# Watch initialization logs
tail -f /var/log/butterfly/node.log | grep -E "phase|progress"

# Query status endpoint
curl http://localhost:8080/status | jq
```

### Expected Output

```json
{
  "state": "LOADING",
  "phase": "loading_weights",
  "progress_percent": 45.2,
  "eta_seconds": 68,
  "cluster_size": 3,
  "coordinator": "node-a-7f3d"
}
```

## Common Issues

### Can't Find Peers

```bash
# Check network
ping coordinator-hostname

# Check port open
nc -zv coordinator-hostname 7000

# Check DNS (if using DNS discovery)
dig +short _butterfly._tcp.example.com SRV
```

### Loading Fails

```bash
# Check S3 access
aws s3 ls s3://ml-models/llama-70b-v2/

# Check disk space
df -h /var/butterfly/models

# Check GPU memory
nvidia-smi
```

### Validation Fails

Usually means:
- Different model versions on different nodes
- Corrupted download

**Fix**: Clear cache and retry:
```bash
butterfly-admin clear-cache
butterfly-node --config config.toml
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                   User/Client                    │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│              Coordinator Node                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Discovery  │  │  Election  │  │ Assignment│  │
│  └────────────┘  └────────────┘  └──────────┘  │
└───────────────────┬─────────────────────────────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Worker 1│ │Worker 2│ │Worker 3│
    │Layers  │ │Layers  │ │Layers  │
    │  0-26  │ │ 27-53  │ │ 54-79  │
    └────────┘ └────────┘ └────────┘
```

## Safety Guarantees

1. **No split-brain**: Only one coordinator at a time
2. **Consistent weights**: All nodes have identical weights
3. **No premature serving**: Won't accept requests until validated
4. **Fault tolerance**: Continues if ≥quorum nodes operational

## Performance Tips

### Faster Initialization

1. **Use local filesystem** instead of S3 (if possible)
2. **Pre-warm cache** on first boot
3. **Use faster network** (10Gbps+ recommended)
4. **SSD/NVMe storage** for weight files

### Reduce Variance

1. **Pin CPU cores** to avoid context switching
2. **Disable CPU frequency scaling**
3. **Use dedicated network** for cluster traffic
4. **Synchronize clocks** (NTP)

## Next Steps

### For Operators

- Read [INITIALIZATION_SUMMARY.md](INITIALIZATION_SUMMARY.md) for full overview
- Set up monitoring dashboards
- Plan failure scenarios

### For Developers

- Read [initialization_design.md](initialization_design.md) for design details
- Read [BOOTSTRAP_PROTOCOL.md](BOOTSTRAP_PROTOCOL.md) for protocol spec
- Read [butterfly-cluster/SPECIFICATION.md](../crates/butterfly-cluster/SPECIFICATION.md) for implementation guide

### For Researchers

- Read formal proofs in [initialization_design.md](initialization_design.md)
- See [coordination_protocol.md](coordination_protocol.md) for runtime protocol
- Review Byzantine tolerance mechanisms

## Metrics to Monitor

```promql
# Time to ready
init_duration_seconds{phase="total"}

# Current state
node_state{state="OPERATIONAL"}

# Cluster health
cluster_operational_nodes / cluster_size

# Loading progress
model_bytes_downloaded_total
rate(model_bytes_downloaded_total[1m])
```

## Getting Help

1. **Check logs**: `/var/log/butterfly/node.log`
2. **Query status**: `curl localhost:8080/status`
3. **Run diagnostics**: `butterfly-admin diagnose`
4. **File issue**: GitHub with `[initialization]` tag

---

**Quick Reference Complete**
**For Details**: See [INITIALIZATION_SUMMARY.md](INITIALIZATION_SUMMARY.md)
