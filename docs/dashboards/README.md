# Butterfly Grafana Dashboards

This directory contains Grafana dashboard specifications for monitoring the Butterfly distributed inference system.

## Dashboard Overview

| Dashboard | Purpose | Key Metrics |
|-----------|---------|-------------|
| `overview.json` | High-level cluster health | Request rate, latency, error rate, cluster size |
| `request-latency.json` | Request performance analysis | P50/P95/P99 latency, breakdown by layer |
| `consensus.json` | Raft consensus monitoring | Term, leader elections, commit lag |
| `network.json` | Network performance | Bandwidth, latency matrix, packet loss |
| `resources.json` | Resource utilization | CPU, memory, GPU per node |
| `byzantine.json` | Byzantine fault detection | Fault rate, reputation scores, verification failures |
| `checkpoints.json` | Checkpoint monitoring | Creation/restore times, sizes, verification |

## Importing Dashboards

### Via Grafana UI

1. Open Grafana (http://grafana:3000)
2. Click "+" → "Import"
3. Upload JSON file or paste JSON content
4. Select Prometheus data source
5. Click "Import"

### Via Provisioning

```yaml
# /etc/grafana/provisioning/dashboards/butterfly.yaml
apiVersion: 1

providers:
  - name: 'Butterfly'
    orgId: 1
    folder: 'Butterfly'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards/butterfly
```

Copy JSON files to `/etc/grafana/dashboards/butterfly/`

### Via API

```bash
# Import dashboard via API
curl -X POST \
  http://admin:admin@grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @overview.json
```

## Dashboard Variables

All dashboards support these variables:

- `$cluster`: Cluster name (default: all)
- `$node`: Node ID (default: all)
- `$model`: Model name (default: all)
- `$interval`: Time range for rate calculations (default: 5m)

## Alerting

Alerts are configured in `alerts/` directory:

- `alerts/latency.yaml`: High latency alerts
- `alerts/errors.yaml`: Error rate alerts
- `alerts/consensus.yaml`: Consensus health alerts
- `alerts/byzantine.yaml`: Byzantine fault alerts

## Creating Custom Dashboards

Use the provided dashboards as templates:

1. Copy existing dashboard JSON
2. Modify panels to show desired metrics
3. Update variable queries if needed
4. Test with your cluster
5. Export and save to this directory

## Dashboard Design Principles

### Layout

- Top row: Key metrics (RED: Rate, Errors, Duration)
- Middle rows: Component-specific metrics
- Bottom row: Detailed breakdowns and traces

### Time Ranges

- Default: Last 1 hour
- Refresh: Every 30 seconds
- Quick ranges: 5m, 15m, 1h, 6h, 24h, 7d

### Color Coding

- Green: Healthy (< threshold)
- Yellow: Warning (approaching threshold)
- Red: Critical (exceeded threshold)

### Panel Types

- **Stat**: Single value (current request rate)
- **Graph**: Time series (latency over time)
- **Heatmap**: Distribution (latency percentiles)
- **Table**: List (top slowest nodes)
- **Gauge**: Current vs. threshold (memory usage)

## Troubleshooting Dashboards

### No Data

**Check:**
1. Prometheus is scraping metrics: `kubectl logs -n butterfly-observability prometheus-0`
2. Metrics endpoint is accessible: `curl http://coordinator-1:9090/metrics`
3. Time range includes data: Try "Last 24 hours"

### Incorrect Values

**Check:**
1. Label selectors match your deployment
2. Variable values are correct
3. Recording rules are evaluating: `curl http://prometheus:9090/api/v1/rules`

### Slow Loading

**Optimize:**
1. Use recording rules for complex queries
2. Reduce time range
3. Decrease refresh rate
4. Use smaller step intervals

## Next Steps

1. Import all dashboards into Grafana
2. Configure data sources
3. Set up alerting rules
4. Customize for your deployment
5. Train team on dashboard usage
