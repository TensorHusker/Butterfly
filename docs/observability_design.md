# Butterfly Observability Architecture

## Executive Summary

This document describes the comprehensive observability platform for Butterfly, a distributed inference system with Byzantine fault tolerance. The design enables deep visibility into distributed request flows, consensus protocols, Byzantine fault detection, and performance characteristics across the cluster.

**Design Philosophy:**
- Observability is not optional - it's a first-class architectural concern
- Every distributed operation must be traceable
- Metrics, logs, and traces are tightly integrated via correlation IDs
- Debugging distributed systems must be as intuitive as debugging monoliths

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Observability Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Prometheus  │  │   Jaeger     │  │     Loki     │         │
│  │  (Metrics)   │  │  (Traces)    │  │    (Logs)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                 │                  │                  │
│         └─────────────────┴──────────────────┘                  │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │    Grafana      │                           │
│                  │  (Visualization)│                           │
│                  └─────────────────┘                           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                   Butterfly Cluster                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Instrumentation Layer (Embedded in Each Component)      │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │  │  Metrics    │  │   Tracing   │  │  Logging    │     │  │
│  │  │  Export     │  │   Export    │  │  Export     │     │  │
│  │  │ (Prom /w3c) │  │(OTLP/gRPC)  │  │  (JSON)     │     │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │  │
│  │         │                 │                 │            │  │
│  │         └─────────────────┴─────────────────┘            │  │
│  │                           │                               │  │
│  │           ┌───────────────▼───────────────┐              │  │
│  │           │   Correlation Engine          │              │  │
│  │           │   (trace_id, request_id)      │              │  │
│  │           └───────────────────────────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Coordinator  │  │ Compute Node │  │  Observers   │         │
│  │   (Raft)     │  │  (Workers)   │  │   (Watch)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Three Pillars Integration

**Metrics + Logs + Traces** are not isolated - they reference each other:

```
Request received (Log)
  trace_id: 7a8b9c0d1e2f3a4b
  request_id: req_abc123
          │
          ├──> Trace Span (inference.request)
          │      span_id: 1e2f3a4b
          │      trace_id: 7a8b9c0d1e2f3a4b
          │
          └──> Metrics
                 butterfly_requests_total{trace_id="7a8b..."}
                 butterfly_request_duration{trace_id="7a8b..."}
```

### 2. Correlation-First Design

Every observable event includes:
- `trace_id`: Distributed trace identifier (W3C standard)
- `span_id`: Current operation identifier
- `request_id`: User-visible request identifier
- `node_id`: Source node
- `consensus_term`: Consensus protocol epoch

### 3. Sampling Strategy

**Metrics**: 100% (no sampling, use aggregation)
**Logs**: 100% (use retention policies)
**Traces**:
  - Errors: 100%
  - Slow requests (p99+): 100%
  - Byzantine faults: 100%
  - Normal requests: 10%

### 4. Retention Policies

| Data Type | Hot Storage | Warm Storage | Cold Storage |
|-----------|-------------|--------------|--------------|
| Metrics   | 7 days      | 30 days      | 90 days (downsampled) |
| Traces    | 7 days      | 30 days      | 90 days (1% sample) |
| Logs      | 7 days      | 30 days      | 90 days (errors only) |

## Component Architecture

### Metrics System (Prometheus)

**Metrics Categories:**
1. Request metrics (RED: Rate, Errors, Duration)
2. Consensus protocol metrics (term, elections, commits)
3. Computation metrics (layer execution, memory, GPU)
4. Communication metrics (network I/O, latency, connections)
5. Resource utilization (CPU, memory, disk)
6. Byzantine fault detection metrics

**Key Design Decisions:**
- Use histogram buckets aligned with SLOs: [1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s]
- Limit cardinality: max 1000 node_ids, 200 layer_ids, 50 error types
- Aggregate node-level metrics to cluster-level every 1m via recording rules
- Export via `/metrics` endpoint on port 9090

See [metrics_specification.md](./metrics_specification.md) for complete metric catalog.

### Tracing System (OpenTelemetry + Jaeger)

**Trace Propagation:**
- Use W3C Trace Context standard (`traceparent` header)
- Inject into all inter-node messages (gRPC metadata, QUIC streams)
- Maintain parent-child relationships across nodes

**Critical Spans:**
- `inference.request`: Top-level inference request
- `layer.execute`: Individual layer computation
- `comm.send_tensor`: Inter-node tensor transfer
- `consensus.propose/vote/commit`: Consensus operations
- `checkpoint.create/restore`: State persistence

**Span Attributes:**
Include rich context:
```rust
span.set_attribute("model.name", "llama-70b");
span.set_attribute("layer.id", 12);
span.set_attribute("node.id", "compute-7");
span.set_attribute("tensor.shape", "[32, 128, 4096]");
span.set_attribute("consensus.term", 42);
```

**Sampling:**
- Head-based: Sample 100% of errors/slow requests, 10% normal
- Tail-based (via OTel Collector): Make sampling decision after request completes

See [tracing_design.md](./tracing_design.md) for detailed instrumentation plan.

### Logging System (Structured JSON + Loki)

**Log Format:**
- JSON structured logs for machine parsing
- Include correlation IDs in every log entry
- Use standard log levels: TRACE, DEBUG, INFO, WARN, ERROR
- Scrub PII (user IDs, input text, IPs)

**Log Aggregation:**
- Ship logs via Promtail to Loki
- Index by: level, node_id, component, trace_id
- Query via LogQL in Grafana

**Log Retention:**
- 30 days full retention
- 90 days ERROR-level only
- Archive to S3 for compliance

See [structured_logging_spec.md](./structured_logging_spec.md) for log schema.

### Health Checks

**Three Types:**
1. **Liveness**: Is process alive? (simple HTTP 200)
2. **Readiness**: Can accept new work? (check quorum, resources)
3. **Startup**: Is initialization complete? (check model loaded, network ready)

**Deep Health Checks:**
- Consensus health (leader elected, commit advancing)
- Network health (connections active, low error rate)
- Resource health (memory <95%, CPU <95%, disk >10% free)

**Endpoints:**
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /health/startup` - Startup probe
- `GET /health/deep` - Detailed component health

### Debug Console (butterfly-cli)

**Interactive REPL for debugging:**
```bash
$ butterfly-cli debug
butterfly> cluster status
butterfly> request trace req_abc123
butterfly> state dump compute-node-7
butterfly> profile start compute-node-7
butterfly> flamegraph generate compute-node-7
```

**Capabilities:**
- Cluster inspection
- Request tracing & replay
- State dump & diff
- Performance profiling (CPU, memory, GPU)
- Network debugging (packet capture, latency matrix)
- Byzantine fault verification

See [debugging_guide.md](./debugging_guide.md) for full command reference.

## Integration with Butterfly Architecture

### Coordinator Node Instrumentation

```rust
// butterfly-coordination/src/coordinator.rs
use tracing::{info_span, instrument};
use prometheus::{IntCounter, Histogram};

lazy_static! {
    static ref REQUESTS_TOTAL: IntCounter = register_int_counter!(
        "butterfly_requests_total",
        "Total inference requests"
    ).unwrap();

    static ref REQUEST_DURATION: Histogram = register_histogram!(
        "butterfly_request_duration_seconds",
        "Request duration",
        vec![0.001, 0.01, 0.1, 1.0, 10.0]
    ).unwrap();
}

#[instrument(
    name = "inference.request",
    fields(
        request_id = %request.id,
        model = %request.model,
        cluster_size = self.cluster_size()
    )
)]
pub async fn handle_inference_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
    let _timer = REQUEST_DURATION.start_timer();
    REQUESTS_TOTAL.inc();

    info!(
        request_id = %request.id,
        model = %request.model,
        "Inference request received"
    );

    let result = self.coordinate_inference(request).await;

    match &result {
        Ok(_) => info!("Inference completed successfully"),
        Err(e) => error!(error = %e, "Inference failed"),
    }

    result
}
```

### Compute Node Instrumentation

```rust
// butterfly-core/src/execution.rs
#[instrument(
    name = "layer.execute",
    fields(
        layer_id = self.layer_id,
        node_id = %self.node_id,
        device = %self.device
    )
)]
pub async fn execute_layer(&self, input: Tensor) -> Result<Tensor> {
    let _timer = LAYER_EXECUTION_DURATION
        .with_label_values(&[&self.layer_id.to_string(), &self.node_id.to_string()])
        .start_timer();

    let memory_before = self.memory_usage();

    let output = self.compute(input).await?;

    let memory_after = self.memory_usage();
    LAYER_MEMORY_USED
        .with_label_values(&[&self.layer_id.to_string()])
        .set((memory_after - memory_before) as f64);

    Ok(output)
}
```

### Communication Layer Instrumentation

```rust
// butterfly-comm/src/transport.rs
#[instrument(
    name = "comm.send_tensor",
    fields(
        from = %self.local_node_id,
        to = %peer_node_id,
        tensor_size_bytes = tensor.size_bytes()
    )
)]
pub async fn send_tensor(&self, peer_node_id: NodeId, tensor: &Tensor) -> Result<()> {
    NETWORK_BYTES_SENT
        .with_label_values(&[&self.local_node_id.to_string(), &peer_node_id.to_string()])
        .inc_by(tensor.size_bytes() as u64);

    let start = Instant::now();
    let result = self.transport.send(peer_node_id, tensor).await;
    let latency = start.elapsed();

    NETWORK_LATENCY
        .with_label_values(&[&self.local_node_id.to_string(), &peer_node_id.to_string()])
        .observe(latency.as_secs_f64());

    result
}
```

## Deployment Architecture

### Kubernetes Deployment

```yaml
# Observability namespace
apiVersion: v1
kind: Namespace
metadata:
  name: butterfly-observability

---
# Prometheus
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
  namespace: butterfly-observability
spec:
  serviceName: prometheus
  replicas: 1
  template:
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--storage.tsdb.retention.time=30d'
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
  volumeClaimTemplates:
  - metadata:
      name: storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# Jaeger (all-in-one for dev, production should use separate components)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: butterfly-observability
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        - name: SPAN_STORAGE_TYPE
          value: elasticsearch
        - name: ES_SERVER_URLS
          value: http://elasticsearch:9200
        ports:
        - containerPort: 14250  # gRPC
        - containerPort: 16686  # UI
        - containerPort: 4317   # OTLP gRPC

---
# Loki
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: loki
  namespace: butterfly-observability
spec:
  serviceName: loki
  replicas: 1
  template:
    spec:
      containers:
      - name: loki
        image: grafana/loki:latest
        args:
          - '-config.file=/etc/loki/loki.yaml'
        ports:
        - containerPort: 3100
        volumeMounts:
        - name: config
          mountPath: /etc/loki
        - name: storage
          mountPath: /loki
      volumes:
      - name: config
        configMap:
          name: loki-config
  volumeClaimTemplates:
  - metadata:
      name: storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# Grafana
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: butterfly-observability
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        env:
        - name: GF_AUTH_ANONYMOUS_ENABLED
          value: "true"
        - name: GF_AUTH_ANONYMOUS_ORG_ROLE
          value: "Admin"
        ports:
        - containerPort: 3000
        volumeMounts:
        - name: dashboards
          mountPath: /etc/grafana/provisioning/dashboards
        - name: datasources
          mountPath: /etc/grafana/provisioning/datasources
      volumes:
      - name: dashboards
        configMap:
          name: grafana-dashboards
      - name: datasources
        configMap:
          name: grafana-datasources
```

### Butterfly Pods with Observability

```yaml
# Coordinator with instrumentation
apiVersion: apps/v1
kind: Deployment
metadata:
  name: butterfly-coordinator
  namespace: butterfly
spec:
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: coordinator
        image: butterfly/coordinator:latest
        env:
        - name: RUST_LOG
          value: "info,butterfly=debug"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger.butterfly-observability:4317"
        - name: NODE_ROLE
          value: "coordinator"
        ports:
        - containerPort: 8080   # API
        - containerPort: 9090   # Metrics
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8080
          periodSeconds: 5
          failureThreshold: 60
```

## Operational Runbooks

### Investigating High Latency

1. **Check Grafana "Request Latency" dashboard**
   - Identify which percentile is elevated (p50, p99, p999)
   - Check if it's cluster-wide or specific nodes

2. **Query Prometheus for slow requests**
   ```promql
   histogram_quantile(0.99,
     rate(butterfly_request_duration_seconds_bucket[5m])
   ) > 1.0
   ```

3. **Find slow traces in Jaeger**
   - Filter by operation: `inference.request`
   - Filter by duration: `> 1s`
   - Examine span waterfall to find bottleneck

4. **Check logs for specific trace**
   ```logql
   {job="butterfly"} | json | trace_id="<trace_id>" | level="WARN" or level="ERROR"
   ```

5. **Profile the slow node**
   ```bash
   butterfly-cli debug
   > profile start <node_id>
   # wait 30s
   > flamegraph generate <node_id>
   ```

### Detecting Byzantine Faults

1. **Check Byzantine fault metrics**
   ```promql
   rate(butterfly_byzantine_faults_detected_total[5m]) > 0
   ```

2. **Identify faulty node**
   ```logql
   {job="butterfly"} | json | message=~".*fault.detected.*"
   ```

3. **Verify checkpoint integrity**
   ```bash
   butterfly-cli debug
   > verify checkpoint <checkpoint_id>
   > verify node <suspicious_node_id>
   ```

4. **Check reputation scores**
   ```promql
   butterfly_reputation_score{node_id="<node_id>"}
   ```

### Debugging Network Partition

1. **Check connection matrix**
   ```bash
   butterfly-cli debug
   > connections list
   > latency matrix
   ```

2. **Examine QUIC connection metrics**
   ```promql
   butterfly_connection_active{from_node="<node1>", to_node="<node2>"} == 0
   ```

3. **Capture packets for analysis**
   ```bash
   butterfly-cli debug
   > packet capture start <node1>,<node2>
   # reproduce issue
   > packet capture stop
   # analyze PCAP with Wireshark
   ```

4. **Check consensus log for partition evidence**
   ```logql
   {job="butterfly"}
     | json
     | component="consensus"
     | message=~".*heartbeat.*failed.*"
   ```

## Performance Considerations

### Instrumentation Overhead

**Metrics:**
- Prometheus client overhead: <0.1% CPU per node
- Metrics scrape: ~100ms every 15s (negligible)

**Tracing:**
- Span creation: ~1μs per span (negligible)
- Span export (10% sample): ~10ms per request (acceptable)

**Logging:**
- JSON serialization: ~5μs per log entry
- Async log writing: non-blocking

**Total overhead estimate: <1% of system resources**

### Scaling Limits

**Prometheus:**
- Can handle ~1M active time series
- Butterfly cluster of 100 nodes with 200 layers = ~500K series
- Headroom: 2x

**Jaeger:**
- Can handle ~10K spans/sec
- With 10% sampling and 50 spans/request at 1000 req/s = 5K spans/sec
- Headroom: 2x

**Loki:**
- Can handle ~1GB/day of logs per node
- 100 nodes = 100GB/day
- With compression: ~10GB/day stored

## Security Considerations

### PII Scrubbing

All logs scrub:
- User identifiers (hash with pepper)
- Input text (preserve length only)
- IP addresses (mask to /16)
- Tensor values (preserve shape/dtype only)

### Access Control

**Grafana:**
- Anonymous read-only access for dashboards
- Admin access requires SSO authentication

**Prometheus/Jaeger/Loki:**
- Internal cluster access only (no external exposure)
- TLS-encrypted communication

**Debug Console:**
- Requires mTLS client certificate
- Audit log of all debug commands

## Migration Path

### Phase 1: Foundation (Week 1)
- Deploy Prometheus + Grafana
- Basic metrics: requests, latency, errors
- Health check endpoints

### Phase 2: Distributed Tracing (Week 2)
- Deploy Jaeger
- Instrument critical paths (request, layer execution)
- Add trace correlation to logs

### Phase 3: Advanced Observability (Week 3)
- Deploy Loki
- Add Byzantine fault metrics
- Create debug console

### Phase 4: Production Hardening (Week 4)
- Set up alerting rules
- Create runbooks
- Load test observability stack
- Train team on debugging workflows

## References

- [Metrics Specification](./metrics_specification.md)
- [Tracing Design](./tracing_design.md)
- [Debugging Guide](./debugging_guide.md)
- [Grafana Dashboards](./dashboards/)
- [Prometheus Configuration](./prometheus.yml)
- [OpenTelemetry Collector Config](./otel-collector-config.yaml)

## Appendix: Observability Checklist

### For Every New Feature

- [ ] Add relevant metrics (rate, errors, duration)
- [ ] Instrument with tracing spans
- [ ] Log key events with correlation IDs
- [ ] Add health checks if applicable
- [ ] Update Grafana dashboards
- [ ] Add alerting rules
- [ ] Document debugging procedures

### Before Production Deployment

- [ ] Load test observability stack
- [ ] Verify metric cardinality is bounded
- [ ] Test trace sampling at scale
- [ ] Confirm log retention policies
- [ ] Verify PII scrubbing
- [ ] Test debug console commands
- [ ] Validate alerting works end-to-end
- [ ] Train on-call team on runbooks
