# Butterfly Distributed Tracing Design

## Overview

Distributed tracing enables end-to-end visibility into inference requests as they flow through the Butterfly cluster. Each request generates a trace composed of spans representing individual operations across multiple nodes.

**Goals:**
- Understand where time is spent in distributed requests
- Identify performance bottlenecks across nodes
- Debug failures in distributed workflows
- Correlate traces with logs and metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Jaeger Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Collector    │  │ Storage      │  │ Query UI     │          │
│  │ (OTLP/gRPC)  │─▶│(Elasticsearch)│◀─│   (HTTP)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ OTLP/gRPC
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                  OpenTelemetry Collector                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Receivers   │─▶│  Processors  │─▶│  Exporters   │          │
│  │   (OTLP)     │  │   (Sampling) │  │   (Jaeger)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐
│  Coordinator  │    │  Compute Node   │    │  Observer   │
│               │    │                 │    │             │
│  OTel SDK     │    │   OTel SDK      │    │  OTel SDK   │
│  (Rust)       │    │   (Rust)        │    │  (Rust)     │
└───────────────┘    └─────────────────┘    └─────────────┘
```

## Trace Context Propagation

### W3C Trace Context Standard

All inter-node communication includes W3C trace context headers:

```
traceparent: 00-{trace_id}-{span_id}-{flags}
             │   │          │         │
             │   │          │         └─ Sampled flag (01 = sampled)
             │   │          └─────────── Parent span ID (16 hex chars)
             │   └────────────────────── Trace ID (32 hex chars)
             └────────────────────────── Version (00)

tracestate: butterfly={node_id}:{hop_count}:{consensus_term}
            │         │         │          │
            │         │         │          └─ Consensus term for debugging
            │         │         └──────────── Hops through cluster
            │         └────────────────────── Originating node
            └──────────────────────────────── Vendor prefix
```

**Example:**
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate: butterfly=coordinator-1:0:42
```

### Propagation Mechanisms

#### gRPC (Tonic)

```rust
// butterfly-comm/src/grpc_client.rs
use opentelemetry::global;
use tonic::metadata::MetadataMap;
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub async fn send_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
    let span = tracing::info_span!("grpc.send_request");
    let cx = span.context();

    // Inject trace context into gRPC metadata
    let mut metadata = MetadataMap::new();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&cx, &mut MetadataInjector(&mut metadata));
    });

    let mut client = self.client.clone();
    let mut request = tonic::Request::new(request);
    *request.metadata_mut() = metadata;

    client.inference(request).await
}

// Server side: extract trace context
pub async fn handle_request(
    &self,
    request: tonic::Request<InferenceRequest>,
) -> Result<tonic::Response<InferenceResponse>, tonic::Status> {
    let parent_cx = global::get_text_map_propagator(|propagator| {
        propagator.extract(&MetadataExtractor(request.metadata()))
    });

    let span = tracing::info_span!("grpc.handle_request");
    span.set_parent(parent_cx);

    // Process request within span context
    let _enter = span.enter();
    self.process_request(request.into_inner()).await
}
```

#### QUIC Streams

```rust
// butterfly-comm/src/quic_transport.rs
use quinn::{SendStream, RecvStream};

pub async fn send_tensor(
    &self,
    mut send_stream: SendStream,
    tensor: &Tensor,
) -> Result<()> {
    let span = tracing::info_span!("quic.send_tensor");
    let cx = span.context();

    // Inject trace context into QUIC stream header
    let mut headers = HashMap::new();
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&cx, &mut HashMapInjector(&mut headers));
    });

    // Serialize headers + tensor
    let header_bytes = bincode::serialize(&headers)?;
    send_stream.write_all(&(header_bytes.len() as u32).to_be_bytes()).await?;
    send_stream.write_all(&header_bytes).await?;
    send_stream.write_all(tensor.as_bytes()).await?;

    Ok(())
}

pub async fn receive_tensor(
    &self,
    mut recv_stream: RecvStream,
) -> Result<Tensor> {
    // Extract trace context from QUIC stream header
    let header_len = recv_stream.read_u32().await? as usize;
    let mut header_bytes = vec![0u8; header_len];
    recv_stream.read_exact(&mut header_bytes).await?;

    let headers: HashMap<String, String> = bincode::deserialize(&header_bytes)?;
    let parent_cx = global::get_text_map_propagator(|propagator| {
        propagator.extract(&HashMapExtractor(&headers))
    });

    let span = tracing::info_span!("quic.receive_tensor");
    span.set_parent(parent_cx);
    let _enter = span.enter();

    // Read tensor data
    let tensor = Tensor::from_stream(&mut recv_stream).await?;
    Ok(tensor)
}
```

## Span Hierarchy

### Inference Request Flow

```
inference.request (Coordinator)
├─ coordination.distribute (Coordinator)
│  ├─ partition.analyze (Coordinator)
│  └─ schedule.assign (Coordinator)
├─ layer.execute [layer_id=0] (Compute Node 1)
│  ├─ computation.forward (Compute Node 1)
│  ├─ memory.allocate (Compute Node 1)
│  └─ comm.send_tensor → to=node2 (Compute Node 1)
├─ comm.send_tensor [from=node1] (Network)
├─ comm.receive_tensor [from=node1] (Compute Node 2)
├─ layer.execute [layer_id=1] (Compute Node 2)
│  ├─ computation.forward (Compute Node 2)
│  └─ comm.send_tensor → to=coordinator (Compute Node 2)
└─ inference.aggregate (Coordinator)
   └─ result.serialize (Coordinator)
```

### Checkpoint Creation Flow

```
checkpoint.create (Coordinator)
├─ consensus.propose (Coordinator)
│  └─ consensus.replicate → to=peers (Coordinator)
├─ consensus.vote (Peer Node 1)
│  └─ checkpoint.verify (Peer Node 1)
├─ consensus.vote (Peer Node 2)
│  └─ checkpoint.verify (Peer Node 2)
├─ consensus.commit (Coordinator)
├─ state.snapshot (All Nodes)
│  ├─ model.serialize (All Nodes)
│  └─ memory.dump (All Nodes)
└─ storage.write (All Nodes)
   ├─ disk.write (All Nodes)
   └─ merkle.compute (All Nodes)
```

## Span Definitions

### Top-Level Spans

#### inference.request
**Location:** Coordinator
**Lifecycle:** Request received → Response sent
**Attributes:**
```rust
span.set_attribute("model.name", "llama-70b");
span.set_attribute("model.version", "v1.0");
span.set_attribute("input.token_count", 128);
span.set_attribute("input.sequence_length", 2048);
span.set_attribute("cluster.size", 10);
span.set_attribute("cluster.state", "healthy");
span.set_attribute("consensus.term", 42);
span.set_attribute("request.priority", "normal");
```

**Events:**
```rust
span.add_event("request.validated", attributes! {
    "validation.duration_ms" => 1.2,
});
span.add_event("partitioning.completed", attributes! {
    "partition.count" => 10,
    "partition.strategy" => "layer_wise",
});
span.add_event("response.ready", attributes! {
    "output.token_count" => 256,
});
```

#### coordination.distribute
**Location:** Coordinator
**Lifecycle:** Partitioning begins → All work assigned
**Attributes:**
```rust
span.set_attribute("partition.strategy", "layer_wise");
span.set_attribute("partition.count", 10);
span.set_attribute("nodes.assigned", format!("{:?}", node_ids));
span.set_attribute("load_balancing.algorithm", "round_robin");
```

#### layer.execute
**Location:** Compute Node
**Lifecycle:** Layer computation starts → Output ready
**Attributes:**
```rust
span.set_attribute("layer.id", 12);
span.set_attribute("layer.type", "transformer_block");
span.set_attribute("node.id", "compute-node-7");
span.set_attribute("input.shape", "[32, 128, 4096]");
span.set_attribute("output.shape", "[32, 128, 4096]");
span.set_attribute("execution.device", "cuda:0");
span.set_attribute("memory.peak_bytes", 4_294_967_296);
span.set_attribute("flops.estimated", 8_796_093_022_208);
```

**Events:**
```rust
span.add_event("computation.started", attributes! {
    "device.id" => "cuda:0",
});
span.add_event("memory.allocated", attributes! {
    "bytes" => 4_294_967_296,
    "pool" => "tensor",
});
span.add_event("kernel.launched", attributes! {
    "kernel.name" => "matmul_kernel",
    "grid_size" => "[1024, 1024, 1]",
});
span.add_event("computation.completed", attributes! {
    "flops.actual" => 8_796_093_022_208,
});
```

### Communication Spans

#### comm.send_tensor
**Location:** Source Node
**Lifecycle:** Serialization starts → Acknowledgment received
**Attributes:**
```rust
span.set_attribute("from.node_id", "compute-node-1");
span.set_attribute("to.node_id", "compute-node-2");
span.set_attribute("tensor.size_bytes", 16_777_216);
span.set_attribute("tensor.shape", "[32, 128, 4096]");
span.set_attribute("tensor.dtype", "float32");
span.set_attribute("protocol", "quic");
span.set_attribute("compression.enabled", true);
span.set_attribute("compression.algorithm", "zstd");
span.set_attribute("compression.ratio", 2.4);
```

**Events:**
```rust
span.add_event("serialization.started");
span.add_event("compression.applied", attributes! {
    "ratio" => 2.4,
    "compressed_bytes" => 6_990_506,
});
span.add_event("network.sent", attributes! {
    "bytes" => 6_990_506,
});
span.add_event("acknowledgment.received", attributes! {
    "latency_ms" => 5.2,
});
```

#### comm.receive_tensor
**Location:** Destination Node
**Lifecycle:** Data arrival → Deserialization complete
**Attributes:**
```rust
span.set_attribute("from.node_id", "compute-node-1");
span.set_attribute("to.node_id", "compute-node-2");
span.set_attribute("bytes.received", 6_990_506);
span.set_attribute("decompression.enabled", true);
```

**Events:**
```rust
span.add_event("network.received", attributes! {
    "bytes" => 6_990_506,
});
span.add_event("decompression.applied", attributes! {
    "decompressed_bytes" => 16_777_216,
});
span.add_event("deserialization.completed");
```

### Consensus Spans

#### consensus.propose
**Location:** Leader Node
**Lifecycle:** Proposal created → Replicated to peers
**Attributes:**
```rust
span.set_attribute("consensus.term", 42);
span.set_attribute("log.index", 1234);
span.set_attribute("proposal.type", "checkpoint");
span.set_attribute("proposer.node_id", "coordinator-1");
```

#### consensus.vote
**Location:** Peer Node
**Lifecycle:** Proposal received → Vote sent
**Attributes:**
```rust
span.set_attribute("consensus.term", 42);
span.set_attribute("log.index", 1234);
span.set_attribute("voter.node_id", "coordinator-2");
span.set_attribute("vote.granted", true);
```

#### consensus.commit
**Location:** Leader Node
**Lifecycle:** Quorum reached → Entry committed
**Attributes:**
```rust
span.set_attribute("consensus.term", 42);
span.set_attribute("log.index", 1234);
span.set_attribute("commit.duration_ms", 12.4);
span.set_attribute("votes.received", 3);
span.set_attribute("quorum.size", 2);
```

### Checkpoint Spans

#### checkpoint.create
**Location:** All Nodes
**Lifecycle:** Snapshot initiated → Checkpoint verified
**Attributes:**
```rust
span.set_attribute("checkpoint.id", "ckpt_20251011_213045");
span.set_attribute("checkpoint.size_bytes", 10_737_418_240);
span.set_attribute("state.hash", "a1b2c3d4...");  // Merkle root
span.set_attribute("snapshot.layer_count", 80);
```

#### checkpoint.restore
**Location:** Recovering Node
**Lifecycle:** Checkpoint loaded → State verified
**Attributes:**
```rust
span.set_attribute("checkpoint.id", "ckpt_20251011_213045");
span.set_attribute("restore.source", "peer_node");  // or "local", "observer"
span.set_attribute("verification.passed", true);
```

## Span Links

Use span links to connect related spans across the cluster:

```rust
// Byzantine fault detection: link faulty node's span to verification span
let faulty_span_ctx = faulty_node_span.context();
let verification_span = tracing::info_span!("verification.byzantine_check");
verification_span.add_link(faulty_span_ctx.span().span_context().clone());

// Consensus: link all vote spans to commit span
let vote_spans: Vec<SpanContext> = votes.iter()
    .map(|v| v.span_context.clone())
    .collect();
let commit_span = tracing::info_span!("consensus.commit");
for vote_span_ctx in vote_spans {
    commit_span.add_link(vote_span_ctx);
}
```

## Sampling Strategy

### Head-Based Sampling (SDK)

Implemented in the tracer provider:

```rust
// butterfly-core/src/telemetry.rs
use opentelemetry::sdk::trace::{Sampler, SamplerResult};

struct ButterflyParentBasedSampler {
    default_sampler: Box<dyn Sampler>,
}

impl Sampler for ButterflyParentBasedSampler {
    fn should_sample(
        &self,
        parent_context: Option<&Context>,
        trace_id: TraceId,
        name: &str,
        span_kind: &SpanKind,
        attributes: &[KeyValue],
        links: &[Link],
    ) -> SamplerResult {
        // Always sample if parent is sampled
        if let Some(parent_cx) = parent_context {
            if parent_cx.span().span_context().is_sampled() {
                return SamplerResult {
                    decision: SamplingDecision::RecordAndSample,
                    attributes: Vec::new(),
                    trace_state: TraceState::default(),
                };
            }
        }

        // Always sample errors
        if attributes.iter().any(|kv| {
            kv.key == "error" && kv.value.as_str() == Some("true")
        }) {
            return SamplerResult {
                decision: SamplingDecision::RecordAndSample,
                attributes: Vec::new(),
                trace_state: TraceState::default(),
            };
        }

        // Always sample Byzantine faults
        if attributes.iter().any(|kv| {
            kv.key == "fault.detected" && kv.value.as_str() == Some("true")
        }) {
            return SamplerResult {
                decision: SamplingDecision::RecordAndSample,
                attributes: Vec::new(),
                trace_state: TraceState::default(),
            };
        }

        // Otherwise use default sampler (10%)
        self.default_sampler.should_sample(
            parent_context,
            trace_id,
            name,
            span_kind,
            attributes,
            links,
        )
    }
}

pub fn init_tracing() -> Result<()> {
    let sampler = ButterflyParentBasedSampler {
        default_sampler: Box::new(opentelemetry::sdk::trace::Sampler::TraceIdRatioBased(0.1)),
    };

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://otel-collector:4317")
        )
        .with_trace_config(
            opentelemetry::sdk::trace::config()
                .with_sampler(sampler)
                .with_resource(Resource::new(vec![
                    KeyValue::new("service.name", "butterfly"),
                    KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ]))
        )
        .install_batch(opentelemetry::runtime::Tokio)?;

    Ok(())
}
```

### Tail-Based Sampling (Collector)

Implemented in OpenTelemetry Collector:

```yaml
# /docs/otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  # Batch spans for efficiency
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Tail-based sampling
  tail_sampling:
    decision_wait: 10s  # Wait for all spans in trace
    num_traces: 100000  # Keep in memory
    policies:
      # Always keep errors
      - name: errors
        type: status_code
        status_code:
          status_codes:
            - ERROR

      # Always keep slow requests (>5s)
      - name: slow_requests
        type: latency
        latency:
          threshold_ms: 5000

      # Always keep Byzantine faults
      - name: byzantine_faults
        type: attribute
        attribute:
          key: fault.detected
          values:
            - "true"

      # Always keep consensus failures
      - name: consensus_failures
        type: string_attribute
        string_attribute:
          key: consensus.status
          values:
            - "failed"
            - "timeout"

      # Sample 10% of normal requests
      - name: sample_normal
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  # Also export to Prometheus for trace metrics
  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: butterfly
    const_labels:
      cluster: prod

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, tail_sampling]
      exporters: [jaeger, prometheus]
```

## Jaeger Deployment

```yaml
# /docs/jaeger-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: butterfly-observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        env:
        # Enable OTLP receiver
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"

        # Storage backend
        - name: SPAN_STORAGE_TYPE
          value: elasticsearch

        - name: ES_SERVER_URLS
          value: http://elasticsearch:9200

        # Sampling
        - name: SAMPLING_STRATEGIES_FILE
          value: /etc/jaeger/sampling.json

        ports:
        # Jaeger UI
        - containerPort: 16686
          name: ui

        # OTLP gRPC
        - containerPort: 4317
          name: otlp-grpc

        # OTLP HTTP
        - containerPort: 4318
          name: otlp-http

        # Jaeger gRPC
        - containerPort: 14250
          name: jaeger-grpc

        volumeMounts:
        - name: sampling-config
          mountPath: /etc/jaeger

        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

      volumes:
      - name: sampling-config
        configMap:
          name: jaeger-sampling-config

---
apiVersion: v1
kind: Service
metadata:
  name: jaeger
  namespace: butterfly-observability
spec:
  selector:
    app: jaeger
  ports:
  - name: ui
    port: 16686
    targetPort: 16686
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: jaeger-grpc
    port: 14250
    targetPort: 14250

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-sampling-config
  namespace: butterfly-observability
data:
  sampling.json: |
    {
      "default_strategy": {
        "type": "probabilistic",
        "param": 0.1
      },
      "service_strategies": [
        {
          "service": "butterfly",
          "type": "probabilistic",
          "param": 0.1
        }
      ]
    }
```

## Querying Traces

### Jaeger UI

Access at `http://jaeger:16686`

**Find slow requests:**
- Service: butterfly
- Operation: inference.request
- Min Duration: 5s
- Limit: 20

**Find errors:**
- Service: butterfly
- Tags: error=true
- Limit: 100

**Find Byzantine faults:**
- Service: butterfly
- Tags: fault.detected=true

### Programmatic Access (Jaeger API)

```rust
// butterfly-cli/src/jaeger_client.rs
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct JaegerTrace {
    trace_id: String,
    spans: Vec<JaegerSpan>,
}

#[derive(Deserialize)]
struct JaegerSpan {
    span_id: String,
    operation_name: String,
    duration: u64,  // microseconds
    tags: Vec<Tag>,
}

pub struct JaegerClient {
    client: Client,
    base_url: String,
}

impl JaegerClient {
    pub async fn find_slow_traces(&self, min_duration_ms: u64) -> Result<Vec<JaegerTrace>> {
        let url = format!(
            "{}/api/traces?service=butterfly&operation=inference.request&minDuration={}ms",
            self.base_url,
            min_duration_ms
        );

        let response = self.client.get(&url).send().await?;
        let traces: Vec<JaegerTrace> = response.json().await?;
        Ok(traces)
    }

    pub async fn get_trace(&self, trace_id: &str) -> Result<JaegerTrace> {
        let url = format!("{}/api/traces/{}", self.base_url, trace_id);
        let response = self.client.get(&url).send().await?;
        let trace: JaegerTrace = response.json().await?;
        Ok(trace)
    }
}
```

## Integration with Logs and Metrics

### Logs → Traces

Every log entry includes trace_id and span_id:

```json
{
  "timestamp": "2025-10-11T21:30:45.123Z",
  "level": "INFO",
  "message": "Layer execution completed",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "node_id": "compute-node-7",
  "layer_id": 12
}
```

**In Loki:** Query logs for a specific trace:
```logql
{job="butterfly"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

**In Grafana:** Click trace_id in log entry → Jump to Jaeger trace

### Traces → Metrics

Export span metrics to Prometheus via OTel Collector:

```yaml
exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: butterfly_traces
    const_labels:
      cluster: prod

# Automatically generates metrics:
# butterfly_traces_duration_milliseconds{operation, status_code}
# butterfly_traces_calls_total{operation, status_code}
```

Query trace-derived metrics:
```promql
# P99 latency by operation
histogram_quantile(0.99,
  rate(butterfly_traces_duration_milliseconds_bucket[5m])
) by (operation)

# Error rate by operation
rate(butterfly_traces_calls_total{status_code="ERROR"}[5m])
  / rate(butterfly_traces_calls_total[5m])
```

### Metrics → Traces

Add trace exemplars to metrics:

```rust
// Record metric with trace exemplar
LAYER_EXECUTION_DURATION
    .with_label_values(&[&layer_id.to_string()])
    .observe_with_exemplar(
        duration.as_secs_f64(),
        &[("trace_id", current_trace_id().to_string())],
    );
```

**In Grafana:** Hover over metric data point → See exemplar trace_id → Click to jump to trace

## Performance Considerations

### Overhead

**Span Creation:** ~1 microsecond per span
**Span Export:** ~10ms per batch (1024 spans)
**Network:** ~100KB per 1000 spans (compressed)

**Total:** <0.5% overhead at 10% sampling rate

### Scaling

**Jaeger Capacity:**
- Collector: 10K spans/sec per instance
- Storage (Elasticsearch): 10K spans/sec write throughput
- Query: <100ms for trace lookup

**Butterfly at 1000 req/sec:**
- 50 spans/request = 50K spans/sec generated
- 10% sampling = 5K spans/sec stored
- Comfortable margin (2x) under Jaeger capacity

### Storage

**Retention:**
- 7 days hot (Elasticsearch): ~50GB
- 30 days warm (S3): ~200GB
- 90 days cold (S3 Glacier, 1% sample): ~60GB

## Troubleshooting

### Missing Spans

**Symptom:** Spans missing from traces

**Causes:**
1. Sampling dropped the span
2. Trace context not propagated
3. Span export failed
4. Collector dropped spans

**Debug:**
```bash
# Check if span was created
grep "span.created" /var/log/butterfly/*.log | grep <trace_id>

# Check if span was exported
grep "span.exported" /var/log/butterfly/*.log | grep <trace_id>

# Check collector metrics
curl http://otel-collector:8888/metrics | grep dropped
```

### Incomplete Traces

**Symptom:** Trace has gaps or orphaned spans

**Causes:**
1. Network partition during trace
2. Node crash before span export
3. Clock skew between nodes

**Debug:**
```bash
# Check for network issues
butterfly-cli debug
> connections list
> latency matrix

# Check for clock skew
butterfly_synchronization_skew_seconds
```

### High Latency in Traces

**Symptom:** Trace shows high latency but no slow spans

**Cause:** Time spent between spans (e.g., waiting in queues)

**Debug:**
```bash
# Analyze span gaps
butterfly-cli trace analyze <trace_id>

# Check queue metrics
butterfly_tasks_queued
butterfly_pipeline_stalls_total
```

## Next Steps

1. Implement tracing in butterfly-core (OTel SDK initialization)
2. Add span instrumentation to all critical paths
3. Deploy OpenTelemetry Collector
4. Deploy Jaeger with Elasticsearch backend
5. Configure sampling strategies
6. Integrate with Grafana
7. Load test to validate overhead is acceptable
