# Butterfly Observability - Quick Start Guide

This guide will get you up and running with Butterfly's observability stack in 30 minutes.

## Prerequisites

- Kubernetes cluster (or docker-compose)
- kubectl configured
- Helm 3+ installed

## 5-Minute Setup (Development)

For local development and testing:

### 1. Start Observability Stack (Docker Compose)

```bash
cd /Users/tensorhusker/Git/Butterfly
cat > docker-compose.observability.yml <<EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docs/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./docs/dashboards:/etc/grafana/dashboards

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki

volumes:
  prometheus-data:
  grafana-data:
  loki-data:
EOF

docker-compose -f docker-compose.observability.yml up -d
```

### 2. Verify Stack is Running

```bash
# Check all services
docker-compose -f docker-compose.observability.yml ps

# Should see:
# - prometheus (port 9090)
# - grafana (port 3000)
# - jaeger (port 16686)
# - loki (port 3100)
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (no login required)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### 4. Import Butterfly Dashboard

1. Open Grafana: http://localhost:3000
2. Click "+" → "Import"
3. Upload: `/Users/tensorhusker/Git/Butterfly/docs/dashboards/overview.json`
4. Select Prometheus as data source
5. Click "Import"

### 5. Configure Butterfly to Export Metrics

Add to your Butterfly configuration:

```toml
# butterfly.toml
[observability]
metrics_enabled = true
metrics_port = 9090

tracing_enabled = true
tracing_endpoint = "http://localhost:4317"
tracing_sample_rate = 0.1

logging_format = "json"
logging_level = "info"
```

### 6. Start Butterfly and Generate Traffic

```bash
# Start Butterfly coordinator
cargo run --bin butterfly-coordinator

# In another terminal, send test requests
for i in {1..100}; do
  curl -X POST http://localhost:8080/inference \
    -H "Content-Type: application/json" \
    -d '{"model": "test-model", "input": "test input"}'
  sleep 0.1
done
```

### 7. View Observability Data

**Metrics in Prometheus:**
```bash
# Open Prometheus: http://localhost:9090
# Query: butterfly_requests_total
# Should see request counts
```

**Dashboard in Grafana:**
```bash
# Open Grafana: http://localhost:3000
# Navigate to: Dashboards → Butterfly Overview
# Should see request rate, latency, etc.
```

**Traces in Jaeger:**
```bash
# Open Jaeger: http://localhost:16686
# Service: butterfly
# Operation: inference.request
# Click "Find Traces"
# Should see trace list
```

---

## 30-Minute Setup (Production)

For production Kubernetes deployment:

### 1. Create Observability Namespace

```bash
kubectl create namespace butterfly-observability
```

### 2. Deploy Prometheus

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace butterfly-observability \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi

# Wait for ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus \
  -n butterfly-observability --timeout=300s
```

### 3. Deploy Jaeger

```bash
# Add Jaeger Helm repo
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install Jaeger
helm install jaeger jaegertracing/jaeger \
  --namespace butterfly-observability \
  --set collector.service.otlp.grpc.name=otlp-grpc \
  --set collector.service.otlp.grpc.port=4317 \
  --set storage.type=elasticsearch \
  --set storage.elasticsearch.host=elasticsearch.butterfly-observability.svc \
  --set storage.elasticsearch.port=9200

# Wait for ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=jaeger \
  -n butterfly-observability --timeout=300s
```

### 4. Deploy Loki

```bash
# Add Grafana Helm repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Loki
helm install loki grafana/loki-stack \
  --namespace butterfly-observability \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=100Gi

# Wait for ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=loki \
  -n butterfly-observability --timeout=300s
```

### 5. Deploy Grafana

```bash
# Grafana is included in kube-prometheus-stack, but configure it
kubectl port-forward -n butterfly-observability \
  svc/prometheus-grafana 3000:80 &

# Get admin password
kubectl get secret -n butterfly-observability prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode
```

### 6. Configure Prometheus Scraping

```bash
# Create ServiceMonitor for Butterfly
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: butterfly-metrics
  namespace: butterfly
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: butterfly
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
EOF
```

### 7. Import Dashboards

```bash
# Create ConfigMap with dashboard
kubectl create configmap butterfly-dashboard \
  --from-file=/Users/tensorhusker/Git/Butterfly/docs/dashboards/overview.json \
  -n butterfly-observability

# Grafana will auto-discover dashboards from ConfigMaps
```

### 8. Configure Alerting

```bash
# Create PrometheusRule for alerts
kubectl apply -f /Users/tensorhusker/Git/Butterfly/docs/dashboards/alerts/alerting-rules.yaml
```

### 9. Deploy Butterfly with Observability

```yaml
# butterfly-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: butterfly-coordinator
  namespace: butterfly
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: butterfly
        component: coordinator
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
          value: "http://jaeger-collector.butterfly-observability:4317"
        - name: METRICS_PORT
          value: "9090"
        ports:
        - name: api
          containerPort: 8080
        - name: metrics
          containerPort: 9090
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
```

```bash
kubectl apply -f butterfly-deployment.yaml
```

### 10. Verify Everything Works

```bash
# Check Prometheus is scraping
kubectl port-forward -n butterfly-observability svc/prometheus-kube-prometheus-prometheus 9090:9090 &
open http://localhost:9090/targets
# Should see butterfly targets

# Check Grafana dashboards
kubectl port-forward -n butterfly-observability svc/prometheus-grafana 3000:80 &
open http://localhost:3000
# Navigate to Butterfly Overview dashboard

# Check Jaeger is receiving traces
kubectl port-forward -n butterfly-observability svc/jaeger-query 16686:16686 &
open http://localhost:16686
# Search for service: butterfly

# Generate test traffic
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- sh
curl -X POST http://butterfly-coordinator.butterfly:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "input": "test"}'
```

---

## Common Issues

### Prometheus not scraping metrics

**Symptoms:** No data in Grafana dashboards

**Fix:**
```bash
# Check ServiceMonitor
kubectl get servicemonitor -n butterfly

# Check Prometheus targets
kubectl port-forward -n butterfly-observability svc/prometheus-kube-prometheus-prometheus 9090:9090
open http://localhost:9090/targets

# Check if metrics endpoint is accessible
kubectl port-forward -n butterfly svc/butterfly-coordinator 9090:9090
curl http://localhost:9090/metrics
```

### Jaeger not receiving traces

**Symptoms:** No traces in Jaeger UI

**Fix:**
```bash
# Check Jaeger collector logs
kubectl logs -n butterfly-observability -l app.kubernetes.io/component=collector

# Verify OTLP endpoint is accessible
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
telnet jaeger-collector.butterfly-observability 4317

# Check Butterfly is exporting traces
kubectl logs -n butterfly -l component=coordinator | grep "trace"
```

### Loki not receiving logs

**Symptoms:** No logs in Grafana

**Fix:**
```bash
# Check Promtail (log shipper) is running
kubectl get pods -n butterfly-observability -l app=promtail

# Check Promtail is scraping Butterfly pods
kubectl logs -n butterfly-observability -l app=promtail | grep butterfly

# Verify logs are being shipped
curl -X POST "http://loki.butterfly-observability:3100/loki/api/v1/query" \
  --data-urlencode 'query={job="butterfly"}'
```

---

## Next Steps

### 1. Explore the Observability Stack

**Prometheus:**
- Query metrics: http://localhost:9090
- Try queries from [metrics_specification.md](./metrics_specification.md)

**Grafana:**
- View dashboards: http://localhost:3000
- Explore time series, zoom, correlate

**Jaeger:**
- View traces: http://localhost:16686
- Analyze slow requests
- Follow trace flows

### 2. Set Up Alerting

```bash
# Configure Alertmanager
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: butterfly-alerts
  namespace: butterfly-observability
spec:
  groups: $(cat /Users/tensorhusker/Git/Butterfly/docs/dashboards/alerts/alerting-rules.yaml | yq eval -o=json)
EOF
```

### 3. Install Debug Console

```bash
# Build debug console
cargo build --release --bin butterfly-cli

# Install
cp target/release/butterfly-cli /usr/local/bin/

# Connect
butterfly-cli debug --coordinator localhost:8080
```

### 4. Read Full Documentation

- **[Observability Design](./observability_design.md)** - Complete architecture
- **[Metrics Specification](./metrics_specification.md)** - All metrics
- **[Tracing Design](./tracing_design.md)** - Tracing details
- **[Debugging Guide](./debugging_guide.md)** - Debugging procedures

### 5. Customize for Your Environment

- Adjust retention policies
- Configure alerting channels (PagerDuty, Slack)
- Create custom dashboards
- Add team-specific metrics

---

## Troubleshooting Commands

```bash
# Check all observability pods
kubectl get pods -n butterfly-observability

# Check Butterfly pods
kubectl get pods -n butterfly

# View Prometheus config
kubectl get configmap -n butterfly-observability prometheus-kube-prometheus-prometheus-rulefiles-0 -o yaml

# View Grafana dashboards
kubectl get configmap -n butterfly-observability -l grafana_dashboard=1

# Test metrics endpoint
kubectl port-forward -n butterfly svc/butterfly-coordinator 9090:9090
curl http://localhost:9090/metrics

# View logs
kubectl logs -n butterfly -l component=coordinator --tail=100

# Debug trace propagation
kubectl logs -n butterfly -l component=coordinator | grep trace_id
```

---

## Quick Reference

### URLs (Local Development)
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686
- Loki: http://localhost:3100

### URLs (Kubernetes - Port Forward)
```bash
kubectl port-forward -n butterfly-observability svc/prometheus-grafana 3000:80
kubectl port-forward -n butterfly-observability svc/prometheus-kube-prometheus-prometheus 9090:9090
kubectl port-forward -n butterfly-observability svc/jaeger-query 16686:16686
kubectl port-forward -n butterfly-observability svc/loki 3100:3100
```

### Key Metrics
```promql
# Request rate
sum(rate(butterfly_requests_total[5m]))

# Error rate
sum(rate(butterfly_request_errors_total[5m])) / sum(rate(butterfly_requests_total[5m]))

# P99 latency
histogram_quantile(0.99, sum(rate(butterfly_request_duration_seconds_bucket[5m])) by (le))

# Active nodes
butterfly_cluster_size{state="active"}
```

### Debug Console Commands
```bash
butterfly> cluster status          # Cluster health
butterfly> request trace <id>      # Trace request
butterfly> node inspect <node>     # Node details
butterfly> profile start <node>    # Start profiling
```

---

## Support

- Documentation: `/Users/tensorhusker/Git/Butterfly/docs/`
- Issues: https://github.com/tensorhusker/butterfly/issues
- Slack: #butterfly-support

## Congratulations!

You now have a fully functional observability stack for Butterfly. Start exploring the dashboards, traces, and logs to gain deep insights into your distributed inference system.

**Happy Debugging! 🦋**
