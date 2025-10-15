//! # Butterfly Metrics
//!
//! Performance monitoring and metrics collection for the Butterfly distributed inference system.
//! Tracks latency, throughput, resource utilization, and system health.

use butterfly_core::NodeId;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Metrics for a single inference task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub task_id: u64,
    pub latency_ms: f64,
    pub throughput_items_per_sec: f64,
    pub nodes_involved: Vec<NodeId>,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_tasks_completed: u64,
    pub average_latency_ms: f64,
    pub peak_throughput: f64,
    pub active_nodes: usize,
}

/// Trait for collecting and aggregating metrics
pub trait MetricsCollector: Send + Sync {
    /// Record the start of a task
    fn start_task(&mut self, task_id: u64);

    /// Record the completion of a task
    fn complete_task(&mut self, task_id: u64, nodes: Vec<NodeId>);

    /// Get metrics for a specific task
    fn task_metrics(&self, task_id: u64) -> Option<TaskMetrics>;

    /// Get aggregated system-wide metrics
    fn system_metrics(&self) -> SystemMetrics;
}

/// In-memory metrics collector implementation
pub struct InMemoryCollector {
    task_timings: std::collections::HashMap<u64, Instant>,
    completed_tasks: Vec<TaskMetrics>,
}

impl InMemoryCollector {
    pub fn new() -> Self {
        Self {
            task_timings: std::collections::HashMap::new(),
            completed_tasks: Vec::new(),
        }
    }
}

impl Default for InMemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector for InMemoryCollector {
    fn start_task(&mut self, task_id: u64) {
        self.task_timings.insert(task_id, Instant::now());
    }

    fn complete_task(&mut self, task_id: u64, nodes: Vec<NodeId>) {
        if let Some(start_time) = self.task_timings.remove(&task_id) {
            let latency = start_time.elapsed();
            let metrics = TaskMetrics {
                task_id,
                latency_ms: latency.as_secs_f64() * 1000.0,
                throughput_items_per_sec: 1.0 / latency.as_secs_f64(),
                nodes_involved: nodes,
            };
            self.completed_tasks.push(metrics);
        }
    }

    fn task_metrics(&self, task_id: u64) -> Option<TaskMetrics> {
        self.completed_tasks
            .iter()
            .find(|m| m.task_id == task_id)
            .cloned()
    }

    fn system_metrics(&self) -> SystemMetrics {
        let total_tasks = self.completed_tasks.len() as u64;

        let average_latency = if total_tasks > 0 {
            self.completed_tasks.iter().map(|m| m.latency_ms).sum::<f64>() / total_tasks as f64
        } else {
            0.0
        };

        let peak_throughput = self
            .completed_tasks
            .iter()
            .map(|m| m.throughput_items_per_sec)
            .fold(0.0f64, |a, b| a.max(b));

        SystemMetrics {
            total_tasks_completed: total_tasks,
            average_latency_ms: average_latency,
            peak_throughput,
            active_nodes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_metrics_collection() {
        let mut collector = InMemoryCollector::new();

        collector.start_task(1);
        sleep(Duration::from_millis(10));
        collector.complete_task(1, vec![NodeId(0)]);

        let metrics = collector.task_metrics(1).unwrap();
        assert!(metrics.latency_ms >= 10.0);

        let system = collector.system_metrics();
        assert_eq!(system.total_tasks_completed, 1);
    }
}
