//! Adaptive φ-accrual failure detector implementation
//!
//! Based on "The φ Accrual Failure Detector" by Hayashibara et al. (SRDS 2004)
//!
//! Unlike traditional binary failure detectors, φ-accrual provides a continuous
//! suspicion level that adapts to network conditions.

use butterfly_core::NodeId;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Maximum heartbeat history to maintain per node
const MAX_HISTORY_SIZE: usize = 100;

/// Minimum samples before making failure decisions
const MIN_SAMPLES: usize = 10;

/// Phi-accrual failure detector
#[derive(Debug)]
pub struct PhiAccrualFailureDetector {
    /// Base heartbeat interval in milliseconds
    base_interval_ms: u64,
    /// φ threshold for suspecting failure
    phi_suspect_threshold: f64,
    /// φ threshold for confirming failure
    phi_failed_threshold: f64,
    /// Heartbeat arrival times per node
    heartbeat_history: HashMap<NodeId, VecDeque<Instant>>,
    /// Computed φ values per node
    phi_values: HashMap<NodeId, f64>,
    /// Nodes currently suspected
    suspected: HashMap<NodeId, Instant>,
    /// Nodes confirmed failed
    failed: HashMap<NodeId, Instant>,
}

impl PhiAccrualFailureDetector {
    /// Create new failure detector
    pub fn new(
        base_interval_ms: u64,
        phi_suspect_threshold: f64,
        phi_failed_threshold: f64,
    ) -> Self {
        Self {
            base_interval_ms,
            phi_suspect_threshold,
            phi_failed_threshold,
            heartbeat_history: HashMap::new(),
            phi_values: HashMap::new(),
            suspected: HashMap::new(),
            failed: HashMap::new(),
        }
    }

    /// Record heartbeat arrival from node
    pub fn record_heartbeat(&mut self, node_id: NodeId) {
        let now = Instant::now();

        let history = self
            .heartbeat_history
            .entry(node_id)
            .or_insert_with(VecDeque::new);

        // Add new heartbeat
        history.push_back(now);

        // Limit history size
        if history.len() > MAX_HISTORY_SIZE {
            history.pop_front();
        }

        // Clear suspicion if node was suspected
        if self.suspected.remove(&node_id).is_some() {
            debug!(?node_id, "Node recovered from suspected failure");
        }

        // Update φ value
        self.update_phi(node_id);
    }

    /// Update φ value for a node
    fn update_phi(&mut self, node_id: NodeId) {
        let history = match self.heartbeat_history.get(&node_id) {
            Some(h) if h.len() >= MIN_SAMPLES => h,
            _ => return, // Not enough samples yet
        };

        let now = Instant::now();
        let last_heartbeat = history.back().unwrap();
        let time_since_last = now.duration_since(*last_heartbeat);

        // Calculate inter-arrival times
        let intervals: Vec<Duration> = history
            .iter()
            .zip(history.iter().skip(1))
            .map(|(t1, t2)| t2.duration_since(*t1))
            .collect();

        if intervals.is_empty() {
            return;
        }

        // Compute mean and standard deviation
        let mean = intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>()
            / intervals.len() as f64;

        let variance = intervals
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / intervals.len() as f64;

        let std_dev = variance.sqrt();

        // Calculate φ value
        // φ = -log₁₀(P(T_now - T_last > threshold))
        // Using normal distribution CDF
        let z_score = (time_since_last.as_secs_f64() - mean) / std_dev.max(0.001);
        let phi = -normal_cdf(z_score).log10();

        self.phi_values.insert(node_id, phi);

        // Check thresholds
        if phi > self.phi_failed_threshold && !self.failed.contains_key(&node_id) {
            warn!(?node_id, phi, "Node confirmed as failed");
            self.failed.insert(node_id, now);
            self.suspected.remove(&node_id);
        } else if phi > self.phi_suspect_threshold
            && !self.suspected.contains_key(&node_id)
            && !self.failed.contains_key(&node_id)
        {
            warn!(?node_id, phi, "Node suspected of failure");
            self.suspected.insert(node_id, now);
        }
    }

    /// Get current φ value for node
    pub fn phi(&self, node_id: NodeId) -> Option<f64> {
        self.phi_values.get(&node_id).copied()
    }

    /// Check if node is suspected of failure
    pub fn is_suspected(&self, node_id: NodeId) -> bool {
        self.suspected.contains_key(&node_id)
    }

    /// Check if node is confirmed failed
    pub fn is_failed(&self, node_id: NodeId) -> bool {
        self.failed.contains_key(&node_id)
    }

    /// Get all suspected nodes
    pub fn suspected_nodes(&self) -> Vec<NodeId> {
        self.suspected.keys().copied().collect()
    }

    /// Get all failed nodes
    pub fn failed_nodes(&self) -> Vec<NodeId> {
        self.failed.keys().copied().collect()
    }

    /// Get adaptive heartbeat interval for node
    pub fn adaptive_interval(&self, node_id: NodeId) -> Duration {
        let history = match self.heartbeat_history.get(&node_id) {
            Some(h) if h.len() >= 2 => h,
            _ => return Duration::from_millis(self.base_interval_ms),
        };

        // Calculate recent interval volatility
        let recent_intervals: Vec<Duration> = history
            .iter()
            .rev()
            .take(10)
            .zip(history.iter().rev().skip(1))
            .map(|(t2, t1)| t2.duration_since(*t1))
            .collect();

        if recent_intervals.is_empty() {
            return Duration::from_millis(self.base_interval_ms);
        }

        let mean = recent_intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>()
            / recent_intervals.len() as f64;

        // Add congestion factor based on interval variance
        let congestion_factor = mean / (self.base_interval_ms as f64 / 1000.0);

        let adaptive_ms = self.base_interval_ms as f64 * (1.0 + congestion_factor.max(0.0));

        Duration::from_millis(adaptive_ms as u64)
    }

    /// Manually mark node as failed (for testing)
    pub fn mark_failed(&mut self, node_id: NodeId) {
        self.failed.insert(node_id, Instant::now());
        self.suspected.remove(&node_id);
    }

    /// Clear failure status (node recovered)
    pub fn clear_failure(&mut self, node_id: NodeId) {
        self.failed.remove(&node_id);
        self.suspected.remove(&node_id);
    }

    /// Get statistics for monitoring
    pub fn stats(&self) -> FailureDetectorStats {
        FailureDetectorStats {
            total_nodes: self.heartbeat_history.len(),
            suspected_count: self.suspected.len(),
            failed_count: self.failed.len(),
            avg_phi: self.phi_values.values().sum::<f64>() / self.phi_values.len().max(1) as f64,
        }
    }
}

/// Statistics about failure detector state
#[derive(Debug, Clone)]
pub struct FailureDetectorStats {
    pub total_nodes: usize,
    pub suspected_count: usize,
    pub failed_count: usize,
    pub avg_phi: f64,
}

/// Approximate normal distribution CDF using error function
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Approximate error function using Abramowitz and Stegun formula
fn erf(x: f64) -> f64 {
    // Constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_phi_calculation() {
        let mut detector = PhiAccrualFailureDetector::new(100, 8.0, 12.0);

        // Send regular heartbeats
        for _ in 0..MIN_SAMPLES {
            detector.record_heartbeat(NodeId(0));
            thread::sleep(Duration::from_millis(100));
        }

        let phi = detector.phi(NodeId(0));
        assert!(phi.is_some());
        assert!(phi.unwrap() < 8.0); // Should not be suspected yet
    }

    #[test]
    fn test_failure_detection() {
        let mut detector = PhiAccrualFailureDetector::new(100, 5.0, 8.0);

        // Establish baseline
        for _ in 0..MIN_SAMPLES {
            detector.record_heartbeat(NodeId(0));
            thread::sleep(Duration::from_millis(100));
        }

        // Simulate failure (no heartbeats)
        thread::sleep(Duration::from_millis(1000));
        detector.update_phi(NodeId(0));

        let phi = detector.phi(NodeId(0)).unwrap();
        assert!(phi > 5.0); // Should be suspected
    }

    #[test]
    fn test_adaptive_interval() {
        let mut detector = PhiAccrualFailureDetector::new(100, 8.0, 12.0);

        // Establish baseline with regular intervals
        for _ in 0..20 {
            detector.record_heartbeat(NodeId(0));
            thread::sleep(Duration::from_millis(100));
        }

        let interval = detector.adaptive_interval(NodeId(0));
        assert!(interval.as_millis() >= 100);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(1.0) > 0.8);
        assert!(normal_cdf(-1.0) < 0.2);
    }
}
