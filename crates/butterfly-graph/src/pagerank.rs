//! PageRank algorithm implementation using BSP model
//!
//! PageRank computes the importance of vertices in a graph based on the link structure.
//! Each vertex distributes its rank to its neighbors, and vertices update their rank
//! based on incoming contributions.
//!
//! # Algorithm
//! 1. Initialize all vertices with rank 1/N (where N is the number of vertices)
//! 2. In each superstep:
//!    - Each vertex sends rank/degree to all neighbors
//!    - Each vertex updates rank = (1-d) + d * sum(incoming ranks)
//!    - Vertices become inactive if rank change < threshold
//! 3. Terminate when all vertices are inactive or max iterations reached

use butterfly_core::{
    Vertex, VertexCompute, Message, MessagePayload, ComputationResult,
    ComputeError, AggregatedResult, VertexData,
};
use crate::{AlgorithmConfig, GraphAlgorithm};
use serde::{Deserialize, Serialize};

/// PageRank algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankConfig {
    /// Damping factor (typically 0.85)
    pub damping_factor: f64,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Maximum iterations
    pub max_iterations: u32,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            convergence_threshold: 0.001,
            max_iterations: 100,
        }
    }
}

/// PageRank computation
pub struct PageRank {
    config: PageRankConfig,
}

impl PageRank {
    pub fn new(config: PageRankConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(PageRankConfig::default())
    }

    pub fn with_damping(damping: f64) -> Self {
        Self::new(PageRankConfig {
            damping_factor: damping,
            ..Default::default()
        })
    }
}

impl Default for PageRank {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl VertexCompute for PageRank {
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError> {
        // Get current PageRank (default to 1.0 if not set)
        let current_rank = vertex.data.get_f64("rank").unwrap_or(1.0);

        // Initialize on first superstep
        if superstep == 0 {
            let num_neighbors = vertex.degree() as f64;

            if num_neighbors == 0.0 {
                // Dead-end vertex
                let mut data = VertexData::new();
                data.set_f64("rank", current_rank);
                return Ok(ComputationResult::new(data).inactive());
            }

            // Send initial rank contribution to neighbors
            let contribution = current_rank / num_neighbors;
            let mut outgoing = Vec::new();

            for &neighbor_id in &vertex.neighbors {
                outgoing.push(Message::new(
                    vertex.id,
                    neighbor_id,
                    MessagePayload::F64(contribution),
                ));
            }

            let mut data = VertexData::new();
            data.set_f64("rank", current_rank);

            return Ok(ComputationResult::new(data)
                .with_messages(outgoing)
                .active());
        }

        // Sum incoming rank contributions
        let rank_sum: f64 = messages
            .iter()
            .filter_map(|m| m.payload.as_f64())
            .sum();

        // Calculate new PageRank: (1-d) + d * sum(contributions)
        let new_rank = (1.0 - self.config.damping_factor)
            + self.config.damping_factor * rank_sum;

        // Check for convergence using both absolute and relative error
        let delta = (new_rank - current_rank).abs();
        let relative_error = if current_rank.abs() > 1e-10 {
            delta / current_rank.abs()
        } else {
            delta
        };
        let converged = delta < self.config.convergence_threshold ||
                       relative_error < self.config.convergence_threshold;

        // Distribute new rank to neighbors
        let num_neighbors = vertex.degree() as f64;
        let mut outgoing = Vec::new();

        if num_neighbors > 0.0 && !converged {
            let contribution = new_rank / num_neighbors;

            for &neighbor_id in &vertex.neighbors {
                outgoing.push(Message::new(
                    vertex.id,
                    neighbor_id,
                    MessagePayload::F64(contribution),
                ));
            }
        }

        // Update vertex data
        let mut data = VertexData::new();
        data.set_f64("rank", new_rank);
        data.set_f64("delta", delta);

        // Return active/inactive based on convergence
        let result = ComputationResult::new(data)
            .with_messages(outgoing);

        if converged {
            Ok(result.inactive())
        } else {
            Ok(result.active())
        }
    }

    fn init(&mut self, vertex: &mut Vertex) -> Result<(), ComputeError> {
        // Initialize all vertices with rank 1.0
        vertex.data.set_f64("rank", 1.0);
        vertex.data.set_f64("delta", f64::INFINITY);
        vertex.activate();
        Ok(())
    }

    fn aggregate(&self, results: &[ComputationResult]) -> AggregatedResult {
        let active_count = results.iter().filter(|r| r.is_active).count();
        let messages_sent = results.iter()
            .map(|r| r.outgoing_messages.len())
            .sum();

        // Calculate average delta for convergence monitoring
        let total_delta: f64 = results.iter()
            .filter_map(|r| r.new_value.get_f64("delta"))
            .sum();
        let avg_delta = if !results.is_empty() {
            total_delta / results.len() as f64
        } else {
            0.0
        };

        let mut agg = AggregatedResult::new()
            .with_active_count(active_count)
            .with_messages_sent(messages_sent);

        agg.add_aggregation("avg_delta", avg_delta);
        agg
    }

    fn should_terminate(&self, superstep: u32, active_vertices: usize) -> bool {
        active_vertices == 0 || superstep >= self.config.max_iterations
    }
}

impl GraphAlgorithm for PageRank {
    fn name(&self) -> &str {
        "PageRank"
    }

    fn config(&self) -> &AlgorithmConfig {
        // Convert PageRankConfig to AlgorithmConfig
        static CONFIG: AlgorithmConfig = AlgorithmConfig {
            max_iterations: 100,
            convergence_threshold: 0.001,
        };
        &CONFIG
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::{Vertex, VertexId, VertexData};

    #[test]
    fn test_pagerank_initialization() {
        let mut pagerank = PageRank::with_defaults();
        let mut vertex = Vertex::new(VertexId(1));
        vertex.add_neighbor(VertexId(2));

        pagerank.init(&mut vertex).unwrap();
        assert_eq!(vertex.data.get_f64("rank"), Some(1.0));
        assert!(vertex.is_active());
    }

    #[test]
    fn test_pagerank_first_superstep() {
        let pagerank = PageRank::with_defaults();
        let mut vertex = Vertex::new(VertexId(1));
        vertex.add_neighbor(VertexId(2));
        vertex.add_neighbor(VertexId(3));
        vertex.data.set_f64("rank", 1.0);

        let result = pagerank.compute(&vertex, &[], 0).unwrap();

        // Should send to 2 neighbors
        assert_eq!(result.outgoing_messages.len(), 2);

        // Each neighbor should get rank/degree = 1.0/2.0 = 0.5
        for msg in &result.outgoing_messages {
            assert_eq!(msg.payload.as_f64(), Some(0.5));
        }

        assert!(result.is_active);
    }

    #[test]
    fn test_pagerank_convergence() {
        let pagerank = PageRank::with_damping(0.85);
        let mut vertex = Vertex::new(VertexId(1));
        vertex.add_neighbor(VertexId(2));
        vertex.data.set_f64("rank", 0.5);

        // Simulate receiving same rank contribution (should converge)
        let messages = vec![
            Message::new(VertexId(2), VertexId(1), MessagePayload::F64(0.425)),
        ];

        let result = pagerank.compute(&vertex, &messages, 1).unwrap();

        // New rank should be (1-0.85) + 0.85 * 0.425 = 0.15 + 0.36125 = 0.51125
        let new_rank = result.new_value.get_f64("rank").unwrap();
        assert!((new_rank - 0.51125).abs() < 0.0001);
    }

    #[test]
    fn test_pagerank_dead_end() {
        let pagerank = PageRank::with_defaults();
        let mut vertex = Vertex::new(VertexId(1)); // No neighbors
        vertex.data.set_f64("rank", 1.0);

        let result = pagerank.compute(&vertex, &[], 0).unwrap();

        // Dead-end vertex should become inactive
        assert!(!result.is_active);
        assert_eq!(result.outgoing_messages.len(), 0);
    }
}
