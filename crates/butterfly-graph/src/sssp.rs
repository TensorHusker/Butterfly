//! Single-Source Shortest Paths (SSSP) algorithm using BSP model
//!
//! Computes shortest paths from a source vertex to all other vertices
//! in a weighted graph.

use butterfly_core::{
    Vertex, VertexCompute, Message, MessagePayload, ComputationResult,
    ComputeError, AggregatedResult, VertexData, VertexId,
};
use crate::{AlgorithmConfig, GraphAlgorithm};

/// Shortest Paths computation
pub struct ShortestPaths {
    source: VertexId,
    config: AlgorithmConfig,
}

impl ShortestPaths {
    pub fn new(source: VertexId) -> Self {
        Self {
            source,
            config: AlgorithmConfig::default(),
        }
    }
}

impl VertexCompute for ShortestPaths {
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError> {
        let current_dist = vertex.data.get_f64("distance")
            .unwrap_or(f64::INFINITY);

        // Initialize source vertex
        let new_dist = if vertex.id == self.source && superstep == 0 {
            0.0
        } else {
            // Find minimum distance from messages
            messages.iter()
                .filter_map(|m| m.payload.as_f64())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(current_dist)
        };

        let mut outgoing = Vec::new();
        if new_dist < current_dist {
            // Propagate new distance to neighbors
            for &neighbor_id in &vertex.neighbors {
                // TODO: Get edge weight from graph
                let edge_weight = 1.0;
                outgoing.push(Message::new(
                    vertex.id,
                    neighbor_id,
                    MessagePayload::F64(new_dist + edge_weight),
                ));
            }
        }

        let mut data = VertexData::new();
        data.set_f64("distance", new_dist);

        Ok(ComputationResult::new(data)
            .with_messages(outgoing)
            .active())
    }

    fn init(&mut self, vertex: &mut Vertex) -> Result<(), ComputeError> {
        if vertex.id == self.source {
            vertex.data.set_f64("distance", 0.0);
        } else {
            vertex.data.set_f64("distance", f64::INFINITY);
        }
        vertex.activate();
        Ok(())
    }
}

impl GraphAlgorithm for ShortestPaths {
    fn name(&self) -> &str {
        "Single-Source Shortest Paths"
    }

    fn config(&self) -> &AlgorithmConfig {
        &self.config
    }
}
