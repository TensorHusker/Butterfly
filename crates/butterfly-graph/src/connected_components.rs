//! Connected Components algorithm using BSP model
//!
//! Finds connected components in an undirected graph. Each vertex is assigned
//! a component ID (the minimum vertex ID in its component).

use butterfly_core::{
    Vertex, VertexCompute, Message, MessagePayload, ComputationResult,
    ComputeError, AggregatedResult, VertexData,
};
use crate::{AlgorithmConfig, GraphAlgorithm};

/// Connected Components computation
pub struct ConnectedComponents {
    config: AlgorithmConfig,
}

impl ConnectedComponents {
    pub fn new() -> Self {
        Self {
            config: AlgorithmConfig::default(),
        }
    }
}

impl Default for ConnectedComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexCompute for ConnectedComponents {
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError> {
        let current_component = vertex.data.get_u64("component")
            .unwrap_or(vertex.id.0);

        // Find minimum component ID from messages
        let min_component = messages.iter()
            .filter_map(|m| m.payload.as_u64())
            .min()
            .unwrap_or(current_component);

        let new_component = min_component.min(current_component);

        // Send component ID to neighbors if it changed
        let mut outgoing = Vec::new();
        if new_component < current_component {
            for &neighbor_id in &vertex.neighbors {
                outgoing.push(Message::new(
                    vertex.id,
                    neighbor_id,
                    MessagePayload::U64(new_component),
                ));
            }
        }

        let mut data = VertexData::new();
        data.set_u64("component", new_component);

        Ok(ComputationResult::new(data)
            .with_messages(outgoing)
            .active())
    }

    fn init(&mut self, vertex: &mut Vertex) -> Result<(), ComputeError> {
        // Initialize each vertex with its own ID as component
        vertex.data.set_u64("component", vertex.id.0);
        vertex.activate();
        Ok(())
    }

    fn should_terminate(&self, superstep: u32, active_vertices: usize) -> bool {
        // Connected components converges when no vertices are active
        active_vertices == 0 || superstep >= self.config.max_iterations
    }
}

impl GraphAlgorithm for ConnectedComponents {
    fn name(&self) -> &str {
        "Connected Components"
    }

    fn config(&self) -> &AlgorithmConfig {
        &self.config
    }
}
