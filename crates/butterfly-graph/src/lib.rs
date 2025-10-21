//! # Butterfly Graph Algorithms
//!
//! Reference implementations of distributed graph algorithms using the BSP model.
//! These algorithms demonstrate vertex-centric computation and serve as examples
//! for building custom graph processing applications.

pub mod pagerank;
pub mod connected_components;
pub mod sssp;

pub use pagerank::PageRank;
pub use connected_components::ConnectedComponents;
pub use sssp::ShortestPaths;

use butterfly_core::{VertexCompute, ComputeError};

/// Common configuration for graph algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Maximum number of supersteps before termination
    pub max_iterations: u32,
    /// Convergence threshold for iterative algorithms
    pub convergence_threshold: f64,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.001,
        }
    }
}

/// Trait for graph algorithms that can be run distributedly
pub trait GraphAlgorithm: VertexCompute {
    /// Get the name of the algorithm
    fn name(&self) -> &str;

    /// Get the configuration
    fn config(&self) -> &AlgorithmConfig;
}
