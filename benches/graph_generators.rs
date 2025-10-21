//! Graph generators for benchmarking
//!
//! Provides various synthetic graph generation functions for testing
//! partitioning and algorithm performance.

use butterfly_core::{Graph, GraphBuilder, VertexId, EdgeId};
use rand::Rng;
use rand::seq::SliceRandom;

/// Generate a power-law graph using preferential attachment (Barabási-Albert model)
///
/// Each new vertex connects to m existing vertices with probability proportional
/// to their degree (preferential attachment).
pub fn generate_power_law_graph(n: usize) -> Graph {
    let m = 3; // Number of edges to attach from new vertex
    let mut builder = GraphBuilder::new("power_law");
    builder.undirected();

    // Start with a small complete graph
    let initial = m + 1;
    for i in 0..initial {
        builder.add_vertex(VertexId(i as u64));
    }

    for i in 0..initial {
        for j in (i + 1)..initial {
            builder.add_edge(VertexId(i as u64), VertexId(j as u64));
        }
    }

    // Add remaining vertices with preferential attachment
    let mut rng = rand::thread_rng();
    for v in initial..n {
        builder.add_vertex(VertexId(v as u64));

        // Collect vertices weighted by degree
        let graph = builder.build();
        let mut vertices_weighted: Vec<u64> = Vec::new();
        for (vid, vertex) in &graph.vertices {
            for _ in 0..vertex.degree().max(1) {
                vertices_weighted.push(vid.0);
            }
        }

        // Rebuild builder with the graph
        builder = GraphBuilder::new("power_law");
        builder.undirected();
        for (vid, vertex) in &graph.vertices {
            builder.add_vertex(*vid);
        }
        for edge in graph.edges.values() {
            builder.add_edge(edge.source, edge.target);
        }

        // Select m vertices to connect to
        vertices_weighted.shuffle(&mut rng);
        let mut connected = std::collections::HashSet::new();
        for &target in vertices_weighted.iter().take(vertices_weighted.len().min(100)) {
            if connected.len() >= m {
                break;
            }
            if target != v as u64 && connected.insert(target) {
                builder.add_edge(VertexId(v as u64), VertexId(target));
            }
        }
    }

    builder.build()
}

/// Generate a random Erdős-Rényi graph
///
/// Each edge exists with probability p.
pub fn generate_random_graph(n: usize, p: f64) -> Graph {
    let mut builder = GraphBuilder::new("random");
    builder.undirected();

    // Add all vertices
    for i in 0..n {
        builder.add_vertex(VertexId(i as u64));
    }

    // Add edges with probability p
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen::<f64>() < p {
                builder.add_edge(VertexId(i as u64), VertexId(j as u64));
            }
        }
    }

    builder.build()
}

/// Generate a 2D grid graph
///
/// Creates a grid with width x height vertices, where each vertex is connected
/// to its 4 neighbors (up, down, left, right).
pub fn generate_grid_graph(width: usize, height: usize) -> Graph {
    let mut builder = GraphBuilder::new("grid");
    builder.undirected();

    let n = width * height;

    // Add all vertices
    for i in 0..n {
        builder.add_vertex(VertexId(i as u64));
    }

    // Add edges
    for y in 0..height {
        for x in 0..width {
            let current = y * width + x;

            // Right neighbor
            if x + 1 < width {
                let neighbor = y * width + (x + 1);
                builder.add_edge(VertexId(current as u64), VertexId(neighbor as u64));
            }

            // Down neighbor
            if y + 1 < height {
                let neighbor = (y + 1) * width + x;
                builder.add_edge(VertexId(current as u64), VertexId(neighbor as u64));
            }
        }
    }

    builder.build()
}

/// Generate a cycle graph
///
/// Creates a simple cycle where vertices are connected in a ring.
pub fn generate_cycle_graph(n: usize) -> Graph {
    let mut builder = GraphBuilder::new("cycle");
    builder.undirected();

    // Add all vertices
    for i in 0..n {
        builder.add_vertex(VertexId(i as u64));
    }

    // Connect in a cycle
    for i in 0..n {
        let next = (i + 1) % n;
        builder.add_edge(VertexId(i as u64), VertexId(next as u64));
    }

    builder.build()
}

/// Generate a complete graph (clique)
///
/// Every vertex is connected to every other vertex.
pub fn generate_complete_graph(n: usize) -> Graph {
    let mut builder = GraphBuilder::new("complete");
    builder.undirected();

    // Add all vertices
    for i in 0..n {
        builder.add_vertex(VertexId(i as u64));
    }

    // Connect all pairs
    for i in 0..n {
        for j in (i + 1)..n {
            builder.add_edge(VertexId(i as u64), VertexId(j as u64));
        }
    }

    builder.build()
}

/// Generate a star graph
///
/// One central vertex connected to all other vertices.
pub fn generate_star_graph(n: usize) -> Graph {
    let mut builder = GraphBuilder::new("star");
    builder.undirected();

    // Add all vertices
    for i in 0..n {
        builder.add_vertex(VertexId(i as u64));
    }

    // Connect all vertices to vertex 0 (center)
    for i in 1..n {
        builder.add_edge(VertexId(0), VertexId(i as u64));
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_law_graph() {
        let graph = generate_power_law_graph(100);
        assert_eq!(graph.vertex_count(), 100);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_random_graph() {
        let graph = generate_random_graph(50, 0.1);
        assert_eq!(graph.vertex_count(), 50);
    }

    #[test]
    fn test_grid_graph() {
        let graph = generate_grid_graph(10, 10);
        assert_eq!(graph.vertex_count(), 100);
        // Grid has (width-1)*height + width*(height-1) edges
        assert_eq!(graph.edge_count(), 9 * 10 + 10 * 9);
    }

    #[test]
    fn test_cycle_graph() {
        let graph = generate_cycle_graph(100);
        assert_eq!(graph.vertex_count(), 100);
        assert_eq!(graph.edge_count(), 100);
    }

    #[test]
    fn test_complete_graph() {
        let graph = generate_complete_graph(10);
        assert_eq!(graph.vertex_count(), 10);
        // Complete graph has n*(n-1)/2 edges
        assert_eq!(graph.edge_count(), 45);
    }

    #[test]
    fn test_star_graph() {
        let graph = generate_star_graph(10);
        assert_eq!(graph.vertex_count(), 10);
        assert_eq!(graph.edge_count(), 9);
    }
}
