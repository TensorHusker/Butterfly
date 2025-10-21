//! Core graph data structures for distributed graph processing
//!
//! This module provides the fundamental graph types used throughout Butterfly's
//! distributed graph processing system. Graphs can represent both general
//! computation graphs (for algorithms like PageRank) and transformer neural networks.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Unique identifier for a vertex in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexId(pub u64);

/// Unique identifier for an edge in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

/// Unique identifier for a partition/subgraph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionId(pub usize);

/// Generic vertex data that can hold any serializable type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexData {
    /// Key-value storage for vertex attributes
    attributes: HashMap<String, VertexValue>,
}

/// Supported vertex value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VertexValue {
    F64(f64),
    I64(i64),
    U64(u64),
    String(String),
    Bool(bool),
    Bytes(Vec<u8>),
}

impl VertexData {
    pub fn new() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }

    pub fn with_f64(key: impl Into<String>, value: f64) -> Self {
        let mut data = Self::new();
        data.set_f64(key, value);
        data
    }

    pub fn set_f64(&mut self, key: impl Into<String>, value: f64) {
        self.attributes.insert(key.into(), VertexValue::F64(value));
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.attributes.get(key)? {
            VertexValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn set_u64(&mut self, key: impl Into<String>, value: u64) {
        self.attributes.insert(key.into(), VertexValue::U64(value));
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.attributes.get(key)? {
            VertexValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.attributes.insert(key.into(), VertexValue::Bool(value));
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.attributes.get(key)? {
            VertexValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl Default for VertexData {
    fn default() -> Self {
        Self::new()
    }
}

/// A vertex in the graph with associated data and neighbors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub id: VertexId,
    pub data: VertexData,
    /// Outgoing edges to neighbor vertices
    pub neighbors: Vec<VertexId>,
    /// Active state for BSP computation
    pub active: bool,
}

impl Vertex {
    pub fn new(id: VertexId) -> Self {
        Self {
            id,
            data: VertexData::new(),
            neighbors: Vec::new(),
            active: true,
        }
    }

    pub fn with_data(id: VertexId, data: VertexData) -> Self {
        Self {
            id,
            data,
            neighbors: Vec::new(),
            active: true,
        }
    }

    pub fn add_neighbor(&mut self, neighbor: VertexId) {
        if !self.neighbors.contains(&neighbor) {
            self.neighbors.push(neighbor);
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn activate(&mut self) {
        self.active = true;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn degree(&self) -> usize {
        self.neighbors.len()
    }
}

/// Edge data for weighted graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub weight: f64,
    /// Optional edge attributes
    pub attributes: HashMap<String, VertexValue>,
}

impl EdgeData {
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            attributes: HashMap::new(),
        }
    }

    pub fn unweighted() -> Self {
        Self::new(1.0)
    }
}

impl Default for EdgeData {
    fn default() -> Self {
        Self::unweighted()
    }
}

/// A directed edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: VertexId,
    pub target: VertexId,
    pub data: EdgeData,
}

impl Edge {
    pub fn new(id: EdgeId, source: VertexId, target: VertexId) -> Self {
        Self {
            id,
            source,
            target,
            data: EdgeData::unweighted(),
        }
    }

    pub fn with_weight(id: EdgeId, source: VertexId, target: VertexId, weight: f64) -> Self {
        Self {
            id,
            source,
            target,
            data: EdgeData::new(weight),
        }
    }
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub name: String,
    pub directed: bool,
    pub weighted: bool,
    /// Custom metadata attributes
    pub attributes: HashMap<String, String>,
}

impl GraphMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            directed: true,
            weighted: false,
            attributes: HashMap::new(),
        }
    }

    pub fn directed(mut self) -> Self {
        self.directed = true;
        self
    }

    pub fn undirected(mut self) -> Self {
        self.directed = false;
        self
    }

    pub fn weighted(mut self) -> Self {
        self.weighted = true;
        self
    }
}

/// A complete graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub vertices: HashMap<VertexId, Vertex>,
    pub edges: HashMap<EdgeId, Edge>,
    pub metadata: GraphMetadata,
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            vertices: HashMap::new(),
            edges: HashMap::new(),
            metadata: GraphMetadata::new(name),
        }
    }

    pub fn add_vertex(&mut self, vertex: Vertex) {
        self.vertices.insert(vertex.id, vertex);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        // Add edge to graph
        self.edges.insert(edge.id, edge.clone());

        // Update source vertex's neighbor list
        if let Some(vertex) = self.vertices.get_mut(&edge.source) {
            vertex.add_neighbor(edge.target);
        }

        // For undirected graphs, add reverse edge
        if !self.metadata.directed {
            if let Some(vertex) = self.vertices.get_mut(&edge.target) {
                vertex.add_neighbor(edge.source);
            }
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get_vertex(&self, id: VertexId) -> Option<&Vertex> {
        self.vertices.get(&id)
    }

    pub fn get_vertex_mut(&mut self, id: VertexId) -> Option<&mut Vertex> {
        self.vertices.get_mut(&id)
    }

    pub fn get_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(&id)
    }
}

/// A subgraph representing a partition of the main graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    /// The underlying graph structure
    pub graph: Graph,
    /// Partition identifier
    pub partition_id: PartitionId,
    /// Vertices that have edges to other partitions (boundary vertices)
    pub boundary_vertices: HashSet<VertexId>,
    /// Cached vertices from other partitions (ghost vertices)
    /// These are read-only copies used for message passing
    pub ghost_vertices: HashMap<VertexId, Vertex>,
}

impl Subgraph {
    pub fn new(partition_id: PartitionId, name: impl Into<String>) -> Self {
        Self {
            graph: Graph::new(name),
            partition_id,
            boundary_vertices: HashSet::new(),
            ghost_vertices: HashMap::new(),
        }
    }

    pub fn is_boundary_vertex(&self, vertex_id: VertexId) -> bool {
        self.boundary_vertices.contains(&vertex_id)
    }

    pub fn add_ghost_vertex(&mut self, vertex: Vertex) {
        self.ghost_vertices.insert(vertex.id, vertex);
    }

    /// Update an existing ghost vertex with new state
    /// This should be called during BSP superstep synchronization
    pub fn update_ghost_vertex(&mut self, vertex: Vertex) {
        if let Some(existing) = self.ghost_vertices.get_mut(&vertex.id) {
            *existing = vertex;
        } else {
            self.ghost_vertices.insert(vertex.id, vertex);
        }
    }

    /// Batch update ghost vertices for efficiency
    pub fn sync_ghost_vertices(&mut self, updates: Vec<Vertex>) {
        for vertex in updates {
            self.update_ghost_vertex(vertex);
        }
    }

    /// Remove ghost vertices that are no longer needed
    pub fn prune_unused_ghosts(&mut self) {
        let boundary_neighbors: std::collections::HashSet<VertexId> = self
            .boundary_vertices
            .iter()
            .filter_map(|vid| self.graph.vertices.get(vid))
            .flat_map(|v| v.neighbors.iter().copied())
            .collect();

        self.ghost_vertices.retain(|vid, _| boundary_neighbors.contains(vid));
    }

    pub fn get_local_vertex(&self, vertex_id: VertexId) -> Option<&Vertex> {
        self.graph.get_vertex(vertex_id)
    }

    pub fn get_local_vertex_mut(&mut self, vertex_id: VertexId) -> Option<&mut Vertex> {
        self.graph.get_vertex_mut(vertex_id)
    }

    pub fn get_ghost_vertex(&self, vertex_id: VertexId) -> Option<&Vertex> {
        self.ghost_vertices.get(&vertex_id)
    }

    /// Get vertex from either local or ghost vertices
    pub fn get_vertex(&self, vertex_id: VertexId) -> Option<&Vertex> {
        self.graph.get_vertex(vertex_id)
            .or_else(|| self.ghost_vertices.get(&vertex_id))
    }

    pub fn local_vertex_count(&self) -> usize {
        self.graph.vertex_count()
    }

    pub fn ghost_vertex_count(&self) -> usize {
        self.ghost_vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Calculate the edge cut (edges crossing partition boundaries)
    pub fn edge_cut(&self) -> usize {
        let mut cut = 0;
        for edge in self.graph.edges.values() {
            // Edge crosses boundary if target is not in local graph
            if !self.graph.vertices.contains_key(&edge.target) {
                cut += 1;
            }
        }
        cut
    }
}

/// Builder for constructing graphs
pub struct GraphBuilder {
    graph: Graph,
    next_edge_id: u64,
}

impl GraphBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: Graph::new(name),
            next_edge_id: 0,
        }
    }

    pub fn directed(&mut self) -> &mut Self {
        self.graph.metadata.directed = true;
        self
    }

    pub fn undirected(&mut self) -> &mut Self {
        self.graph.metadata.directed = false;
        self
    }

    pub fn weighted(&mut self) -> &mut Self {
        self.graph.metadata.weighted = true;
        self
    }

    pub fn add_vertex(&mut self, id: VertexId) -> &mut Self {
        self.graph.add_vertex(Vertex::new(id));
        self
    }

    pub fn add_vertex_with_data(&mut self, id: VertexId, data: VertexData) -> &mut Self {
        self.graph.add_vertex(Vertex::with_data(id, data));
        self
    }

    pub fn add_edge(&mut self, source: VertexId, target: VertexId) -> &mut Self {
        // Auto-add vertices if they don't exist
        if !self.graph.vertices.contains_key(&source) {
            self.graph.add_vertex(Vertex::new(source));
        }
        if !self.graph.vertices.contains_key(&target) {
            self.graph.add_vertex(Vertex::new(target));
        }

        let edge_id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.graph.add_edge(Edge::new(edge_id, source, target));
        self
    }

    pub fn add_weighted_edge(&mut self, source: VertexId, target: VertexId, weight: f64) -> &mut Self {
        // Auto-add vertices if they don't exist
        if !self.graph.vertices.contains_key(&source) {
            self.graph.add_vertex(Vertex::new(source));
        }
        if !self.graph.vertices.contains_key(&target) {
            self.graph.add_vertex(Vertex::new(target));
        }

        let edge_id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.graph.add_edge(Edge::with_weight(edge_id, source, target, weight));
        self
    }

    pub fn build(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation() {
        let vertex = Vertex::new(VertexId(1));
        assert_eq!(vertex.id, VertexId(1));
        assert!(vertex.is_active());
        assert_eq!(vertex.degree(), 0);
    }

    #[test]
    fn test_vertex_data() {
        let mut data = VertexData::new();
        data.set_f64("pagerank", 0.25);
        assert_eq!(data.get_f64("pagerank"), Some(0.25));
    }

    #[test]
    fn test_graph_builder_directed() {
        let mut builder = GraphBuilder::new("test");
        builder.directed()
            .add_vertex(VertexId(1))
            .add_vertex(VertexId(2))
            .add_vertex(VertexId(3))
            .add_edge(VertexId(1), VertexId(2))
            .add_edge(VertexId(2), VertexId(3));
        let graph = builder.build();

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.metadata.directed);
    }

    #[test]
    fn test_graph_builder_undirected() {
        let mut builder = GraphBuilder::new("test");
        builder.undirected()
            .add_vertex(VertexId(1))
            .add_vertex(VertexId(2))
            .add_edge(VertexId(1), VertexId(2));

        let graph = builder.build();

        // Undirected: both vertices should have each other as neighbors
        let v1 = graph.get_vertex(VertexId(1)).unwrap();
        let v2 = graph.get_vertex(VertexId(2)).unwrap();

        assert!(v1.neighbors.contains(&VertexId(2)));
        assert!(v2.neighbors.contains(&VertexId(1)));
    }

    #[test]
    fn test_subgraph() {
        let mut subgraph = Subgraph::new(PartitionId(0), "partition-0");

        let mut vertex = Vertex::new(VertexId(1));
        vertex.data.set_f64("value", 1.0);
        subgraph.graph.add_vertex(vertex);

        assert_eq!(subgraph.local_vertex_count(), 1);
        assert_eq!(subgraph.ghost_vertex_count(), 0);

        // Add ghost vertex
        let ghost = Vertex::new(VertexId(99));
        subgraph.add_ghost_vertex(ghost);
        assert_eq!(subgraph.ghost_vertex_count(), 1);
    }

    #[test]
    fn test_edge_cut() {
        let mut subgraph = Subgraph::new(PartitionId(0), "partition-0");

        // Add local vertices
        subgraph.graph.add_vertex(Vertex::new(VertexId(1)));
        subgraph.graph.add_vertex(Vertex::new(VertexId(2)));

        // Add local edge (no cut)
        subgraph.graph.add_edge(Edge::new(EdgeId(0), VertexId(1), VertexId(2)));

        // Add edge to vertex in different partition (cut edge)
        subgraph.graph.add_edge(Edge::new(EdgeId(1), VertexId(2), VertexId(99)));

        // Mark boundary vertex
        subgraph.boundary_vertices.insert(VertexId(2));

        assert_eq!(subgraph.edge_cut(), 1);
    }
}
