//! Graph I/O operations for loading and exporting graph data
//!
//! This module provides utilities for reading graphs from various file formats
//! and exporting computation results for analysis and visualization.

use crate::graph::{Edge, EdgeData, EdgeId, Graph, GraphBuilder, Vertex, VertexId, VertexData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during graph I/O operations
#[derive(Debug, Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Vertex not found: {0:?}")]
    VertexNotFound(VertexId),

    #[error("Attribute not found: {0}")]
    AttributeNotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Comma-separated values format
    Csv,
    /// JSON format with full graph structure
    Json,
}

/// Graph loader for importing graphs from various file formats
pub struct GraphLoader;

impl GraphLoader {
    /// Load a graph from an edge list file
    ///
    /// Edge list format:
    /// ```text
    /// source target [weight]
    /// 1 2
    /// 2 3 0.5
    /// ```
    ///
    /// Lines starting with '#' are treated as comments.
    /// Vertices are auto-created from edge definitions.
    pub fn from_edge_list(path: &Path, directed: bool) -> Result<Graph, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("graph");

        let mut builder = GraphBuilder::new(name);
        if directed {
            builder.directed();
        } else {
            builder.undirected();
        }

        let mut weighted = false;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < 2 {
                return Err(IoError::ParseError {
                    line: line_num + 1,
                    message: format!("Expected at least 2 fields, got {}", parts.len()),
                });
            }

            let source = parts[0].parse::<u64>().map_err(|_| IoError::ParseError {
                line: line_num + 1,
                message: format!("Invalid source vertex ID: {}", parts[0]),
            })?;

            let target = parts[1].parse::<u64>().map_err(|_| IoError::ParseError {
                line: line_num + 1,
                message: format!("Invalid target vertex ID: {}", parts[1]),
            })?;

            if parts.len() >= 3 {
                // Weighted edge
                let weight = parts[2].parse::<f64>().map_err(|_| IoError::ParseError {
                    line: line_num + 1,
                    message: format!("Invalid weight: {}", parts[2]),
                })?;
                builder.add_weighted_edge(VertexId(source), VertexId(target), weight);
                weighted = true;
            } else {
                // Unweighted edge
                builder.add_edge(VertexId(source), VertexId(target));
            }
        }

        if weighted {
            builder.weighted();
        }

        Ok(builder.build())
    }

    /// Load a graph from an adjacency list file
    ///
    /// Adjacency list format:
    /// ```text
    /// vertex: neighbor1 neighbor2 neighbor3
    /// 1: 2 3 4
    /// 2: 3
    /// 3: 1 4
    /// ```
    ///
    /// Lines starting with '#' are treated as comments.
    /// Creates a directed graph by default.
    pub fn from_adjacency_list(path: &Path) -> Result<Graph, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("graph");

        let mut builder = GraphBuilder::new(name);
        builder.directed();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(':').collect();

            if parts.len() != 2 {
                return Err(IoError::ParseError {
                    line: line_num + 1,
                    message: "Expected format 'vertex: neighbor1 neighbor2 ...'".to_string(),
                });
            }

            let vertex_id = parts[0].trim().parse::<u64>().map_err(|_| IoError::ParseError {
                line: line_num + 1,
                message: format!("Invalid vertex ID: {}", parts[0]),
            })?;

            // Add the vertex
            builder.add_vertex(VertexId(vertex_id));

            // Parse neighbors
            let neighbors_str = parts[1].trim();
            if !neighbors_str.is_empty() {
                for neighbor_str in neighbors_str.split_whitespace() {
                    let neighbor_id =
                        neighbor_str.parse::<u64>().map_err(|_| IoError::ParseError {
                            line: line_num + 1,
                            message: format!("Invalid neighbor ID: {}", neighbor_str),
                        })?;

                    builder.add_edge(VertexId(vertex_id), VertexId(neighbor_id));
                }
            }
        }

        Ok(builder.build())
    }

    /// Auto-detect format and load graph
    ///
    /// Detection rules:
    /// - Files ending in .adjlist or .adj -> adjacency list
    /// - Files ending in .edgelist or .edges -> edge list
    /// - Otherwise, try to detect from content (first non-comment line)
    pub fn from_file(path: &Path, directed: bool) -> Result<Graph, IoError> {
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        match extension {
            "adj" | "adjlist" => Self::from_adjacency_list(path),
            "edges" | "edgelist" => Self::from_edge_list(path, directed),
            _ => {
                // Try to auto-detect from content
                let file = File::open(path)?;
                let reader = BufReader::new(file);

                // Read first non-comment line
                for line in reader.lines() {
                    let line = line?;
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        // Check if it contains a colon (adjacency list format)
                        if trimmed.contains(':') {
                            return Self::from_adjacency_list(path);
                        } else {
                            return Self::from_edge_list(path, directed);
                        }
                    }
                }

                Err(IoError::InvalidFormat(
                    "Could not detect format from file content".to_string(),
                ))
            }
        }
    }
}

/// Result exporter for saving computation results
pub struct ResultExporter;

impl ResultExporter {
    /// Export vertex data to a file
    ///
    /// For CSV format, creates a file with headers: vertex_id,value
    /// For JSON format, exports complete graph structure with computed values
    pub fn export_vertex_data(
        graph: &Graph,
        attribute: &str,
        path: &Path,
        format: ExportFormat,
    ) -> Result<(), IoError> {
        match format {
            ExportFormat::Csv => Self::export_csv(graph, attribute, path),
            ExportFormat::Json => Self::export_json(graph, attribute, path),
        }
    }

    /// Export to CSV format
    fn export_csv(graph: &Graph, attribute: &str, path: &Path) -> Result<(), IoError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(writer, "vertex_id,{}", attribute)?;

        // Collect and sort vertices by ID for deterministic output
        let mut vertices: Vec<_> = graph.vertices.values().collect();
        vertices.sort_by_key(|v| v.id.0);

        // Write data rows
        for vertex in vertices {
            let value = vertex
                .data
                .get_f64(attribute)
                .ok_or_else(|| IoError::AttributeNotFound(attribute.to_string()))?;

            writeln!(writer, "{},{}", vertex.id.0, value)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Export to JSON format
    fn export_json(graph: &Graph, attribute: &str, path: &Path) -> Result<(), IoError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        // Create export structure
        let export_data = JsonGraphExport {
            metadata: JsonMetadata {
                name: graph.metadata.name.clone(),
                directed: graph.metadata.directed,
                weighted: graph.metadata.weighted,
                vertex_count: graph.vertex_count(),
                edge_count: graph.edge_count(),
                attribute: attribute.to_string(),
            },
            vertices: graph
                .vertices
                .values()
                .map(|v| {
                    Ok(JsonVertex {
                        id: v.id.0,
                        value: v
                            .data
                            .get_f64(attribute)
                            .ok_or_else(|| IoError::AttributeNotFound(attribute.to_string()))?,
                        neighbors: v.neighbors.iter().map(|n| n.0).collect(),
                    })
                })
                .collect::<Result<Vec<_>, IoError>>()?,
        };

        serde_json::to_writer_pretty(writer, &export_data)?;
        Ok(())
    }

    /// Export multiple attributes to JSON
    pub fn export_multi_attribute(
        graph: &Graph,
        attributes: &[&str],
        path: &Path,
    ) -> Result<(), IoError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        let vertices: Vec<_> = graph
            .vertices
            .values()
            .map(|v| {
                let mut vertex_attrs = HashMap::new();
                vertex_attrs.insert("id".to_string(), serde_json::json!(v.id.0));

                for attr in attributes {
                    if let Some(value) = v.data.get_f64(attr) {
                        vertex_attrs.insert((*attr).to_string(), serde_json::json!(value));
                    }
                }

                vertex_attrs.insert(
                    "neighbors".to_string(),
                    serde_json::json!(v.neighbors.iter().map(|n| n.0).collect::<Vec<_>>()),
                );

                vertex_attrs
            })
            .collect();

        let export_data = serde_json::json!({
            "graph": {
                "name": graph.metadata.name,
                "directed": graph.metadata.directed,
                "weighted": graph.metadata.weighted,
                "vertex_count": graph.vertex_count(),
                "edge_count": graph.edge_count(),
            },
            "attributes": attributes,
            "vertices": vertices,
        });

        serde_json::to_writer_pretty(writer, &export_data)?;
        Ok(())
    }
}

// JSON export structures
#[derive(Debug, Serialize, Deserialize)]
struct JsonGraphExport {
    metadata: JsonMetadata,
    vertices: Vec<JsonVertex>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonMetadata {
    name: String,
    directed: bool,
    weighted: bool,
    vertex_count: usize,
    edge_count: usize,
    attribute: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonVertex {
    id: u64,
    value: f64,
    neighbors: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_edge_list_unweighted() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Test graph").unwrap();
        writeln!(temp_file, "1 2").unwrap();
        writeln!(temp_file, "2 3").unwrap();
        writeln!(temp_file, "3 1").unwrap();
        temp_file.flush().unwrap();

        let graph = GraphLoader::from_edge_list(temp_file.path(), true).unwrap();

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.metadata.directed);
        assert!(!graph.metadata.weighted);
    }

    #[test]
    fn test_load_edge_list_weighted() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1 2 0.5").unwrap();
        writeln!(temp_file, "2 3 1.0").unwrap();
        writeln!(temp_file, "3 1 0.25").unwrap();
        temp_file.flush().unwrap();

        let graph = GraphLoader::from_edge_list(temp_file.path(), false).unwrap();

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert!(!graph.metadata.directed);
        assert!(graph.metadata.weighted);

        // Check edge weights
        let edge = graph.edges.values().next().unwrap();
        assert!(edge.data.weight > 0.0);
    }

    #[test]
    fn test_load_adjacency_list() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Adjacency list").unwrap();
        writeln!(temp_file, "1: 2 3").unwrap();
        writeln!(temp_file, "2: 3").unwrap();
        writeln!(temp_file, "3: ").unwrap();
        temp_file.flush().unwrap();

        let graph = GraphLoader::from_adjacency_list(temp_file.path()).unwrap();

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        let v1 = graph.get_vertex(VertexId(1)).unwrap();
        assert_eq!(v1.neighbors.len(), 2);
        assert!(v1.neighbors.contains(&VertexId(2)));
        assert!(v1.neighbors.contains(&VertexId(3)));
    }

    #[test]
    fn test_export_csv() {
        let mut builder = GraphBuilder::new("test");
        builder.directed();

        for i in 1..=3 {
            let mut data = VertexData::new();
            data.set_f64("rank", i as f64 * 0.1);
            builder.add_vertex_with_data(VertexId(i), data);
        }

        let graph = builder.build();

        let temp_file = NamedTempFile::new().unwrap();
        ResultExporter::export_vertex_data(&graph, "rank", temp_file.path(), ExportFormat::Csv)
            .unwrap();

        // Read back and verify
        let contents = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(contents.contains("vertex_id,rank"));
        assert!(contents.contains("1,0.1"));
        assert!(contents.contains("2,0.2"));
        assert!(contents.contains("3,0.3"));
    }

    #[test]
    fn test_export_json() {
        let mut builder = GraphBuilder::new("test");
        builder.directed().add_edge(VertexId(1), VertexId(2));

        // Add data to vertices
        let mut graph = builder.build();
        graph.get_vertex_mut(VertexId(1)).unwrap().data.set_f64("value", 1.5);
        graph.get_vertex_mut(VertexId(2)).unwrap().data.set_f64("value", 2.5);

        let temp_file = NamedTempFile::new().unwrap();
        ResultExporter::export_vertex_data(&graph, "value", temp_file.path(), ExportFormat::Json)
            .unwrap();

        // Read back and verify
        let contents = std::fs::read_to_string(temp_file.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

        assert_eq!(parsed["metadata"]["name"], "test");
        assert_eq!(parsed["metadata"]["vertex_count"], 2);
        assert_eq!(parsed["vertices"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_roundtrip_edge_list() {
        // Create a graph
        let mut builder = GraphBuilder::new("roundtrip");
        builder
            .directed()
            .add_weighted_edge(VertexId(1), VertexId(2), 0.5)
            .add_weighted_edge(VertexId(2), VertexId(3), 1.0);

        let original = builder.build();

        // Save to edge list
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = BufWriter::new(File::create(temp_file.path()).unwrap());
        for edge in original.edges.values() {
            writeln!(
                writer,
                "{} {} {}",
                edge.source.0, edge.target.0, edge.data.weight
            )
            .unwrap();
        }
        writer.flush().unwrap();

        // Load back
        let loaded = GraphLoader::from_edge_list(temp_file.path(), true).unwrap();

        assert_eq!(loaded.vertex_count(), original.vertex_count());
        assert_eq!(loaded.edge_count(), original.edge_count());
    }

    #[test]
    fn test_parse_error_handling() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "invalid line").unwrap();
        temp_file.flush().unwrap();

        let result = GraphLoader::from_edge_list(temp_file.path(), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_attribute_error() {
        let mut builder = GraphBuilder::new("test");
        builder.add_vertex(VertexId(1));
        let graph = builder.build();

        let temp_file = NamedTempFile::new().unwrap();
        let result =
            ResultExporter::export_vertex_data(&graph, "missing", temp_file.path(), ExportFormat::Csv);

        assert!(matches!(result, Err(IoError::AttributeNotFound(_))));
    }

    #[test]
    fn test_export_multi_attribute() {
        let mut builder = GraphBuilder::new("multi");
        builder.directed();

        for i in 1..=2 {
            let mut data = VertexData::new();
            data.set_f64("rank", i as f64 * 0.1);
            data.set_f64("distance", i as f64 * 10.0);
            builder.add_vertex_with_data(VertexId(i), data);
        }

        let graph = builder.build();

        let temp_file = NamedTempFile::new().unwrap();
        ResultExporter::export_multi_attribute(&graph, &["rank", "distance"], temp_file.path())
            .unwrap();

        // Verify JSON structure
        let contents = std::fs::read_to_string(temp_file.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

        assert_eq!(parsed["attributes"].as_array().unwrap().len(), 2);
        assert!(parsed["vertices"][0].as_object().unwrap().contains_key("rank"));
        assert!(parsed["vertices"][0].as_object().unwrap().contains_key("distance"));
    }
}
