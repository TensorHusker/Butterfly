//! Core types for the coordination protocol

use butterfly_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Messages exchanged in the coordination protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    /// Phase 1: Coordinator assigns work to nodes
    WorkAssignment(WorkAssignment),

    /// Phase 2: Node signals readiness at barrier
    BarrierReady(NodeId, CheckpointHash),

    /// Phase 2: Coordinator releases barrier
    BarrierRelease(Epoch),

    /// Phase 4: Coordinator proposes result (PRE-PREPARE)
    PrePrepare(InferenceResult, Proof),

    /// Phase 4: Node validates and votes (PREPARE)
    Prepare(ResultHash),

    /// Phase 4: Node commits result (COMMIT)
    Commit(InferenceResult),

    /// Failure detection: Heartbeat
    Heartbeat(NodeId, f64), // node_id, phi_value

    /// Failure detection: Suspicion report
    Suspicion(NodeId, FailureEvidence),

    /// Recovery: Request checkpoint
    CheckpointRequest(Epoch),

    /// Recovery: Checkpoint data
    CheckpointResponse(CheckpointData),
}

/// Work assignment mapping layers to nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    pub epoch: Epoch,
    pub assignments: HashMap<NodeId, LayerAssignment>,
    pub dependencies: DependencyGraph,
}

/// Layers assigned to a specific node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub node_id: NodeId,
    pub layer_ids: Vec<usize>,
    pub input_dependencies: Vec<NodeId>,
    pub output_consumers: Vec<NodeId>,
    pub estimated_compute_time_ms: f64,
}

/// Dependency graph between node computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// edges[from_node] = [to_node1, to_node2, ...]
    pub edges: HashMap<NodeId, Vec<NodeId>>,
    /// Topologically sorted execution order
    pub execution_order: Vec<NodeId>,
}

/// Result of inference computation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceResult {
    pub epoch: Epoch,
    pub data: Vec<u8>, // Serialized tensor data
    pub hash: ResultHash,
}

/// Cryptographic proof of correct computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub layer_id: usize,
    pub input_hash: [u8; 32],
    pub output_hash: [u8; 32],
    pub intermediate_checksums: Vec<[u8; 32]>,
    pub signature: Vec<u8>,
    pub timestamp: i64,
}

/// Evidence of node failure or Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureEvidence {
    /// Node unresponsive to heartbeats
    Unresponsive {
        last_seen: i64,
        phi_value: f64,
    },
    /// Node produced incorrect result
    IncorrectComputation {
        expected_hash: ResultHash,
        actual_hash: ResultHash,
        proof: Proof,
    },
    /// Node sent conflicting messages (equivocation)
    Equivocation {
        message1: Box<CoordinationMessage>,
        message2: Box<CoordinationMessage>,
    },
    /// Node violated protocol
    ProtocolViolation {
        description: String,
        evidence_hash: [u8; 32],
    },
}

/// Checkpoint data for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub epoch: Epoch,
    pub token_position: usize,
    pub node_states: HashMap<NodeId, Vec<u8>>, // Serialized state
    pub intermediate_results: HashMap<NodeId, Vec<u8>>,
    pub metadata: CheckpointMetadata,
}

/// Metadata about a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub timestamp: i64,
    pub version: u32,
    pub checksum: [u8; 32],
}

/// Type aliases for clarity
pub type Epoch = u64;
pub type CheckpointHash = [u8; 32];
pub type ResultHash = [u8; 32];

/// Errors that can occur in coordination
#[derive(Debug, thiserror::Error)]
pub enum CoordinationError {
    #[error("Invalid phase transition from {from:?} to {to:?}")]
    InvalidPhaseTransition {
        from: String,
        to: String,
    },

    #[error("Quorum not reached: {current}/{required} nodes")]
    QuorumNotReached {
        current: usize,
        required: usize,
    },

    #[error("Byzantine behavior detected: {0}")]
    ByzantineDetected(String),

    #[error("Checkpoint not found: epoch {0}")]
    CheckpointNotFound(Epoch),

    #[error("Invalid proof: {0}")]
    InvalidProof(String),

    #[error("Node not operational: {0:?}")]
    NodeNotOperational(NodeId),

    #[error("Communication error: {0}")]
    Communication(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl Proof {
    /// Verify proof integrity
    pub fn verify(&self) -> bool {
        // TODO: Implement signature verification with Ed25519
        // For now, basic sanity checks
        !self.signature.is_empty()
            && self.input_hash != [0u8; 32]
            && self.output_hash != [0u8; 32]
    }

    /// Generate proof from computation
    pub fn generate(
        layer_id: usize,
        input: &[u8],
        output: &[u8],
        intermediate: &[&[u8]],
        signing_key: &[u8],
    ) -> Self {
        use sha2::{Sha256, Digest};

        let input_hash = Sha256::digest(input).into();
        let output_hash = Sha256::digest(output).into();
        let intermediate_checksums: Vec<[u8; 32]> = intermediate
            .iter()
            .map(|data| Sha256::digest(data).into())
            .collect();

        // TODO: Sign with Ed25519
        let signature = signing_key.to_vec();

        Self {
            layer_id,
            input_hash,
            output_hash,
            intermediate_checksums,
            signature,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

impl InferenceResult {
    /// Create result from tensor data
    pub fn new(epoch: Epoch, data: Vec<u8>) -> Self {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(&data).into();
        Self { epoch, data, hash }
    }

    /// Verify result hash matches data
    pub fn verify_hash(&self) -> bool {
        use sha2::{Sha256, Digest};
        let computed: [u8; 32] = Sha256::digest(&self.data).into();
        computed == self.hash
    }
}

impl DependencyGraph {
    /// Create new empty dependency graph
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    /// Add dependency edge
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.entry(from).or_insert_with(Vec::new).push(to);
    }

    /// Compute topological sort for execution order
    pub fn compute_execution_order(&mut self, nodes: &[NodeId]) -> Result<(), String> {
        // Kahn's algorithm for topological sort
        let mut in_degree: HashMap<NodeId, usize> = nodes.iter().map(|&n| (n, 0)).collect();

        // Calculate in-degrees
        for successors in self.edges.values() {
            for &successor in successors {
                *in_degree.entry(successor).or_insert(0) += 1;
            }
        }

        // Queue of nodes with in-degree 0
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);

            if let Some(successors) = self.edges.get(&node) {
                for &successor in successors {
                    if let Some(deg) = in_degree.get_mut(&successor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(successor);
                        }
                    }
                }
            }
        }

        if result.len() != nodes.len() {
            return Err("Dependency graph contains cycle".to_string());
        }

        self.execution_order = result;
        Ok(())
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_result_hash_verification() {
        let data = vec![1, 2, 3, 4, 5];
        let result = InferenceResult::new(0, data);
        assert!(result.verify_hash());
    }

    #[test]
    fn test_dependency_graph_topological_sort() {
        let mut graph = DependencyGraph::new();
        let n0 = NodeId(0);
        let n1 = NodeId(1);
        let n2 = NodeId(2);

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);

        let nodes = vec![n0, n1, n2];
        graph.compute_execution_order(&nodes).unwrap();

        assert_eq!(graph.execution_order, vec![n0, n1, n2]);
    }

    #[test]
    fn test_proof_generation() {
        let input = b"input data";
        let output = b"output data";
        let intermediate = vec![b"step1".as_slice(), b"step2".as_slice()];
        let key = b"signing_key";

        let proof = Proof::generate(0, input, output, &intermediate, key);

        assert_eq!(proof.layer_id, 0);
        assert_ne!(proof.input_hash, [0u8; 32]);
        assert_ne!(proof.output_hash, [0u8; 32]);
        assert_eq!(proof.intermediate_checksums.len(), 2);
    }
}
