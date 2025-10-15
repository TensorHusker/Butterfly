//! Byzantine agreement implementation using modified PBFT
//!
//! Implements a three-phase Byzantine agreement protocol:
//! 1. PRE-PREPARE: Coordinator proposes result
//! 2. PREPARE: Nodes validate and vote
//! 3. COMMIT: Nodes commit after quorum reached
//!
//! Optimistic fast path: If all nodes agree immediately, skip PREPARE phase

use crate::types::{CoordinationError, InferenceResult, Proof, ResultHash};
use butterfly_core::NodeId;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Phase of Byzantine agreement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementPhase {
    /// Waiting for coordinator proposal
    Idle,
    /// PRE-PREPARE received, validating
    PrePrepare,
    /// PREPARE phase, collecting votes
    Prepare,
    /// COMMIT phase, finalizing
    Commit,
    /// Agreement reached
    Committed,
}

/// Byzantine agreement coordinator
#[derive(Debug)]
pub struct ByzantineAgreement {
    /// Current phase
    phase: AgreementPhase,
    /// Cluster size
    cluster_size: usize,
    /// Maximum Byzantine failures
    max_byzantine: usize,
    /// Quorum size (2f + 1)
    quorum_size: usize,
    /// Proposed results awaiting agreement
    proposals: HashMap<ResultHash, (InferenceResult, Proof)>,
    /// Nodes that sent PREPARE for each result
    prepare_votes: HashMap<ResultHash, HashSet<NodeId>>,
    /// Nodes that sent COMMIT for each result
    commit_votes: HashMap<ResultHash, HashSet<NodeId>>,
    /// Committed result (if any)
    committed_result: Option<InferenceResult>,
    /// Whether optimistic fast path succeeded
    fast_path_active: bool,
}

impl ByzantineAgreement {
    /// Create new Byzantine agreement instance
    pub fn new(cluster_size: usize, max_byzantine: usize) -> Self {
        let quorum_size = 2 * max_byzantine + 1;

        Self {
            phase: AgreementPhase::Idle,
            cluster_size,
            max_byzantine,
            quorum_size,
            proposals: HashMap::new(),
            prepare_votes: HashMap::new(),
            commit_votes: HashMap::new(),
            committed_result: None,
            fast_path_active: false,
        }
    }

    /// Handle PRE-PREPARE message from coordinator
    pub async fn handle_pre_prepare(
        &mut self,
        result: InferenceResult,
        proof: Proof,
    ) -> Result<(), CoordinationError> {
        if self.phase != AgreementPhase::Idle {
            warn!(?self.phase, "Received PRE-PREPARE in wrong phase");
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.phase),
                to: "PrePrepare".to_string(),
            });
        }

        // Validate proof
        if !proof.verify() {
            return Err(CoordinationError::InvalidProof(
                "Proof verification failed".to_string(),
            ));
        }

        // Validate result hash
        if !result.verify_hash() {
            return Err(CoordinationError::InvalidProof(
                "Result hash mismatch".to_string(),
            ));
        }

        info!(hash = ?result.hash, "PRE-PREPARE received, validating result");

        self.proposals.insert(result.hash, (result.clone(), proof));
        self.phase = AgreementPhase::PrePrepare;

        // Check if we can use optimistic fast path
        // (requires all nodes to agree immediately)
        self.fast_path_active = false;

        Ok(())
    }

    /// Handle PREPARE vote from node
    pub async fn handle_prepare(
        &mut self,
        result_hash: ResultHash,
    ) -> Result<Option<ResultHash>, CoordinationError> {
        if !matches!(
            self.phase,
            AgreementPhase::PrePrepare | AgreementPhase::Prepare
        ) {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.phase),
                to: "Prepare".to_string(),
            });
        }

        // Verify the result hash matches a known proposal
        if !self.proposals.contains_key(&result_hash) {
            return Err(CoordinationError::InvalidProof(
                "Unknown result hash".to_string(),
            ));
        }

        self.phase = AgreementPhase::Prepare;

        // Note: In real implementation, would track which node sent the vote
        // to prevent double-voting. Simplified here.
        let votes = self
            .prepare_votes
            .entry(result_hash)
            .or_insert_with(HashSet::new);

        // Simulate vote from a node
        let dummy_voter = NodeId(votes.len() as u64);
        votes.insert(dummy_voter);

        debug!(
            hash = ?result_hash,
            votes = votes.len(),
            required = self.quorum_size,
            "PREPARE vote received"
        );

        // Check if we reached quorum
        if votes.len() >= self.quorum_size {
            info!(hash = ?result_hash, "PREPARE quorum reached, advancing to COMMIT");
            return Ok(Some(result_hash));
        }

        Ok(None)
    }

    /// Record PREPARE vote from specific node
    pub fn record_prepare_vote(
        &mut self,
        node_id: NodeId,
        result_hash: ResultHash,
    ) -> Result<bool, CoordinationError> {
        if !self.proposals.contains_key(&result_hash) {
            return Err(CoordinationError::InvalidProof(
                "Unknown result hash".to_string(),
            ));
        }

        let votes = self
            .prepare_votes
            .entry(result_hash)
            .or_insert_with(HashSet::new);

        votes.insert(node_id);

        debug!(
            ?node_id,
            hash = ?result_hash,
            votes = votes.len(),
            "Recorded PREPARE vote"
        );

        Ok(votes.len() >= self.quorum_size)
    }

    /// Handle COMMIT message from node
    pub async fn handle_commit(
        &mut self,
        result: InferenceResult,
    ) -> Result<bool, CoordinationError> {
        if !matches!(self.phase, AgreementPhase::Prepare | AgreementPhase::Commit) {
            return Err(CoordinationError::InvalidPhaseTransition {
                from: format!("{:?}", self.phase),
                to: "Commit".to_string(),
            });
        }

        // Verify PREPARE quorum was reached
        let prepare_count = self
            .prepare_votes
            .get(&result.hash)
            .map(|v| v.len())
            .unwrap_or(0);

        if prepare_count < self.quorum_size {
            return Err(CoordinationError::QuorumNotReached {
                current: prepare_count,
                required: self.quorum_size,
            });
        }

        self.phase = AgreementPhase::Commit;

        let votes = self
            .commit_votes
            .entry(result.hash)
            .or_insert_with(HashSet::new);

        let dummy_voter = NodeId(votes.len() as u64);
        votes.insert(dummy_voter);

        debug!(
            hash = ?result.hash,
            votes = votes.len(),
            required = self.quorum_size,
            "COMMIT vote received"
        );

        // Check if we reached commit quorum
        if votes.len() >= self.quorum_size {
            info!(hash = ?result.hash, "COMMIT quorum reached, finalizing");
            self.phase = AgreementPhase::Committed;
            self.committed_result = Some(result);
            return Ok(true);
        }

        Ok(false)
    }

    /// Record COMMIT vote from specific node
    pub fn record_commit_vote(
        &mut self,
        node_id: NodeId,
        result_hash: ResultHash,
    ) -> Result<bool, CoordinationError> {
        let votes = self
            .commit_votes
            .entry(result_hash)
            .or_insert_with(HashSet::new);

        votes.insert(node_id);

        debug!(
            ?node_id,
            hash = ?result_hash,
            votes = votes.len(),
            "Recorded COMMIT vote"
        );

        let committed = votes.len() >= self.quorum_size;

        if committed {
            self.phase = AgreementPhase::Committed;
            // Retrieve result from proposals
            if let Some((result, _proof)) = self.proposals.get(&result_hash) {
                self.committed_result = Some(result.clone());
            }
        }

        Ok(committed)
    }

    /// Get committed result if agreement reached
    pub fn committed_result(&self) -> Option<&InferenceResult> {
        self.committed_result.as_ref()
    }

    /// Check if agreement reached
    pub fn is_committed(&self) -> bool {
        self.phase == AgreementPhase::Committed
    }

    /// Get current phase
    pub fn phase(&self) -> AgreementPhase {
        self.phase
    }

    /// Get vote count for a result
    pub fn prepare_vote_count(&self, result_hash: &ResultHash) -> usize {
        self.prepare_votes
            .get(result_hash)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get commit vote count for a result
    pub fn commit_vote_count(&self, result_hash: &ResultHash) -> usize {
        self.commit_votes
            .get(result_hash)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Reset for next round
    pub fn reset(&mut self) {
        self.phase = AgreementPhase::Idle;
        self.proposals.clear();
        self.prepare_votes.clear();
        self.commit_votes.clear();
        self.committed_result = None;
        self.fast_path_active = false;
    }

    /// Try optimistic fast path (skip PREPARE if all agree immediately)
    pub fn try_fast_path(&mut self, result_hash: ResultHash) -> bool {
        // Check if all nodes immediately agreed
        let immediate_votes = self
            .prepare_votes
            .get(&result_hash)
            .map(|v| v.len())
            .unwrap_or(0);

        if immediate_votes == self.cluster_size {
            info!("Fast path activated, skipping PREPARE phase");
            self.fast_path_active = true;
            self.phase = AgreementPhase::Commit;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result() -> InferenceResult {
        InferenceResult::new(0, vec![1, 2, 3, 4, 5])
    }

    fn create_test_proof() -> Proof {
        Proof::generate(0, b"input", b"output", &[b"step1"], b"key")
    }

    #[tokio::test]
    async fn test_agreement_flow() {
        let mut agreement = ByzantineAgreement::new(5, 2);

        let result = create_test_result();
        let proof = create_test_proof();

        // PRE-PREPARE
        agreement
            .handle_pre_prepare(result.clone(), proof)
            .await
            .unwrap();
        assert_eq!(agreement.phase(), AgreementPhase::PrePrepare);

        // PREPARE votes (need 5 for quorum)
        for i in 0..5 {
            let reached = agreement
                .record_prepare_vote(NodeId(i), result.hash)
                .unwrap();
            if i >= 4 {
                assert!(reached);
            }
        }

        // COMMIT votes
        for i in 0..5 {
            let committed = agreement
                .record_commit_vote(NodeId(i), result.hash)
                .unwrap();
            if i >= 4 {
                assert!(committed);
            }
        }

        assert!(agreement.is_committed());
        assert_eq!(agreement.committed_result().unwrap(), &result);
    }

    #[test]
    fn test_quorum_calculation() {
        let agreement = ByzantineAgreement::new(7, 2);
        assert_eq!(agreement.quorum_size, 5);
    }

    #[tokio::test]
    async fn test_invalid_proof_rejection() {
        let mut agreement = ByzantineAgreement::new(5, 2);

        let mut result = create_test_result();
        result.hash = [0u8; 32]; // Invalid hash

        let proof = create_test_proof();

        let err = agreement.handle_pre_prepare(result, proof).await;
        assert!(err.is_err());
    }
}
