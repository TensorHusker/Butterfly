//! Bulk Synchronous Parallel (BSP) computation model
//!
//! This module implements the BSP computation paradigm for distributed graph processing.
//! The BSP model consists of supersteps, where each superstep has three phases:
//! 1. Local computation on each vertex
//! 2. Message passing to neighbor vertices
//! 3. Barrier synchronization across all workers

use crate::graph::{Vertex, VertexId, VertexData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A message sent between vertices during BSP computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Source vertex that sent the message
    pub source_id: VertexId,
    /// Target vertex to receive the message
    pub target_id: VertexId,
    /// Message payload
    pub payload: MessagePayload,
    /// Superstep number when message was sent
    pub superstep: u32,
}

impl Message {
    pub fn new(source: VertexId, target: VertexId, payload: MessagePayload) -> Self {
        Self {
            source_id: source,
            target_id: target,
            payload,
            superstep: 0,
        }
    }

    pub fn with_superstep(mut self, superstep: u32) -> Self {
        self.superstep = superstep;
        self
    }
}

/// Payload types that can be sent in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    F64(f64),
    U64(u64),
    I64(i64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Custom(Vec<u8>),
}

impl MessagePayload {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MessagePayload::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MessagePayload::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MessagePayload::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MessagePayload::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Result of a vertex computation in a superstep
#[derive(Debug, Clone)]
pub struct ComputationResult {
    /// New value to store in the vertex
    pub new_value: VertexData,
    /// Messages to send to other vertices
    pub outgoing_messages: Vec<Message>,
    /// Whether the vertex is still active (should participate in next superstep)
    pub is_active: bool,
}

impl ComputationResult {
    pub fn new(new_value: VertexData) -> Self {
        Self {
            new_value,
            outgoing_messages: Vec::new(),
            is_active: false,
        }
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.outgoing_messages = messages;
        self
    }

    pub fn active(mut self) -> Self {
        self.is_active = true;
        self
    }

    pub fn inactive(mut self) -> Self {
        self.is_active = false;
        self
    }
}

/// Core trait for vertex-centric computation in the BSP model
///
/// Implementations define how individual vertices process incoming messages
/// and generate outgoing messages during each superstep.
pub trait VertexCompute: Send + Sync {
    /// Compute function called for each active vertex in a superstep
    ///
    /// # Arguments
    /// * `vertex` - The current vertex being processed
    /// * `messages` - Messages received by this vertex in the previous superstep
    /// * `superstep` - Current superstep number
    ///
    /// # Returns
    /// ComputationResult containing updated vertex state, outgoing messages, and active status
    fn compute(
        &self,
        vertex: &Vertex,
        messages: &[Message],
        superstep: u32,
    ) -> Result<ComputationResult, ComputeError>;

    /// Initialize vertex state before first superstep (optional)
    fn init(&mut self, _vertex: &mut Vertex) -> Result<(), ComputeError> {
        Ok(())
    }

    /// Global aggregation function (optional)
    ///
    /// Called after each superstep to aggregate values across all vertices.
    /// For example, PageRank uses this to check convergence.
    fn aggregate(&self, _results: &[ComputationResult]) -> AggregatedResult {
        AggregatedResult::default()
    }

    /// Termination condition (optional)
    ///
    /// Return true when computation should terminate.
    /// Default: terminate when no vertices are active.
    fn should_terminate(&self, superstep: u32, active_vertices: usize) -> bool {
        active_vertices == 0 || superstep > 1000
    }
}

/// Aggregated result from a superstep across all vertices
#[derive(Debug, Clone, Default)]
pub struct AggregatedResult {
    /// Total number of active vertices
    pub active_count: usize,
    /// Total messages sent
    pub messages_sent: usize,
    /// Custom aggregation values
    pub aggregations: HashMap<String, f64>,
}

impl AggregatedResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_active_count(mut self, count: usize) -> Self {
        self.active_count = count;
        self
    }

    pub fn with_messages_sent(mut self, count: usize) -> Self {
        self.messages_sent = count;
        self
    }

    pub fn add_aggregation(&mut self, key: impl Into<String>, value: f64) {
        self.aggregations.insert(key.into(), value);
    }
}

/// Errors that can occur during BSP computation
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("Vertex not found: {0:?}")]
    VertexNotFound(VertexId),

    #[error("Invalid message format: {0}")]
    InvalidMessage(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(u32),
}

/// Result of a complete superstep execution
#[derive(Debug, Clone)]
pub struct SuperstepResult {
    /// Superstep number
    pub iteration: u32,
    /// Number of active vertices after this superstep
    pub active_vertices: usize,
    /// Number of messages sent during this superstep
    pub messages_sent: usize,
    /// Whether computation has converged
    pub converged: bool,
    /// Aggregated results
    pub aggregation: AggregatedResult,
}

impl SuperstepResult {
    pub fn new(iteration: u32) -> Self {
        Self {
            iteration,
            active_vertices: 0,
            messages_sent: 0,
            converged: false,
            aggregation: AggregatedResult::new(),
        }
    }
}

/// Message buffer for managing incoming and outgoing messages
#[derive(Debug, Clone)]
pub struct MessageBuffer {
    /// Messages grouped by target vertex
    incoming: HashMap<VertexId, Vec<Message>>,
    /// Messages to be sent out
    outgoing: Vec<Message>,
    /// Current superstep number
    current_superstep: u32,
}

impl MessageBuffer {
    pub fn new() -> Self {
        Self {
            incoming: HashMap::new(),
            outgoing: Vec::new(),
            current_superstep: 0,
        }
    }

    /// Add an incoming message to the buffer
    pub fn add_incoming(&mut self, message: Message) {
        self.incoming
            .entry(message.target_id)
            .or_insert_with(Vec::new)
            .push(message);
    }

    /// Add multiple incoming messages
    pub fn add_incoming_batch(&mut self, messages: Vec<Message>) {
        for message in messages {
            self.add_incoming(message);
        }
    }

    /// Add an outgoing message to the buffer
    pub fn add_outgoing(&mut self, mut message: Message) {
        message.superstep = self.current_superstep;
        self.outgoing.push(message);
    }

    /// Add multiple outgoing messages
    pub fn add_outgoing_batch(&mut self, messages: Vec<Message>) {
        for message in messages {
            self.add_outgoing(message);
        }
    }

    /// Get all messages for a specific vertex
    pub fn get_messages(&self, vertex_id: VertexId) -> &[Message] {
        self.incoming
            .get(&vertex_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Take all outgoing messages and clear the buffer
    pub fn take_outgoing(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.outgoing)
    }

    /// Clear incoming messages after they've been processed
    pub fn clear_incoming(&mut self) {
        self.incoming.clear();
    }

    /// Advance to next superstep
    pub fn next_superstep(&mut self) {
        self.current_superstep += 1;
        self.clear_incoming();
    }

    /// Get current superstep number
    pub fn current_superstep(&self) -> u32 {
        self.current_superstep
    }

    /// Get total number of incoming messages
    pub fn incoming_count(&self) -> usize {
        self.incoming.values().map(|v| v.len()).sum()
    }

    /// Get total number of outgoing messages
    pub fn outgoing_count(&self) -> usize {
        self.outgoing.len()
    }
}

impl Default for MessageBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Combiner function that reduces multiple messages to the same vertex
///
/// This is an optimization to reduce message volume. For example, in PageRank,
/// multiple rank contributions to the same vertex can be summed before sending.
pub trait MessageCombiner: Send + Sync {
    /// Combine multiple messages into a single message
    fn combine(&self, messages: Vec<Message>) -> Result<Message, ComputeError>;
}

/// Simple sum combiner for numeric messages
pub struct SumCombiner;

impl MessageCombiner for SumCombiner {
    fn combine(&self, messages: Vec<Message>) -> Result<Message, ComputeError> {
        let first = messages.first()
            .ok_or_else(|| ComputeError::InvalidMessage(
                "Cannot combine empty message list".to_string()
            ))?;

        let target = first.target_id;
        let source = first.source_id;

        let sum: f64 = messages
            .iter()
            .filter_map(|m| m.payload.as_f64())
            .sum();

        Ok(Message::new(source, target, MessagePayload::F64(sum)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::VertexId;

    #[test]
    fn test_message_creation() {
        let msg = Message::new(
            VertexId(1),
            VertexId(2),
            MessagePayload::F64(0.5),
        );

        assert_eq!(msg.source_id, VertexId(1));
        assert_eq!(msg.target_id, VertexId(2));
        assert_eq!(msg.payload.as_f64(), Some(0.5));
    }

    #[test]
    fn test_message_buffer() {
        let mut buffer = MessageBuffer::new();

        // Add incoming message
        let msg1 = Message::new(
            VertexId(1),
            VertexId(2),
            MessagePayload::F64(0.5),
        );
        buffer.add_incoming(msg1.clone());

        assert_eq!(buffer.incoming_count(), 1);
        assert_eq!(buffer.get_messages(VertexId(2)).len(), 1);

        // Add outgoing message
        buffer.add_outgoing(msg1);
        assert_eq!(buffer.outgoing_count(), 1);

        // Clear and advance
        buffer.next_superstep();
        assert_eq!(buffer.incoming_count(), 0);
        assert_eq!(buffer.current_superstep(), 1);
    }

    #[test]
    fn test_sum_combiner() {
        let combiner = SumCombiner;
        let messages = vec![
            Message::new(VertexId(1), VertexId(2), MessagePayload::F64(0.5)),
            Message::new(VertexId(1), VertexId(2), MessagePayload::F64(0.3)),
            Message::new(VertexId(1), VertexId(2), MessagePayload::F64(0.2)),
        ];

        let combined = combiner.combine(messages).unwrap();
        assert_eq!(combined.payload.as_f64(), Some(1.0));
    }

    #[test]
    fn test_sum_combiner_empty() {
        let combiner = SumCombiner;
        let messages = vec![];

        let result = combiner.combine(messages);
        assert!(result.is_err());
    }

    #[test]
    fn test_computation_result() {
        let mut data = VertexData::new();
        data.set_f64("value", 1.0);

        let result = ComputationResult::new(data)
            .with_messages(vec![Message::new(
                VertexId(1),
                VertexId(2),
                MessagePayload::F64(0.5),
            )])
            .active();

        assert_eq!(result.outgoing_messages.len(), 1);
        assert!(result.is_active);
    }
}
