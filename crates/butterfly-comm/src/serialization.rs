//! Message serialization and compression utilities.
//!
//! Provides efficient binary serialization using bincode with optional
//! LZ4 compression for large messages.

use crate::{CommunicationError, Message};

/// Compression threshold in bytes (compress if message > 1KB)
const COMPRESSION_THRESHOLD: usize = 1024;

/// Protocol version for compatibility checking
const PROTOCOL_VERSION: u8 = 1;

/// Compression flags
const COMPRESSION_NONE: u8 = 0;
const COMPRESSION_LZ4: u8 = 1;

/// Serialize a message to bytes
pub fn serialize_message(msg: &Message) -> Result<Vec<u8>, CommunicationError> {
    // Serialize with bincode
    let data = bincode::serialize(msg)
        .map_err(|e| CommunicationError::Serialization(format!("Bincode error: {}", e)))?;

    // Check if compression would be beneficial
    let compressed = compress_if_beneficial(data)?;

    Ok(compressed)
}

/// Deserialize a message from bytes
pub fn deserialize_message(bytes: &[u8]) -> Result<Message, CommunicationError> {
    if bytes.is_empty() {
        return Err(CommunicationError::Serialization(
            "Empty message data".to_string(),
        ));
    }

    // Check protocol version
    let version = bytes[0];
    if version != PROTOCOL_VERSION {
        return Err(CommunicationError::Serialization(format!(
            "Unsupported protocol version: {}",
            version
        )));
    }

    // Check compression flag
    let compression = bytes[1];
    let data = match compression {
        COMPRESSION_NONE => &bytes[2..],
        COMPRESSION_LZ4 => {
            // Read original size (4 bytes)
            if bytes.len() < 6 {
                return Err(CommunicationError::Serialization(
                    "Invalid compressed message".to_string(),
                ));
            }
            let mut size_bytes = [0u8; 4];
            size_bytes.copy_from_slice(&bytes[2..6]);
            let original_size = u32::from_be_bytes(size_bytes) as usize;

            // Decompress
            let compressed = &bytes[6..];
            let decompressed = decompress_lz4(compressed, original_size)?;
            return bincode::deserialize(&decompressed)
                .map_err(|e| CommunicationError::Serialization(format!("Bincode error: {}", e)));
        }
        _ => {
            return Err(CommunicationError::Serialization(format!(
                "Unknown compression type: {}",
                compression
            )))
        }
    };

    bincode::deserialize(data)
        .map_err(|e| CommunicationError::Serialization(format!("Bincode error: {}", e)))
}

/// Compress data if it would reduce size significantly
fn compress_if_beneficial(data: Vec<u8>) -> Result<Vec<u8>, CommunicationError> {
    if data.len() < COMPRESSION_THRESHOLD {
        // Too small to benefit from compression
        let mut result = Vec::with_capacity(data.len() + 2);
        result.push(PROTOCOL_VERSION);
        result.push(COMPRESSION_NONE);
        result.extend_from_slice(&data);
        return Ok(result);
    }

    // Try compression
    match compress_lz4(&data) {
        Ok(compressed) => {
            // Check if compression was beneficial (at least 20% reduction)
            if compressed.len() < data.len() * 4 / 5 {
                // Store: version(1) + compression(1) + original_size(4) + compressed_data
                let mut result = Vec::with_capacity(compressed.len() + 6);
                result.push(PROTOCOL_VERSION);
                result.push(COMPRESSION_LZ4);
                result.extend_from_slice(&(data.len() as u32).to_be_bytes());
                result.extend_from_slice(&compressed);
                Ok(result)
            } else {
                // Compression didn't help, send uncompressed
                let mut result = Vec::with_capacity(data.len() + 2);
                result.push(PROTOCOL_VERSION);
                result.push(COMPRESSION_NONE);
                result.extend_from_slice(&data);
                Ok(result)
            }
        }
        Err(_) => {
            // Compression failed, send uncompressed
            let mut result = Vec::with_capacity(data.len() + 2);
            result.push(PROTOCOL_VERSION);
            result.push(COMPRESSION_NONE);
            result.extend_from_slice(&data);
            Ok(result)
        }
    }
}

/// Compress data using LZ4
fn compress_lz4(data: &[u8]) -> Result<Vec<u8>, CommunicationError> {
    lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(9)), false)
        .map_err(|e| CommunicationError::Serialization(format!("LZ4 compression error: {}", e)))
}

/// Decompress LZ4 data
fn decompress_lz4(data: &[u8], original_size: usize) -> Result<Vec<u8>, CommunicationError> {
    lz4::block::decompress(data, Some(original_size as i32))
        .map_err(|e| CommunicationError::Serialization(format!("LZ4 decompression error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use butterfly_core::NodeId;

    #[test]
    fn test_serialization_roundtrip() {
        let msg = Message::Heartbeat {
            node_id: NodeId(42),
        };

        let serialized = serialize_message(&msg).unwrap();
        let deserialized = deserialize_message(&serialized).unwrap();

        match deserialized {
            Message::Heartbeat { node_id } => assert_eq!(node_id, NodeId(42)),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_tensor_data_serialization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let msg = Message::TensorData {
            from: NodeId(1),
            to: NodeId(2),
            data: data.clone(),
        };

        let serialized = serialize_message(&msg).unwrap();
        let deserialized = deserialize_message(&serialized).unwrap();

        match deserialized {
            Message::TensorData {
                from,
                to,
                data: recv_data,
            } => {
                assert_eq!(from, NodeId(1));
                assert_eq!(to, NodeId(2));
                assert_eq!(recv_data, data);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_large_message_compression() {
        // Create large message that should be compressed
        let large_data = vec![0.0f32; 10000];
        let msg = Message::TensorData {
            from: NodeId(1),
            to: NodeId(2),
            data: large_data.clone(),
        };

        let serialized = serialize_message(&msg).unwrap();

        // Should be compressed (much smaller than original)
        let uncompressed_size = bincode::serialize(&msg).unwrap().len();
        assert!(serialized.len() < uncompressed_size / 2);

        // Should still deserialize correctly
        let deserialized = deserialize_message(&serialized).unwrap();
        match deserialized {
            Message::TensorData { data, .. } => assert_eq!(data, large_data),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_compression_not_applied_to_small_messages() {
        let msg = Message::Ack { message_id: 123 };
        let serialized = serialize_message(&msg).unwrap();

        // Check compression flag (byte 1)
        assert_eq!(serialized[1], COMPRESSION_NONE);
    }

    #[test]
    fn test_invalid_protocol_version() {
        let mut data = vec![99, 0, 1, 2, 3]; // Invalid version 99
        let result = deserialize_message(&data);
        assert!(result.is_err());
    }
}
