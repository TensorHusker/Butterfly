//! # Model Execution Engine
//!
//! Provides ML model execution capabilities.
//! Currently uses a mock implementation for testing. Real ONNX backend to be added.

use std::ops::Range;
use std::path::Path;
use thiserror::Error;

/// Executor errors
#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Invalid layer range: {0:?}")]
    InvalidLayerRange(Range<usize>),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("Inference error: {0}")]
    InferenceError(String),
}

/// Device type for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda(usize), // GPU device ID
    Metal,       // Apple Silicon
}

/// Model executor for transformer inference
///
/// **Note**: This is currently a mock implementation for integration testing.
/// Real ONNX/tract backend will be added in a future update.
pub struct ModelExecutor {
    /// Model input shape (batch, seq_len, hidden_dim)
    input_shape: Vec<usize>,

    /// Model output shape
    output_shape: Vec<usize>,

    /// Number of layers in the model
    num_layers: usize,

    /// Device the model runs on
    device: DeviceType,
}

impl ModelExecutor {
    /// Load an ONNX model from a file (mock implementation)
    pub fn load_onnx(_model_path: &Path, device_type: DeviceType) -> Result<Self, ExecutorError> {
        // Mock: Create a simple model configuration
        Ok(Self {
            input_shape: vec![1, 128, 768], // GPT-2 style
            output_shape: vec![1, 128, 50257], // vocab size
            num_layers: 12,
            device: device_type,
        })
    }

    /// Create a test model
    pub fn new_test_model(device_type: DeviceType) -> Result<Self, ExecutorError> {
        Ok(Self {
            input_shape: vec![1, 10, 768],
            output_shape: vec![1, 10, 768],
            num_layers: 12,
            device: device_type,
        })
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the model's device
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get input shape
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Get output shape
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Execute the model on input data (mock: returns zeros)
    pub fn execute(&self, input: &[f32]) -> Result<Vec<f32>, ExecutorError> {
        let expected_size: usize = self.input_shape.iter().product();
        if input.len() != expected_size {
            return Err(ExecutorError::ShapeMismatch {
                expected: format!("{:?} ({} elements)", self.input_shape, expected_size),
                actual: format!("{} elements", input.len()),
            });
        }

        // Mock: return zeros of output shape
        let output_size: usize = self.output_shape.iter().product();
        Ok(vec![0.0; output_size])
    }

    /// Execute specific layers (for distributed inference)
    pub fn execute_layers(
        &self,
        input: &[f32],
        layer_range: Range<usize>,
    ) -> Result<Vec<f32>, ExecutorError> {
        if layer_range.start >= self.num_layers || layer_range.end > self.num_layers {
            return Err(ExecutorError::InvalidLayerRange(layer_range));
        }

        if layer_range.start >= layer_range.end {
            return Err(ExecutorError::InvalidLayerRange(layer_range));
        }

        // Mock: just validate and return input (identity)
        Ok(input.to_vec())
    }

    /// Forward pass with token IDs
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, ExecutorError> {
        // Pad to expected input size
        let expected_size: usize = self.input_shape.iter().product();
        let mut input = vec![0.0; expected_size];

        for (i, &token) in token_ids.iter().enumerate().take(expected_size) {
            input[i] = token as f32;
        }

        self.execute(&input)
    }
}

/// Helper to convert tensor data
pub fn tensor_to_vec(data: &[f32]) -> Vec<f32> {
    data.to_vec()
}

/// Helper to create tensor from vec
pub fn vec_to_tensor(data: Vec<f32>, _shape: &[usize]) -> Result<Vec<f32>, ExecutorError> {
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type() {
        let device = DeviceType::Cpu;
        assert_eq!(device, DeviceType::Cpu);
    }

    #[test]
    fn test_executor_creation() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu);
        assert!(executor.is_ok());

        let executor = executor.unwrap();
        assert_eq!(executor.device(), DeviceType::Cpu);
        assert_eq!(executor.num_layers(), 12);
    }

    #[test]
    fn test_input_output_shapes() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        assert_eq!(executor.input_shape(), &[1, 10, 768]);
        assert_eq!(executor.output_shape(), &[1, 10, 768]);
    }

    #[test]
    fn test_execute() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let input_size: usize = executor.input_shape().iter().product();
        let input = vec![1.0; input_size];

        let output = executor.execute(&input).unwrap();
        assert_eq!(output.len(), executor.output_shape().iter().product::<usize>());
    }

    #[test]
    fn test_invalid_input_size() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let wrong_input = vec![1.0, 2.0, 3.0];

        let result = executor.execute(&wrong_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_layers() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let input_size: usize = executor.input_shape().iter().product();
        let input = vec![1.0; input_size];

        let result = executor.execute_layers(&input, 0..5);
        assert!(result.is_ok());

        // Invalid range
        let result = executor.execute_layers(&input, 0..20);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_with_tokens() {
        let executor = ModelExecutor::new_test_model(DeviceType::Cpu).unwrap();
        let token_ids = vec![1, 2, 3, 4, 5];

        let result = executor.forward(&token_ids);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_helpers() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let copied = tensor_to_vec(&data);
        assert_eq!(copied, data);

        let tensor = vec_to_tensor(data.clone(), &[2, 2]).unwrap();
        assert_eq!(tensor, data);
    }
}
