//! Zero-copy tensor representations and SIMD-optimized operations

use std::sync::Arc;
use std::ops::Range;

/// Zero-copy tensor reference using shared ownership
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Shared ownership of underlying data
    data: Arc<[f32]>,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Offset into the data buffer (for slicing)
    offset: usize,
    /// Stride information for multi-dimensional indexing
    strides: Vec<usize>,
}

impl TensorRef {
    /// Create a new tensor from owned data
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::from(data.into_boxed_slice()),
            shape,
            offset: 0,
            strides,
        }
    }

    /// Create a tensor from existing Arc
    pub fn from_arc(data: Arc<[f32]>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            offset: 0,
            strides,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a slice view of the tensor data
    pub fn as_slice(&self) -> &[f32] {
        let len = self.len();
        &self.data[self.offset..self.offset + len]
    }

    /// Zero-copy slice operation (shares underlying data)
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.end <= self.len(), "Slice range out of bounds");
        let new_shape = vec![range.len()];
        Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            offset: self.offset + range.start,
            strides: vec![1],
        }
    }

    /// Zero-copy reshape (view with different shape)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<Self> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return None;
        }

        Some(Self {
            data: Arc::clone(&self.data),
            shape: new_shape.clone(),
            offset: self.offset,
            strides: Self::compute_strides(&new_shape),
        })
    }

    /// Compute strides for row-major layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Check if this is the only reference to the data
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }
}

/// Memory pool for tensor allocation
pub struct TensorPool {
    /// Pools organized by size class (powers of 2)
    pools: Vec<parking_lot::Mutex<Vec<Arc<[f32]>>>>,
    /// Statistics
    allocations: std::sync::atomic::AtomicU64,
    reuses: std::sync::atomic::AtomicU64,
}

impl TensorPool {
    /// Create a new tensor pool with size classes up to 2^max_power elements
    pub fn new(max_power: usize) -> Self {
        let pools = (0..=max_power)
            .map(|_| parking_lot::Mutex::new(Vec::new()))
            .collect();

        Self {
            pools,
            allocations: std::sync::atomic::AtomicU64::new(0),
            reuses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Acquire a buffer of at least the given size
    pub fn acquire(&self, size: usize) -> Arc<[f32]> {
        let size_class = self.size_to_class(size);

        if size_class < self.pools.len() {
            if let Some(buffer) = self.pools[size_class].lock().pop() {
                self.reuses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return buffer;
            }
        }

        // Allocate new buffer
        self.allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.allocate_aligned(size)
    }

    /// Release a buffer back to the pool
    pub fn release(&self, buffer: Arc<[f32]>) {
        // Only pool if we're the last reference
        if Arc::strong_count(&buffer) == 1 {
            let size = buffer.len();
            let size_class = self.size_to_class(size);

            if size_class < self.pools.len() {
                self.pools[size_class].lock().push(buffer);
            }
        }
    }

    /// Convert size to power-of-2 size class
    fn size_to_class(&self, size: usize) -> usize {
        if size == 0 {
            return 0;
        }
        let bits = std::mem::size_of::<usize>() * 8;
        bits - size.leading_zeros() as usize - 1
    }

    /// Allocate cache-aligned memory for SIMD operations
    fn allocate_aligned(&self, size: usize) -> Arc<[f32]> {
        // Round up to next power of 2 for better pooling
        let alloc_size = size.next_power_of_two();

        // Allocate with standard allocator (Rust's allocator provides good alignment)
        let vec = vec![0.0f32; alloc_size];
        Arc::from(vec.into_boxed_slice())
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_allocations: self
                .allocations
                .load(std::sync::atomic::Ordering::Relaxed),
            total_reuses: self.reuses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// Statistics about pool usage
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub total_allocations: u64,
    pub total_reuses: u64,
}

impl Default for TensorPool {
    fn default() -> Self {
        // Default: support tensors up to 2^30 = ~1 billion elements
        Self::new(30)
    }
}

/// SIMD-optimized tensor operations
pub mod simd {
    use super::TensorRef;

    /// Add two tensors element-wise with SIMD optimization
    pub fn add(a: &TensorRef, b: &TensorRef) -> TensorRef {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Tensors must have same shape for addition"
        );

        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let mut result = vec![0.0f32; a_slice.len()];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            add_avx2(a_slice, b_slice, &mut result);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            add_scalar(a_slice, b_slice, &mut result);
        }

        TensorRef::new(result, a.shape().to_vec())
    }

    /// Scalar fallback for addition
    fn add_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    /// AVX2-accelerated addition
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 8;

        // Process 8 elements at a time with AVX
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            let sum = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(idx), sum);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            result[i] = a[i] + b[i];
        }
    }

    /// Multiply tensor by scalar with SIMD
    pub fn scale(tensor: &TensorRef, scalar: f32) -> TensorRef {
        let data = tensor.as_slice();
        let mut result = vec![0.0f32; data.len()];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            scale_avx2(data, scalar, &mut result);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            scale_scalar(data, scalar, &mut result);
        }

        TensorRef::new(result, tensor.shape().to_vec())
    }

    fn scale_scalar(data: &[f32], scalar: f32, result: &mut [f32]) {
        for i in 0..data.len() {
            result[i] = data[i] * scalar;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn scale_avx2(data: &[f32], scalar: f32, result: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let len = data.len();
        let chunks = len / 8;
        let scalar_vec = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let idx = i * 8;
            let data_vec = _mm256_loadu_ps(data.as_ptr().add(idx));
            let result_vec = _mm256_mul_ps(data_vec, scalar_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(idx), result_vec);
        }

        for i in (chunks * 8)..len {
            result[i] = data[i] * scalar;
        }
    }

    /// Dot product with SIMD acceleration
    pub fn dot(a: &TensorRef, b: &TensorRef) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length for dot product");

        let a_slice = a.as_slice();
        let b_slice = b.as_slice();

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            return dot_avx2(a_slice, b_slice);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            dot_scalar(a_slice, b_slice)
        }
    }

    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 8;
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }

        // Horizontal sum of SIMD register
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum);
        let mut result: f32 = temp.iter().sum();

        // Add remaining elements
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = TensorRef::new(data, vec![2, 2]);

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_zero_copy_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = TensorRef::new(data, vec![5]);

        let sliced = tensor.slice(1..4);
        assert_eq!(sliced.as_slice(), &[2.0, 3.0, 4.0]);
        assert_eq!(sliced.len(), 3);
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorRef::new(data, vec![6]);

        let reshaped = tensor.reshape(vec![2, 3]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.len(), 6);
    }

    #[test]
    fn test_tensor_pool() {
        let pool = TensorPool::new(20);

        let buf1 = pool.acquire(1024);
        assert_eq!(buf1.len(), 1024);

        pool.release(buf1);

        let buf2 = pool.acquire(1024);
        let stats = pool.stats();

        assert!(stats.total_reuses > 0);
    }

    #[test]
    fn test_simd_add() {
        let a = TensorRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = TensorRef::new(vec![5.0, 6.0, 7.0, 8.0], vec![4]);

        let result = simd::add(&a, &b);
        assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_simd_scale() {
        let tensor = TensorRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = simd::scale(&tensor, 2.0);
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_dot() {
        let a = TensorRef::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = TensorRef::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = simd::dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }
}
