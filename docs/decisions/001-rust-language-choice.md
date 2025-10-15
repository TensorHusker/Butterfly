# ADR-001: Use Rust as Primary Implementation Language

**Status**: Accepted

**Date**: 2025-10-11

## Context

Butterfly is a distributed inference system that requires:
1. High performance for processing large tensors and model operations
2. Memory safety to prevent crashes and security vulnerabilities
3. Efficient concurrency for managing multiple workers and asynchronous I/O
4. Low-level control over memory layout and system resources
5. Strong correctness guarantees for complex distributed algorithms

The choice of programming language fundamentally shapes the development velocity, system reliability, and runtime performance. We need a language that can handle the demands of high-performance distributed systems while maintaining developer productivity.

Key constraints:
- System must handle multi-GB tensors efficiently
- Network communication requires careful memory management
- Concurrent access to shared state must be safe
- Performance must be comparable to C/C++ implementations
- Codebase must be maintainable by a small team

## Decision

We will use Rust as the primary implementation language for all Butterfly components (orchestrator, worker, and common library).

Rust provides:
- Zero-cost abstractions without runtime overhead
- Memory safety without garbage collection
- Fearless concurrency through ownership system
- Strong type system catching errors at compile time
- Modern tooling (cargo, clippy, rustfmt)
- Growing ecosystem for systems and ML infrastructure

## Consequences

### Positive Consequences

**Memory Safety Without GC Overhead**
- Ownership system prevents use-after-free, double-free, and data races at compile time
- No garbage collection pauses during latency-sensitive inference
- Predictable memory usage patterns critical for resource management
- Eliminates entire classes of bugs common in C/C++ distributed systems

**Fearless Concurrency**
- Type system prevents data races by design
- Safe abstractions over async/await for efficient I/O
- Channels and mutexes are statically verified for correctness
- Easier to implement complex coordination protocols without subtle race conditions

**Performance Comparable to C/C++**
- Zero-cost abstractions compile to efficient machine code
- Manual memory layout control when needed
- SIMD support for tensor operations
- Efficient FFI for integrating with existing ML libraries (ONNX Runtime, etc.)

**Strong Type System**
- Algebraic data types model domain concepts precisely
- Pattern matching eliminates missing case bugs
- Type inference reduces boilerplate
- Traits enable modular, testable abstractions

**Excellent Tooling**
- Cargo for dependency management and building
- Built-in testing framework
- Clippy for catching common mistakes
- Rustfmt for consistent code style
- Excellent IDE support (rust-analyzer)

**Growing ML/AI Ecosystem**
- ndarray for n-dimensional arrays
- tokio/async-std for async runtime
- serde for serialization
- tonic for gRPC (potential future protocol)
- burn, candle for native ML operations

### Negative Consequences

**Steeper Learning Curve**
- Ownership and borrowing concepts are unfamiliar to developers from GC languages
- Lifetimes can be challenging for complex data structures
- Async Rust has additional complexity (Send/Sync bounds)
- Error messages, while improving, can be verbose for beginners

**Slower Initial Development**
- Fighting the borrow checker during prototyping phase
- More time spent on type design and API boundaries
- Refactoring can require significant type system changes
- Less "just make it work" rapid iteration compared to Python

**Smaller Talent Pool**
- Fewer developers proficient in Rust compared to Python/Java/C++
- Hiring may take longer
- Onboarding new team members requires Rust training
- Less StackOverflow content for esoteric issues

**Ecosystem Still Maturing**
- ML libraries less mature than Python equivalents (PyTorch, TensorFlow)
- Some patterns still being established by community
- Occasional breaking changes in popular crates
- Fewer third-party libraries compared to older languages

**Compile Times**
- Large projects can have significant compile times
- Incremental compilation helps but is imperfect
- May slow down development iteration
- CI/CD builds take longer than interpreted languages

### Neutral Consequences

**Explicit Error Handling**
- Result and Option types force handling of error cases
- Eliminates unexpected exceptions in production
- Requires more upfront thought about error propagation
- Can be verbose but prevents bugs

**FFI Requirements**
- Will need unsafe code to interface with C/C++ ML libraries
- Requires careful auditing of unsafe blocks
- Bindings generation tools (bindgen) help but add complexity
- Creates boundary between safe Rust and unsafe external code

**Community Culture**
- Strong emphasis on documentation and best practices
- Active discussion of API design patterns
- Regular breaking changes in pre-1.0 crates
- Helpful but opinionated community

## Alternatives Considered

### Alternative 1: Python

**Description**: Implement orchestrator and workers in Python, using native extensions for performance-critical paths.

**Pros**:
- Fastest development velocity for prototyping
- Largest ML/AI ecosystem (PyTorch, TensorFlow, Transformers)
- Easy to find developers
- Excellent for experimentation

**Cons**:
- GIL prevents true parallelism
- Memory overhead of interpreter
- GC pauses unpredictable
- Difficult to guarantee memory safety
- Poor concurrency primitives
- Requires C/C++ extensions for performance

**Why Rejected**: Python's GIL and GC pauses are problematic for a low-latency distributed system. While great for prototyping, production systems require predictable performance. The need for C extensions undermines Python's development velocity advantage.

### Alternative 2: C++

**Description**: Modern C++ (C++17/20) with careful use of RAII, smart pointers, and standard library.

**Pros**:
- Mature ecosystem for systems programming
- Excellent performance
- Direct integration with ML libraries (ONNX Runtime, TensorRT)
- Large talent pool
- Full control over memory layout

**Cons**:
- Memory safety requires constant vigilance
- Easy to introduce undefined behavior
- No protection against data races
- Complex build systems (CMake)
- Difficult to audit for security
- Slower compilation than Rust
- Error-prone memory management

**Why Rejected**: While C++ offers excellent performance, the lack of memory safety guarantees makes it risky for a complex distributed system. The ownership discipline that Rust enforces is possible in C++ but requires perfect human vigilance. We value correctness over raw development speed.

### Alternative 3: Go

**Description**: Use Go for its strong concurrency primitives and simplicity.

**Pros**:
- Excellent concurrency (goroutines, channels)
- Fast compilation
- Simple language, easy to learn
- Good for network services
- Growing systems programming ecosystem

**Cons**:
- Garbage collection (pauses for large heaps)
- No generics (until recently, still limited)
- Less control over memory layout
- Weaker type system than Rust
- Interface{} escape hatch undermines safety
- Poor support for SIMD/performance-critical code
- Smaller ML ecosystem

**Why Rejected**: Go's GC is problematic for processing multi-GB tensors. While goroutines are great, Rust's async/await provides similar benefits with more control. Go's simplicity is appealing but comes at the cost of expressiveness needed for complex algorithms.

### Alternative 4: Java/JVM

**Description**: Use Java or Kotlin with JVM ecosystem.

**Pros**:
- Mature ecosystem
- Strong typing
- Good concurrency primitives (java.util.concurrent)
- Large talent pool
- Excellent tooling

**Cons**:
- GC pauses (even with low-latency GCs)
- JVM warmup time
- Memory overhead
- Poor control over memory layout
- FFI (JNI) is painful
- Not ideal for systems programming

**Why Rejected**: JVM's GC and memory overhead make it unsuitable for high-performance tensor processing. The warmup time is problematic for worker startup. While ZGC improves pause times, it's still not deterministic enough for our needs.

### Alternative 5: Hybrid Approach (Python + Rust)

**Description**: Python for orchestrator, Rust for workers.

**Pros**:
- Leverages Python's rapid development for control plane
- Rust performance for data plane
- Best of both worlds

**Cons**:
- Two languages increases complexity
- Shared code requires bindings
- Different testing ecosystems
- Harder to reason about full system
- Split talent requirements

**Why Rejected**: While tempting, the complexity of maintaining two languages outweighs the benefits. The orchestrator has performance requirements too (handling many concurrent requests). A single language simplifies development and allows code reuse.

## Implementation Notes

### Phase 1: Core Infrastructure (Current)
- Use stable Rust (edition 2021)
- Leverage tokio for async runtime
- Use serde for serialization
- Keep unsafe code minimal and well-documented

### Phase 2: Performance Optimization
- Profile hot paths with perf/flamegraph
- Consider SIMD intrinsics where beneficial
- Evaluate GPU acceleration via CUDA/ROCm bindings
- Optimize memory allocations with custom allocators

### Phase 3: Production Hardening
- Audit all unsafe code
- Fuzz test serialization paths
- Memory sanitizer in CI
- Continuous benchmarking

### Development Guidelines

**Code Organization**:
- Use workspace for modular crates
- Common library for shared types
- Clear separation of concerns

**Error Handling**:
- Custom error types using thiserror
- Propagate errors with ? operator
- Log errors at boundaries
- Never panic in production code paths

**Testing**:
- Unit tests for pure functions
- Integration tests for components
- Property tests for algorithms
- Chaos testing for distributed scenarios

**Performance**:
- Benchmark critical paths
- Profile before optimizing
- Document performance characteristics
- Use release builds for perf testing

## References

- [Rust Book](https://doc.rust-lang.org/book/)
- [Async Rust Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Are We Learning Yet?](https://www.arewelearningyet.com/) - ML ecosystem status

## Revision History

| Date | Author | Change Description |
|------|--------|--------------------|
| 2025-10-11 | Agent-4 | Initial version |

## Notes

This decision recognizes that Rust has a learning curve but believes the long-term benefits of memory safety and performance outweigh the short-term velocity costs. The choice prioritizes correctness and reliability over rapid prototyping.

We acknowledge that some team members may need Rust training, and we commit to providing resources and mentorship. The investment in learning Rust will pay dividends in reduced debugging time and fewer production incidents.

If the language ecosystem or team constraints change dramatically, we may revisit this decision, but as of now, Rust is the best fit for Butterfly's requirements.
