#!/bin/bash
# Butterfly Comprehensive Test Runner
#
# This script runs the complete test suite including unit tests,
# integration tests, benchmarks, and code quality checks.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
RUN_BENCHMARKS=false
RUN_EXAMPLES=false
QUICK_MODE=false

# Print colored output
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${MAGENTA}================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}================================${NC}"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -b|--benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        -e|--examples)
            RUN_EXAMPLES=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose      Enable verbose output"
            echo "  -b, --benchmarks   Run benchmarks"
            echo "  -e, --examples     Build and test examples"
            echo "  -q, --quick        Quick mode (skip slow tests)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "🦋 Butterfly Test Suite"

# Start timing
START_TIME=$(date +%s)

# Check if we're in the project root
if [ ! -f "Cargo.toml" ]; then
    print_error "Must be run from project root directory"
    exit 1
fi

# Set logging level
if [ "$VERBOSE" = true ]; then
    export RUST_LOG=debug
else
    export RUST_LOG=info
fi

# 1. Code formatting check
print_header "Step 1: Code Formatting Check"
print_status "Running rustfmt..."
if cargo fmt --all -- --check; then
    print_success "Code is properly formatted"
else
    print_error "Code formatting issues found"
    print_status "Run 'cargo fmt' to fix formatting"
    exit 1
fi

# 2. Linting with Clippy
print_header "Step 2: Linting with Clippy"
print_status "Running clippy..."
CLIPPY_ARGS="--all-targets --all-features -- -D warnings"
if [ "$VERBOSE" = true ]; then
    CLIPPY_ARGS="$CLIPPY_ARGS -v"
fi
if cargo clippy $CLIPPY_ARGS; then
    print_success "No clippy warnings"
else
    print_error "Clippy found issues"
    exit 1
fi

# 3. Build project
print_header "Step 3: Building Project"
print_status "Building in debug mode..."
BUILD_ARGS=""
if [ "$VERBOSE" = true ]; then
    BUILD_ARGS="$BUILD_ARGS -v"
fi
if cargo build --all-features $BUILD_ARGS; then
    print_success "Build successful"
else
    print_error "Build failed"
    exit 1
fi

# 4. Unit tests
print_header "Step 4: Unit Tests"
print_status "Running unit tests..."
TEST_ARGS="--all-features"
if [ "$VERBOSE" = true ]; then
    TEST_ARGS="$TEST_ARGS -- --nocapture"
fi
if [ "$QUICK_MODE" = true ]; then
    TEST_ARGS="$TEST_ARGS --lib"
fi

if cargo test $TEST_ARGS; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# 5. Integration tests
if [ "$QUICK_MODE" = false ]; then
    print_header "Step 5: Integration Tests"
    print_status "Running integration tests..."
    if cargo test --test integration_test; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        exit 1
    fi
fi

# 6. Documentation tests
print_header "Step 6: Documentation Tests"
print_status "Running doc tests..."
if cargo test --doc; then
    print_success "Documentation tests passed"
else
    print_error "Documentation tests failed"
    exit 1
fi

# 7. Documentation build
print_header "Step 7: Documentation Build"
print_status "Building documentation..."
if cargo doc --no-deps --all-features; then
    print_success "Documentation built successfully"
else
    print_error "Documentation build failed"
    exit 1
fi

# 8. Examples (optional)
if [ "$RUN_EXAMPLES" = true ]; then
    print_header "Step 8: Building Examples"
    print_status "Building all examples..."

    for example in examples/*.rs; do
        if [ -f "$example" ]; then
            example_name=$(basename "$example" .rs)
            print_status "Building example: $example_name"
            if cargo build --example "$example_name"; then
                print_success "Example '$example_name' built successfully"
            else
                print_error "Example '$example_name' failed to build"
                exit 1
            fi
        fi
    done
fi

# 9. Benchmarks (optional)
if [ "$RUN_BENCHMARKS" = true ]; then
    print_header "Step 9: Running Benchmarks"
    print_status "Running criterion benchmarks..."
    if cargo bench --no-fail-fast; then
        print_success "Benchmarks completed"
    else
        print_warning "Some benchmarks failed"
    fi
fi

# 10. Security audit (if cargo-audit is installed)
print_header "Step 10: Security Audit"
if command -v cargo-audit >/dev/null 2>&1; then
    print_status "Running security audit..."
    if cargo audit; then
        print_success "No security vulnerabilities found"
    else
        print_warning "Security audit found issues"
    fi
else
    print_warning "cargo-audit not installed, skipping security audit"
    print_status "Install with: cargo install cargo-audit"
fi

# 11. Test coverage (if tarpaulin is installed)
if [ "$QUICK_MODE" = false ]; then
    print_header "Step 11: Test Coverage"
    if command -v cargo-tarpaulin >/dev/null 2>&1; then
        print_status "Generating test coverage report..."
        if cargo tarpaulin --out Html --output-dir coverage; then
            print_success "Coverage report generated in coverage/"
        else
            print_warning "Coverage generation failed"
        fi
    else
        print_warning "cargo-tarpaulin not installed, skipping coverage"
        print_status "Install with: cargo install cargo-tarpaulin"
    fi
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

print_header "Test Suite Complete! 🎉"

echo ""
echo "Summary:"
echo "  ✓ Code formatting: PASSED"
echo "  ✓ Linting: PASSED"
echo "  ✓ Build: PASSED"
echo "  ✓ Unit tests: PASSED"
if [ "$QUICK_MODE" = false ]; then
    echo "  ✓ Integration tests: PASSED"
fi
echo "  ✓ Documentation: PASSED"
if [ "$RUN_EXAMPLES" = true ]; then
    echo "  ✓ Examples: PASSED"
fi
if [ "$RUN_BENCHMARKS" = true ]; then
    echo "  ✓ Benchmarks: COMPLETED"
fi
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "All tests passed! 🦋"
