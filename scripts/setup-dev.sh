#!/bin/bash
# Butterfly Development Environment Setup Script
#
# This script sets up a complete development environment for Butterfly,
# including all dependencies, tools, and configurations.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_header "🦋 Butterfly Development Environment Setup"

# Check Rust installation
print_status "Checking Rust installation..."
if command_exists rustc; then
    RUST_VERSION=$(rustc --version)
    print_success "Rust is installed: $RUST_VERSION"
else
    print_error "Rust is not installed!"
    print_status "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    print_success "Rust installed successfully"
fi

# Check Cargo installation
print_status "Checking Cargo installation..."
if command_exists cargo; then
    CARGO_VERSION=$(cargo --version)
    print_success "Cargo is installed: $CARGO_VERSION"
else
    print_error "Cargo is not installed! Please install Rust toolchain."
    exit 1
fi

# Update Rust toolchain
print_status "Updating Rust toolchain..."
rustup update
print_success "Rust toolchain updated"

# Install required components
print_status "Installing Rust components..."
rustup component add rustfmt clippy
print_success "Rust components installed"

# Check for and install development tools
print_status "Checking development tools..."

if command_exists git; then
    print_success "Git is installed"
else
    print_error "Git is not installed! Please install Git."
    exit 1
fi

# Install cargo-watch for auto-recompilation
if command_exists cargo-watch; then
    print_success "cargo-watch is installed"
else
    print_status "Installing cargo-watch..."
    cargo install cargo-watch
    print_success "cargo-watch installed"
fi

# Install cargo-edit for dependency management
if command_exists cargo-add; then
    print_success "cargo-edit is installed"
else
    print_status "Installing cargo-edit..."
    cargo install cargo-edit
    print_success "cargo-edit installed"
fi

# Install cargo-nextest for faster testing
if command_exists cargo-nextest; then
    print_success "cargo-nextest is installed"
else
    print_status "Installing cargo-nextest..."
    cargo install cargo-nextest
    print_success "cargo-nextest installed"
fi

# Install cargo-criterion for benchmarking
if command_exists cargo-criterion; then
    print_success "cargo-criterion is installed"
else
    print_status "Installing cargo-criterion..."
    cargo install cargo-criterion
    print_success "cargo-criterion installed"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p tests/fixtures
mkdir -p benches
mkdir -p examples
mkdir -p scripts
mkdir -p target
print_success "Project directories created"

# Set up git hooks (if in git repo)
if [ -d .git ]; then
    print_status "Setting up git hooks..."

    # Pre-commit hook for formatting and linting
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook: Format and lint code

set -e

echo "Running pre-commit checks..."

# Format code
echo "  - Running rustfmt..."
cargo fmt --all -- --check

# Run clippy
echo "  - Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
echo "  - Running tests..."
cargo test --all

echo "✓ Pre-commit checks passed"
EOF

    chmod +x .git/hooks/pre-commit
    print_success "Git hooks configured"
else
    print_warning "Not a git repository, skipping git hooks setup"
fi

# Build the project
print_status "Building project..."
if cargo build; then
    print_success "Project built successfully"
else
    print_warning "Build failed, but setup will continue"
fi

# Run tests to ensure everything works
print_status "Running tests..."
if cargo test; then
    print_success "Tests passed"
else
    print_warning "Some tests failed, but setup is complete"
fi

# Set up environment variables
print_status "Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Butterfly Development Environment Variables

# Logging
RUST_LOG=info,butterfly=debug

# Network Configuration
BUTTERFLY_LISTEN_ADDR=127.0.0.1:9000
BUTTERFLY_MAX_CONNECTIONS=100

# Performance Tuning
BUTTERFLY_BATCH_SIZE=32
BUTTERFLY_WORKER_THREADS=4

# Development Settings
BUTTERFLY_DEV_MODE=true
EOF
    print_success "Environment file created (.env)"
else
    print_success "Environment file already exists"
fi

# Create VS Code configuration (optional)
print_status "Creating VS Code configuration..."
mkdir -p .vscode
if [ ! -f .vscode/settings.json ]; then
    cat > .vscode/settings.json << 'EOF'
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.features": "all",
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.tabSize": 4
  },
  "files.exclude": {
    "**/target": true
  }
}
EOF
    print_success "VS Code settings created"
else
    print_success "VS Code settings already exist"
fi

print_header "Setup Complete! 🎉"

echo "Development environment is ready. You can now:"
echo ""
echo "  1. Build the project:          cargo build"
echo "  2. Run tests:                  cargo test"
echo "  3. Run benchmarks:             cargo bench"
echo "  4. Run examples:               cargo run --example hello_butterfly"
echo "  5. Watch for changes:          cargo watch -x test"
echo "  6. Format code:                cargo fmt"
echo "  7. Run linter:                 cargo clippy"
echo ""
echo "Happy coding! 🦋"
