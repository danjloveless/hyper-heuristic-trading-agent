#!/bin/bash
# scripts/dev-test.sh

set -e

echo "🧪 Running development tests..."

# Source environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Ensure services are running
if ! docker-compose -f docker-compose.dev.yml ps | grep -q "healthy"; then
    echo "🚀 Starting development services..."
    docker-compose -f docker-compose.dev.yml up -d
    sleep 10
fi

# Run tests
echo "🔧 Building workspace..."
cargo build --workspace

echo "🧪 Running unit tests..."
cargo test --workspace

echo "🗄️  Testing database integration..."
cargo test -p shared-utils --features integration-tests

echo "✅ All tests passed!"