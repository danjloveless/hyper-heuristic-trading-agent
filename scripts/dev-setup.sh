#!/bin/bash
# scripts/dev-setup.sh

set -e

echo "🚀 Setting up QuantumTrade AI development environment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.development .env
    echo "✅ .env file created. Please review and modify as needed."
fi

# Start the development environment
echo "🐳 Starting development services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
timeout=120
counter=0

while [ $counter -lt $timeout ]; do
    if docker-compose -f docker-compose.dev.yml ps | grep -q "healthy"; then
        break
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    echo "❌ Services failed to start within $timeout seconds"
    docker-compose -f docker-compose.dev.yml logs
    exit 1
fi

echo ""
echo "✅ Development environment is ready!"
echo ""
echo "📊 Service URLs:"
echo "  - ClickHouse HTTP: http://localhost:8123"
echo "  - Redis: localhost:6379"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose -f docker-compose.dev.yml logs -f"
echo "  - Stop services: docker-compose -f docker-compose.dev.yml down"
echo "  - Reset data: docker-compose -f docker-compose.dev.yml down -v"
echo "  - ClickHouse client: docker-compose -f docker-compose.dev.yml run --rm clickhouse-client"
echo "  - Redis client: docker-compose -f docker-compose.dev.yml run --rm redis-client"
echo ""
echo "🏗️  Next steps:"
echo "  1. Run 'cargo build' to build the project"
echo "  2. Run 'cargo test -p shared-utils' to test the ClickHouse client"
echo "  3. Start developing your services!"