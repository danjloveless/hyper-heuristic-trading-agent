#!/bin/bash
# scripts/dev-reset.sh

echo "🔄 Resetting development environment..."

# Stop and remove containers, networks, and volumes
docker-compose -f docker-compose.dev.yml down -v

# Remove dangling images
docker image prune -f

# Restart services
echo "🚀 Restarting services..."
docker-compose -f docker-compose.dev.yml up -d

echo "✅ Development environment reset complete!"