#!/bin/bash

# QuantumTrade AI Deployment Script
# This script deploys the entire system to AWS

set -e

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}
PROJECT_NAME="quantumtrade-ai"

echo "🚀 Deploying QuantumTrade AI to $ENVIRONMENT environment in $REGION"

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "❌ Terraform is required but not installed. Aborting." >&2; exit 1; }

# Validate AWS credentials
echo "🔐 Validating AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "❌ Failed to get AWS account ID. Please check your AWS credentials."
    exit 1
fi

echo "✅ AWS credentials validated. Account ID: $AWS_ACCOUNT_ID"

# Build and push Docker images
echo "🐳 Building and pushing Docker images..."

# Build shared libraries first
echo "📦 Building shared libraries..."
cargo build --release --workspace

# Build service images
SERVICES=("data-ingestion" "feature-engineering" "hyper-heuristic" "ml-inference" "api-gateway" "performance-monitor")

for service in "${SERVICES[@]}"; do
    echo "🔨 Building $service service..."
    docker build -t $PROJECT_NAME-$service:latest -f services/$service/Dockerfile .
    docker tag $PROJECT_NAME-$service:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME-$service:latest
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME-$service:latest
done

# Build Python explanation service
echo "🐍 Building explanation service..."
docker build -t $PROJECT_NAME-explanation-service:latest -f services/explanation-service/Dockerfile .
docker tag $PROJECT_NAME-explanation-service:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME-explanation-service:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$PROJECT_NAME-explanation-service:latest

# Deploy infrastructure with Terraform
echo "🏗️  Deploying infrastructure..."
cd terraform

# Initialize Terraform
terraform init

# Plan the deployment
terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$REGION" -out=tfplan

# Apply the plan
terraform apply tfplan

# Get outputs
CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
ALB_DNS=$(terraform output -raw alb_dns_name)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)

echo "✅ Infrastructure deployed successfully!"
echo "📊 Cluster: $CLUSTER_NAME"
echo "🌐 Load Balancer: $ALB_DNS"
echo "🔴 Redis: $REDIS_ENDPOINT"

# Deploy ECS services
echo "🚀 Deploying ECS services..."

# Update ECS services with new image tags
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-data-ingestion --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-feature-engineering --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-hyper-heuristic --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-ml-inference --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-explanation-service --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-api-gateway --force-new-deployment
aws ecs update-service --cluster $CLUSTER_NAME --service $PROJECT_NAME-performance-monitor --force-new-deployment

# Wait for services to be stable
echo "⏳ Waiting for services to stabilize..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services \
    $PROJECT_NAME-data-ingestion \
    $PROJECT_NAME-feature-engineering \
    $PROJECT_NAME-hyper-heuristic \
    $PROJECT_NAME-ml-inference \
    $PROJECT_NAME-explanation-service \
    $PROJECT_NAME-api-gateway \
    $PROJECT_NAME-performance-monitor

echo "✅ All services deployed and stable!"

# Run health checks
echo "🏥 Running health checks..."
sleep 30

# Test API endpoints
echo "🧪 Testing API endpoints..."
curl -f http://$ALB_DNS/health || echo "❌ Health check failed"

echo "🎉 Deployment completed successfully!"
echo "🌐 Application URL: http://$ALB_DNS"
echo "📊 CloudWatch Dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=$PROJECT_NAME" 