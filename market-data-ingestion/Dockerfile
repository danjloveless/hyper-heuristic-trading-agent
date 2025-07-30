# Multi-stage Dockerfile for Market Data Ingestion Service

# Build stage
FROM rust:1.75-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY src/lib.rs src/lib.rs

# Create a dummy main file for initial dependency compilation
RUN mkdir -p src/bin && echo "fn main() {}" > src/bin/service.rs

# Build dependencies only
RUN cargo build --release --bin market-data-service
RUN rm -rf src/

# Copy actual source code
COPY src/ src/
COPY config/ config/

# Build the actual application
RUN cargo build --release --bin market-data-service

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Create app directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /usr/src/app/target/release/market-data-service /app/
COPY --from=builder /usr/src/app/config/ /app/config/

# Create logs directory
RUN mkdir -p /app/logs && chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the service
CMD ["./market-data-service", "--config", "config/production.toml"] 