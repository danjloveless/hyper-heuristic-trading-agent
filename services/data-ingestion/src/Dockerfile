# services/data-ingestion/Dockerfile
FROM rust:1.88 as builder

# Create app directory
WORKDIR /usr/src/app

# Copy workspace files
COPY Cargo.toml ./
COPY shared/ ./shared/

# Copy service-specific files
COPY services/data-ingestion/ ./services/data-ingestion/

# Build dependencies first (for better caching)
RUN cargo build --release --bin data-ingestion

# Runtime stage
FROM debian:bookworm-slim

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd -r -s /bin/false appuser

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /usr/src/app/target/release/data-ingestion ./data-ingestion

# Copy configuration template
COPY services/data-ingestion/config/ ./config/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Health check (using built-in health check instead of curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["./data-ingestion", "--health-check"] || exit 1

# Expose port
EXPOSE 3001

# Set default environment variables
ENV RUST_LOG=info
ENV PORT=3001
ENV CONFIG_PATH=/app/config/data-ingestion.toml

# Run the binary
CMD ["./data-ingestion"]