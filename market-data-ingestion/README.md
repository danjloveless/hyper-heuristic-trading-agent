# QuantumTrade AI Market Data Ingestion Service

A high-performance, real-time market data ingestion service built with Rust for the QuantumTrade AI platform. This service collects, validates, and stores financial market data from various sources including Alpha Vantage API.

## 🚀 Features

- **Intelligent Data Collection**: Avoids redundant API calls by checking data freshness and collecting only incremental data
- **Real-time Data Collection**: Collects market data from Alpha Vantage API with configurable intervals
- **High Performance**: Built with Rust for optimal performance and memory efficiency
- **Data Quality**: Built-in validation, deduplication, and quality checks
- **Scalable Architecture**: Supports concurrent collections and batch processing
- **Multiple Environments**: Separate configurations for development, testing, and production
- **Health Monitoring**: Comprehensive health checks and metrics
- **Rate Limiting**: Intelligent rate limiting to respect API quotas
- **Docker Support**: Full containerization with Docker and Docker Compose

## 📋 Prerequisites

- **Rust 1.75+** and Cargo
- **Docker** and **Docker Compose**
- **Alpha Vantage API Key** (free tier available)
- **PowerShell 5.1+**

## 🛠️ Quick Start

### 1. Clone and Setup

```powershell
# Clone the repository
git clone <repository-url>
cd market-data-ingestion

# Run setup script
.\scripts\setup.ps1
```

### 2. Configure Environment

```powershell
# Copy environment file
Copy-Item env.example .env

# Edit .env and set your API key
# ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### 3. Start Development

```powershell
.\scripts\start-dev.ps1
```

### 4. Test the Service

```powershell
.\scripts\test-api.ps1
```

## 📁 Project Structure

```
market-data-ingestion/
├── config/                     # Configuration files
│   ├── production.toml        # Production settings
│   ├── development.toml       # Development settings
│   ├── testing.toml          # Testing settings
│   └── clickhouse/           # ClickHouse configuration
├── scripts/                   # Utility scripts
│   ├── setup.ps1             # Initial setup
│   ├── start-dev.ps1         # Development startup
│   ├── test-api.ps1          # API testing
│   └── benchmark.ps1         # Performance testing
├── src/                      # Source code
│   ├── lib.rs               # Library entry point
│   └── bin/
│       └── service.rs       # Service binary
├── tests/                   # Integration tests
├── examples/                # Usage examples
├── docker-compose.yml       # Docker services
├── Dockerfile              # Container definition
├── Cargo.toml             # Rust dependencies
└── README.md              # This file
```

## ⚙️ Configuration

### Environment Variables

Key environment variables in `.env`:

```powershell
# Required
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Database
DATABASE_URL=clickhouse://localhost:8123/quantumtrade
REDIS_URL=redis://localhost:6379

# Service
SERVICE_PORT=8080
WORKER_THREADS=4
MAX_CONCURRENT_COLLECTIONS=50

# Logging
RUST_LOG=market_data_ingestion=info,service=info
```

### Configuration Files

The service uses TOML configuration files for different environments:

- **`config/development.toml`**: Development settings with reduced limits
- **`config/production.toml`**: Production settings with full capabilities
- **`config/testing.toml`**: Testing settings with minimal resources

### Key Configuration Sections

#### Service Configuration
```toml
[service]
service_name = "market-data-ingestion"
port = 8080
worker_threads = 8
max_concurrent_collections = 100
```

#### Alpha Vantage API
```toml
[alpha_vantage]
base_url = "https://www.alphavantage.co/query"
api_key = "${ALPHA_VANTAGE_API_KEY}"
timeout_seconds = 30
max_retries = 5
```

#### Rate Limiting
```toml
[rate_limits]
calls_per_minute = 75
calls_per_day = 75000
is_premium = true
```

#### Data Collection
```toml
[collection]
default_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
priority_symbols = ["SPY", "QQQ", "VIX"]
collection_intervals = ["1min", "5min", "15min", "30min", "1hour", "1day"]
max_batch_size = 5000
parallel_collections = 25
```

## 🐳 Docker Deployment

### Using Docker Compose

```powershell
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f market-data-ingestion

# Stop services
docker-compose down
```

### Using Docker

```powershell
# Build image
docker build -t market-data-ingestion .

# Run container
docker run -d `
  --name market-data-ingestion `
  -p 8080:8080 `
  -e ALPHA_VANTAGE_API_KEY=your_key `
  market-data-ingestion
```

## 🔌 API Endpoints

### Health and Monitoring

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health information
- `GET /metrics` - Service metrics
- `GET /info` - Service information

### Data Collection

- `POST /collect/{symbol}?interval={interval}&force={true|false}` - Collect data for a symbol (intelligent collection)
- `POST /collect/batch` - Collect data for multiple symbols
- `GET /collections` - List recent collections
- `GET /freshness/{symbol}?interval={interval}` - Check data freshness for a symbol

### Configuration

- `GET /config` - Current configuration
- `GET /config/symbols` - Configured symbols

### Administrative

- `POST /admin/trigger-collection` - Trigger collection for all configured symbols
- `POST /admin/force-collection/{symbol}` - Force collection for a symbol (bypasses freshness checks)

### Examples

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8080/health"

# Collect data for AAPL
Invoke-RestMethod -Uri "http://localhost:8080/collect/AAPL?interval=5min" -Method POST

# Get service info
Invoke-RestMethod -Uri "http://localhost:8080/info"

# Get metrics
Invoke-RestMethod -Uri "http://localhost:8080/metrics"
```

## 🧪 Testing

### Run Tests

```powershell
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# Performance tests
cargo test --release test_performance
```

### API Testing

```powershell
.\scripts\test-api.ps1
```

### Performance Benchmarking

```powershell
.\scripts\benchmark.ps1
```

## 📊 Monitoring and Metrics

The service provides comprehensive monitoring:

### Health Checks
- Service health status
- Database connectivity
- API key validity
- Rate limit status

### Metrics
- Data points collected
- Collection success rate
- Processing time
- Error rates
- Memory usage

### Logging
- Structured logging with different levels
- Request/response logging
- Error tracking
- Performance metrics

## 🧠 Intelligent Data Collection

The service implements intelligent data collection that minimizes API calls and optimizes data freshness:

### How It Works

1. **Data Freshness Check**: Before making an API call, the service checks the latest timestamp in the database
2. **Freshness Thresholds**: Different intervals have different freshness thresholds:
   - 1-minute data: 2 minutes
   - 5-minute data: 7 minutes  
   - 15-minute data: 20 minutes
   - 30-minute data: 35 minutes
   - 1-hour data: 65 minutes
3. **Incremental Collection**: Only fetches data points newer than the latest timestamp
4. **Skip Logic**: If data is fresh, the API call is skipped entirely

### API Usage

```powershell
# Normal collection (uses intelligent logic)
Invoke-RestMethod -Uri "http://localhost:8080/collect/AAPL?interval=5min" -Method POST

# Force collection (bypasses freshness checks)
Invoke-RestMethod -Uri "http://localhost:8080/collect/AAPL?interval=5min&force=true" -Method POST

# Check data freshness
Invoke-RestMethod -Uri "http://localhost:8080/freshness/AAPL?interval=5min"
```

### Example Response

```json
{
  "success": true,
  "message": "Data for AAPL is fresh, no new data collected",
  "data_points_collected": 0,
  "processing_time_ms": 15,
  "batch_id": "batch_123",
  "skip_reason": "data_fresh",
  "latest_timestamp": "2024-01-15T16:00:00Z",
  "force_collection": null
}
```

### Benefits

- **Reduced API Costs**: Minimizes redundant API calls
- **Better Performance**: Faster response times for fresh data
- **Rate Limit Optimization**: Respects API rate limits more effectively
- **Data Consistency**: Ensures data continuity without gaps

## 🔧 Development

### Building

```powershell
# Debug build
cargo build

# Release build
cargo build --release

# Specific binary
cargo build --bin market-data-service
```

### Running

```powershell
# Development mode
cargo run --bin market-data-service -- --config config/development.toml

# Production mode
cargo run --release --bin market-data-service -- --config config/production.toml
```

### Adding New Features

1. **Configuration**: Add new settings to TOML files
2. **API Endpoints**: Add handlers in `src/bin/service.rs`
3. **Data Processing**: Extend the library in `src/lib.rs`
4. **Tests**: Add unit and integration tests
5. **Documentation**: Update README and API docs

## 🚨 Troubleshooting

### Common Issues

1. **Docker not running**
   ```powershell
   # Check Docker status
   docker info
   
   # Start Docker Desktop
   # Or start Docker service
   Start-Service docker
   ```

2. **Port conflicts**
   ```powershell
   # Check port usage
   netstat -an | Select-String "8080"
   
   # Kill process using port
   Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   ```

3. **API key issues**
   ```powershell
   # Verify API key
   Invoke-RestMethod -Uri "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey=YOUR_KEY"
   ```

4. **Database connection issues**
   ```powershell
   # Check ClickHouse
   Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%201"
   
   # Check Redis
   docker exec (docker-compose ps -q redis) redis-cli ping
   ```

### Debug Mode

```powershell
# Enable debug logging
$env:RUST_LOG="market_data_ingestion=debug"; cargo run

# Verbose Docker logs
docker-compose logs -f --tail=100 market-data-ingestion
```

## 📈 Performance

### Benchmarks

The service is optimized for high-performance data ingestion:

- **Throughput**: Up to 10,000 data points/second
- **Latency**: < 100ms average response time
- **Memory**: < 512MB typical usage
- **Concurrency**: Supports 100+ concurrent collections

### Optimization Tips

1. **Use release builds** for production
2. **Adjust worker threads** based on CPU cores
3. **Configure batch sizes** for optimal throughput
4. **Monitor memory usage** and adjust limits
5. **Use connection pooling** for database connections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### Code Style

- Follow Rust conventions
- Use meaningful variable names
- Add comments for complex logic
- Include error handling
- Write comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the development team

## 🔗 Related Projects

- **QuantumTrade AI Platform**: Main trading platform
- **Database Abstraction Layer**: Shared database utilities
- **Configuration Management**: Centralized configuration service
- **Logging & Monitoring**: Unified logging and metrics

---

**Note**: This service is part of the QuantumTrade AI platform. For complete platform documentation, see the main project repository.