# Data Ingestion Service

The Data Ingestion Service is a high-performance Rust microservice responsible for collecting financial data from multiple sources including Yahoo Finance, Reddit, and news APIs. It provides both real-time and scheduled data collection with automatic sentiment analysis.

## Features

- **Multi-Source Data Collection**: Yahoo Finance (market data), Reddit (sentiment), News APIs (sentiment)
- **Automated Scheduling**: Background jobs with configurable cron expressions  
- **High Performance**: Rust-based with async/await for maximum throughput
- **Rate Limiting**: Built-in rate limiting and retry logic for external APIs
- **Batch Processing**: Efficient batch operations for database inserts
- **Health Monitoring**: Comprehensive health checks and status reporting
- **Configurable**: TOML-based configuration with environment variable overrides

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Yahoo Finance │    │     Reddit      │    │    News APIs    │
│      API        │    │      API        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                ┌─────────────────▼─────────────────┐
                │      Data Ingestion Service      │
                │                                   │
                │  • HTTP API endpoints             │
                │  • Background scheduler           │
                │  • Rate limiting & retries        │
                │  • Sentiment analysis             │
                │  • Batch processing               │
                └─────────────────┬─────────────────┘
                                 │
                ┌─────────────────▼─────────────────┐
                │         ClickHouse               │
                │     (Time Series Storage)         │
                └───────────────────────────────────┘
```

## API Endpoints

### Health & Status
- `GET /health` - Service health check
- `GET /api/v1/internal/ingest/status` - Ingestion status for all sources

### Data Collection Triggers
- `POST /api/v1/internal/ingest/yahoo` - Trigger Yahoo Finance collection
- `POST /api/v1/internal/ingest/yahoo/{symbol}` - Collect data for specific symbol
- `POST /api/v1/internal/ingest/reddit` - Trigger Reddit sentiment collection  
- `POST /api/v1/internal/ingest/reddit/{symbol}` - Reddit data for specific symbol
- `POST /api/v1/internal/ingest/news` - Trigger news sentiment collection
- `POST /api/v1/internal/ingest/news/{symbol}` - News data for specific symbol

### Scheduler Management
- `POST /api/v1/admin/scheduler/start` - Start background scheduler
- `POST /api/v1/admin/scheduler/stop` - Stop background scheduler
- `GET /api/v1/admin/scheduler/status` - Get scheduler status
- `GET /api/v1/admin/config` - Get current configuration (sanitized)

## Configuration

### Environment Variables

```bash
# Database Configuration
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_DATABASE=quantumtrade
CLICKHOUSE_USERNAME=default
CLICKHOUSE_PASSWORD=

# Service Configuration  
PORT=3001
RUST_LOG=info
CONFIG_PATH=/app/config/data-ingestion.toml
ENABLE_SCHEDULER=true

# API Credentials (Required)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret  
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
NEWS_API_KEY=your_newsapi_key

# Rate Limiting
YAHOO_FINANCE_RATE_LIMIT=60
REDDIT_RATE_LIMIT=60
NEWS_RATE_LIMIT=100

# Symbols
PRIMARY_SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA,META
```

### Configuration File

The service uses a TOML configuration file located at `config/data-ingestion.toml`. Key sections:

- **yahoo_finance**: Yahoo Finance API settings
- **reddit**: Reddit API credentials and subreddit configuration
- **news**: News API settings and source selection
- **scheduler**: Cron expressions for automated collection
- **symbols**: Stock symbols to track
- **storage**: Database batch settings

## Setup & Development

### Prerequisites

1. **Rust 1.75+** with Cargo
2. **ClickHouse** database running
3. **Redis** for caching (optional but recommended)
4. **API Credentials**:
   - Reddit: Create app at https://www.reddit.com/prefs/apps
   - NewsAPI: Get key from https://newsapi.org

### Local Development

1. **Clone and build**:
```bash
git clone <repository>
cd quantumtrade-ai
cargo build --bin data-ingestion
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

3. **Start dependencies**:
```bash
docker-compose -f docker-compose.dev.yml up -d clickhouse redis
```

4. **Run the service**:
```bash
cargo run --bin data-ingestion
```

### Docker Development

1. **Build and run all services**:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

2. **Check service health**:
```bash
curl http://localhost:3001/health
```

3. **Trigger manual collection**:
```bash
# Collect Yahoo Finance data
curl -X POST http://localhost:3001/api/v1/internal/ingest/yahoo

# Collect Reddit sentiment  
curl -X POST http://localhost:3001/api/v1/internal/ingest/reddit

# Collect news sentiment
curl -X POST http://localhost:3001/api/v1/internal/ingest/news
```

## Data Sources

### Yahoo Finance
- **Data Type**: OHLCV market data, real-time quotes
- **Frequency**: Every 5 minutes (configurable)
- **Symbols**: Stocks, ETFs, crypto (BTC-USD, ETH-USD)
- **Rate Limit**: 60 requests/minute
- **Free**: Yes

### Reddit
- **Data Type**: Post sentiment from financial subreddits
- **Sources**: r/wallstreetbets, r/stocks, r/investing, r/SecurityAnalysis
- **Frequency**: Every 15 minutes (configurable)  
- **Sentiment**: Keyword-based analysis with confidence scoring
- **Rate Limit**: 60 requests/minute
- **Auth Required**: Yes (OAuth2)

### News APIs
- **Data Type**: Financial news sentiment
- **Sources**: Bloomberg, Reuters, WSJ, Financial Post
- **Frequency**: Every 30 minutes (configurable)
- **Sentiment**: Enhanced analysis with symbol relevance scoring
- **Rate Limit**: 100 requests/minute
- **Auth Required**: Yes (API key)

## Scheduled Jobs

The service runs background jobs based on cron expressions:

- **Market Data**: `0 */5 * * * *` (every 5 minutes)
- **Reddit Sentiment**: `0 */15 * * * *` (every 15 minutes) 
- **News Sentiment**: `0 */30 * * * *` (every 30 minutes)
- **Data Cleanup**: `0 0 2 * * *` (daily at 2 AM)

### Market Hours Detection

When `enable_market_hours_only` is true, the service will:
- Skip market data collection outside trading hours
- Continue sentiment collection (24/7 social media)
- Respect US market holidays

## Performance

### Throughput Targets
- **Market Data**: 1000+ symbols/minute
- **Reddit Posts**: 500+ posts/minute analyzed
- **News Articles**: 200+ articles/minute processed
- **Database Inserts**: 10,000+ records/minute

### Resource Usage
- **Memory**: ~100-200 MB typical usage
- **CPU**: Low usage with burst during collection
- **Network**: Respects API rate limits
- **Storage**: Batch inserts for efficiency

## Monitoring

### Health Checks
```bash
# Service health
curl http://localhost:3001/health

# Detailed status
curl http://localhost:3001/api/v1/internal/ingest/status

# Scheduler status  
curl http://localhost:3001/api/v1/admin/scheduler/status
```

### Logs
The service provides structured JSON logging:
```bash
# View logs
docker logs quantumtrade-data-ingestion

# Follow logs
docker logs -f quantumtrade-data-ingestion
```

### Metrics
Key metrics to monitor:
- Collection success rates per source
- API response times and error rates  
- Database insert performance
- Scheduler job execution times
- Memory and CPU usage

## Testing

### Unit Tests
```bash
cargo test --lib
```

### Integration Tests  
```bash
# Requires running ClickHouse and Redis
cargo test --test integration
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:3001/health

# Test manual collection
curl -X POST http://localhost:3001/api/v1/internal/ingest/yahoo?symbols=AAPL&limit=10
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check ClickHouse is running: `curl http://localhost:8123/ping`
   - Verify connection settings in config/environment

2. **API Authentication Errors**
   - Verify Reddit credentials are valid
   - Check NewsAPI key is active and has quota
   - Ensure environment variables are set correctly

3. **Rate Limiting**
   - Monitor API usage against limits
   - Adjust rate_limit_per_minute in configuration
   - Check for 429 errors in logs

4. **Scheduler Not Running**
   - Check `ENABLE_SCHEDULER=true` in environment
   - Verify cron expressions are valid
   - Look for scheduler errors in logs

5. **Missing Data**
   - Check symbol validity (Yahoo Finance supports the symbol)
   - Verify API responses are not empty
   - Look for parsing errors in logs

### Debug Mode
```bash
# Enable debug logging
RUST_LOG=debug cargo run --bin data-ingestion

# Or in Docker
docker-compose -f docker-compose.dev.yml up -d --build \
  -e RUST_LOG=debug data-ingestion
```

## Security

- **API Keys**: Store in environment variables, never in code
- **Rate Limiting**: Built-in protection against API abuse  
- **Input Validation**: All external data is validated and sanitized
- **Error Handling**: Graceful failure without exposing internals
- **Least Privilege**: Service runs as non-root user in container

## Performance Tuning

### Database Optimization
- Increase `batch_size` for higher throughput
- Adjust `buffer_timeout_seconds` for latency vs throughput
- Enable `compression` for storage savings

### API Optimization  
- Tune `max_retries` and `retry_delay_seconds` for reliability
- Adjust rate limits based on API quotas
- Use `max_symbols_per_batch` to optimize Yahoo Finance calls

### Memory Management
- Set `max_memory_buffer_mb` to control memory usage
- Monitor RSS memory usage in production
- Consider reducing batch sizes if memory constrained

## Contributing

1. Follow Rust best practices and use `cargo clippy`
2. Add tests for new functionality  
3. Update configuration schema when adding options
4. Document API changes in this README
5. Test with actual API credentials before submitting PRs

## License

This project is licensed under the MIT License.