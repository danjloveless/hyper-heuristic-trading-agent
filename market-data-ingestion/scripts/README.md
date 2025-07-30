# Market Data Ingestion Service Scripts

This directory contains PowerShell scripts to help with development, testing, and deployment of the Market Data Ingestion Service.

## Available Scripts

### Setup Scripts

#### `setup.ps1`
Initial setup script that:
- Checks for required dependencies (Cargo, Docker, Docker Compose)
- Creates necessary directories
- Copies configuration files
- Builds the project
- Starts databases (ClickHouse, Redis)
- Runs tests
- Provides next steps

**Usage:**
```powershell
# Basic setup
.\scripts\setup.ps1

# Skip tests
.\scripts\setup.ps1 -SkipTests

# Skip build
.\scripts\setup.ps1 -SkipBuild
```

### Development Scripts

#### `start-dev.ps1`
Starts the service in development mode:
- Loads environment variables
- Checks for API key
- Starts databases if not running
- Runs the service with debug logging

**Usage:**
```powershell
.\scripts\start-dev.ps1
```

### Testing Scripts

#### `test-api.ps1`
Tests all API endpoints:
- Health endpoint
- Service info
- Metrics
- Configuration
- Symbol collection (if API key available)
- Symbols endpoint
- Detailed health
- Data freshness check

**Usage:**
```powershell
.\scripts\test-api.ps1
```

### Performance Scripts

#### `benchmark.ps1`
Runs performance benchmarks:
- Builds in release mode
- Runs Cargo benchmarks
- Executes performance tests
- Tests memory efficiency
- Tests high volume processing

**Usage:**
```powershell
.\scripts\benchmark.ps1
```

## Prerequisites

Before running any scripts, ensure you have:

1. **Rust and Cargo** installed
2. **Docker** and **Docker Compose** installed
3. **Alpha Vantage API Key** (set in `.env` file)
4. **PowerShell 5.1+**

## Environment Setup

1. Copy `env.example` to `.env`
2. Set your `ALPHA_VANTAGE_API_KEY` in the `.env` file
3. Review and adjust configuration in `config/development.toml`

## Quick Start

1. **Initial Setup:**
   ```powershell
   .\scripts\setup.ps1
   ```

2. **Start Development:**
   ```powershell
   .\scripts\start-dev.ps1
   ```

3. **Test API:**
   ```powershell
   .\scripts\test-api.ps1
   ```

## Troubleshooting

### Common Issues

1. **Docker not running:**
   - Ensure Docker Desktop is started
   - Check Docker service status

2. **Port conflicts:**
   - Check if ports 8080, 8123, 6379 are available
   - Stop conflicting services

3. **API key issues:**
   - Verify `ALPHA_VANTAGE_API_KEY` is set in `.env`
   - Check API key validity

4. **Database connection issues:**
   - Ensure Docker containers are running
   - Check container health with `docker-compose ps`

### Debug Mode

For debugging, you can run scripts with verbose output:

```powershell
# PowerShell debug mode
$VerbosePreference = "Continue"
.\scripts\setup.ps1
```

## Script Customization

All scripts can be customized by editing the configuration files:

- `config/production.toml` - Production settings
- `config/development.toml` - Development settings
- `config/testing.toml` - Testing settings
- `.env` - Environment variables

## Contributing

When adding new scripts:

1. Create PowerShell (`.ps1`) versions only
2. Add proper error handling
3. Include usage documentation
4. Test on Windows
5. Update this README with new script information 