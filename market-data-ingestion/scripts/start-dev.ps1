# Development startup script (PowerShell)

Write-Host "🚀 Starting QuantumTrade AI Market Data Ingestion Service (Development)" -ForegroundColor Green

# Load environment variables
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

# Check for API key
if (-not $env:ALPHA_VANTAGE_API_KEY) {
    Write-Host "❌ ALPHA_VANTAGE_API_KEY not set. Please set it in .env file." -ForegroundColor Red
    exit 1
}

# Start databases if not running
Write-Host "🐳 Starting databases..." -ForegroundColor Yellow
docker-compose up -d clickhouse redis

# Wait for databases
Write-Host "⏳ Waiting for databases..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the service in development mode
Write-Host "🚀 Starting service..." -ForegroundColor Yellow
$env:RUST_LOG = "market_data_ingestion=debug,service=info"
cargo run --bin market-data-service -- --config config/development.toml --log-level debug 