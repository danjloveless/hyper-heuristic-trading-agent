# Setup script for Market Data Ingestion Service (PowerShell)

param(
    [switch]$SkipTests,
    [switch]$SkipBuild
)

Write-Host "🚀 Setting up QuantumTrade AI Market Data Ingestion Service" -ForegroundColor Green

# Check for required tools
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

Write-Host "📋 Checking dependencies..." -ForegroundColor Yellow

$requiredTools = @("cargo", "docker", "docker-compose")
foreach ($tool in $requiredTools) {
    if (-not (Test-Command $tool)) {
        Write-Host "❌ $tool is not installed. Please install it first." -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path config, logs, data | Out-Null

# Copy example configuration if it doesn't exist
if (-not (Test-Path "config/development.toml")) {
    Write-Host "📝 Creating default configuration..." -ForegroundColor Yellow
    if (Test-Path "config/development.toml.example") {
        Copy-Item "config/development.toml.example" "config/development.toml"
    }
}

# Copy environment file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "📝 Creating environment file..." -ForegroundColor Yellow
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-Host "⚠️  Please edit .env file and set your ALPHA_VANTAGE_API_KEY" -ForegroundColor Yellow
    }
}

# Build the project
if (-not $SkipBuild){
    Write-Host "🔨 Building the project..." -ForegroundColor Yellow
    cargo build --release
}

# Start databases
Write-Host "🐳 Starting databases..." -ForegroundColor Yellow
docker-compose up -d clickhouse redis

# Wait for databases to be ready
Write-Host "⏳ Waiting for databases to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check database health
Write-Host "🏥 Checking database health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8123/?query=SELECT%201" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ ClickHouse is ready" -ForegroundColor Green
    } else {
        Write-Host "❌ ClickHouse is not ready" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ ClickHouse is not ready" -ForegroundColor Red
}

try {
    $redisContainer = docker-compose ps -q redis
    $result = docker exec $redisContainer redis-cli ping
    if ($result -eq "PONG") {
        Write-Host "✅ Redis is ready" -ForegroundColor Green
    } else {
        Write-Host "❌ Redis is not ready" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Redis is not ready" -ForegroundColor Red
}

# Run tests
if (-not $SkipTests) {
    Write-Host "🧪 Running tests..." -ForegroundColor Yellow
    cargo test
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file and set your ALPHA_VANTAGE_API_KEY" -ForegroundColor White
Write-Host "2. Review config/development.toml configuration" -ForegroundColor White
Write-Host "3. Run the service: cargo run --bin market-data-service" -ForegroundColor White
Write-Host "4. Check health: curl http://localhost:8080/health" -ForegroundColor White
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor White 