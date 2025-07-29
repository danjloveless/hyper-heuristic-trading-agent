param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$Logs
)

Write-Host "🐳 QuantumTrade AI Database Management" -ForegroundColor Cyan

if ($Stop) {
    Write-Host "🛑 Stopping databases..." -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml down
    Write-Host "✅ Databases stopped" -ForegroundColor Green
    exit 0
}

if ($Restart) {
    Write-Host "🔄 Restarting databases..." -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml down
    Start-Sleep -Seconds 2
    docker-compose -f docker-compose.dev.yml up -d
    Write-Host "✅ Databases restarted" -ForegroundColor Green
    exit 0
}

if ($Status) {
    Write-Host "📊 Database Status:" -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml ps
    exit 0
}

if ($Logs) {
    Write-Host "📋 Database Logs:" -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml logs -f
    exit 0
}

# Default action: Start databases
Write-Host "🚀 Starting development databases..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Start the databases
Write-Host "📦 Starting ClickHouse and Redis..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start databases" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Databases started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "🔍 Testing connections..." -ForegroundColor Yellow

# Test ClickHouse
try {
    $clickhouseTest = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%201" -Method Get -TimeoutSec 5
    if ($clickhouseTest -eq "1`n") {
        Write-Host "✅ ClickHouse: Ready" -ForegroundColor Green
    }
} catch {
    Write-Host "⏳ ClickHouse: Starting..." -ForegroundColor Yellow
}

# Test Redis
try {
    $redisTest = docker exec hyper-heuristic-agents-redis-1 redis-cli ping 2>$null
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis: Ready" -ForegroundColor Green
    }
} catch {
    Write-Host "⏳ Redis: Starting..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup complete! You can now:" -ForegroundColor Cyan
Write-Host "  1. Test the system: cargo run --example basic_usage" -ForegroundColor White
Write-Host "  2. Build the workspace: cargo build --workspace" -ForegroundColor White
Write-Host ""
Write-Host "📋 Management commands:" -ForegroundColor Gray
Write-Host "  .\scripts\start-databases.ps1 -Status    # Check status" -ForegroundColor Gray
Write-Host "  .\scripts\start-databases.ps1 -Logs      # View logs" -ForegroundColor Gray
Write-Host "  .\scripts\start-databases.ps1 -Restart   # Restart databases" -ForegroundColor Gray
Write-Host "  .\scripts\start-databases.ps1 -Stop      # Stop databases" -ForegroundColor Gray