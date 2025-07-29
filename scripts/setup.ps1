# Create workspace structure
Write-Host "Creating QuantumTrade AI Database Abstraction Layer structure..." -ForegroundColor Green

# Create directory structure
$directories = @(
    "shared-libs/database-abstraction/src/clickhouse",
    "shared-libs/database-abstraction/src/redis", 
    "shared-libs/database-abstraction/migrations/clickhouse",
    "shared-libs/shared-types/src",
    "config",
    "scripts"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force
    Write-Host "Created directory: $dir" -ForegroundColor Blue
}

Write-Host "Directory structure created successfully!" -ForegroundColor Green

# Start development databases
Write-Host "Starting development databases..." -ForegroundColor Green
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test connections
Write-Host "Testing database connections..." -ForegroundColor Green

# Test ClickHouse
try {
    $clickhouseTest = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%201" -Method Get
    if ($clickhouseTest -eq "1`n") {
        Write-Host "✅ ClickHouse connection successful" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ ClickHouse connection failed: $_" -ForegroundColor Red
}

# Test Redis
try {
    $redisTest = redis-cli ping 2>$null
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis connection successful" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Redis connection failed: $_" -ForegroundColor Red
}

Write-Host "Setup completed! You can now:" -ForegroundColor Cyan
Write-Host "1. Build the workspace: cargo build --workspace" -ForegroundColor White
Write-Host "2. Run tests: cargo test --workspace" -ForegroundColor White
Write-Host "3. Run migrations: cargo run --example migrate" -ForegroundColor White