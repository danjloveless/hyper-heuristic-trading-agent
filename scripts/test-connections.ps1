# scripts/test-connections.ps1
# Test script to verify ClickHouse and Redis connections

Write-Host "🧪 Testing QuantumTrade AI database connections..." -ForegroundColor Green

# Test ClickHouse HTTP interface
Write-Host "Testing ClickHouse HTTP interface..." -ForegroundColor Yellow
try {
    $pingResponse = Invoke-RestMethod -Uri "http://localhost:8123/ping" -TimeoutSec 5
    Write-Host "✅ ClickHouse ping successful: $pingResponse" -ForegroundColor Green
} catch {
    Write-Host "❌ ClickHouse ping failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test ClickHouse query
Write-Host "Testing ClickHouse query..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8123/" -Method Post -Body "SELECT 'Hello from ClickHouse!' as message, now() as timestamp" -TimeoutSec 10
    Write-Host "✅ ClickHouse query successful: $response" -ForegroundColor Green
} catch {
    Write-Host "❌ ClickHouse query failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test ClickHouse database
Write-Host "Testing QuantumTrade database..." -ForegroundColor Yellow
try {
    $dbResponse = Invoke-RestMethod -Uri "http://localhost:8123/" -Method Post -Body "SHOW DATABASES" -TimeoutSec 10
    Write-Host "✅ Available databases:" -ForegroundColor Green
    Write-Host "$dbResponse" -ForegroundColor White
} catch {
    Write-Host "❌ Database query failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test Redis connection
Write-Host "Testing Redis connection..." -ForegroundColor Yellow
try {
    $redisResponse = docker exec quantumtrade-redis redis-cli ping
    if ($redisResponse -eq "PONG") {
        Write-Host "✅ Redis ping successful: $redisResponse" -ForegroundColor Green
    } else {
        Write-Host "❌ Redis ping failed: Expected PONG, got $redisResponse" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Redis ping failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test Redis basic operations
Write-Host "Testing Redis operations..." -ForegroundColor Yellow
try {
    docker exec quantumtrade-redis redis-cli set test_key "Hello Redis!" | Out-Null
    $redisValue = docker exec quantumtrade-redis redis-cli get test_key
    Write-Host "✅ Redis set/get successful: $redisValue" -ForegroundColor Green
    docker exec quantumtrade-redis redis-cli del test_key | Out-Null
} catch {
    Write-Host "❌ Redis operations failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Connection tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Container status:" -ForegroundColor Cyan
docker-compose -f docker-compose.dev.yml ps