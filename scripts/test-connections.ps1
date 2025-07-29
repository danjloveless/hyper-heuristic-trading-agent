Write-Host "Testing QuantumTrade AI Database Connections..." -ForegroundColor Green

# Test ClickHouse
Write-Host "`nTesting ClickHouse connection..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%20version()" -Method Get
    Write-Host "✅ ClickHouse Version: $response" -ForegroundColor Green
    
    # Test database creation
    $dbTest = Invoke-RestMethod -Uri "http://localhost:8123/?query=CREATE%20DATABASE%20IF%20NOT%20EXISTS%20quantumtrade" -Method Get
    Write-Host "✅ QuantumTrade database verified" -ForegroundColor Green
} catch {
    Write-Host "❌ ClickHouse connection failed: $_" -ForegroundColor Red
}

# Test Redis
Write-Host "`nTesting Redis connection..." -ForegroundColor Yellow
try {
    $redisPing = redis-cli ping 2>$null
    if ($redisPing -eq "PONG") {
        Write-Host "✅ Redis connection successful" -ForegroundColor Green
        
        # Test Redis info
        $redisInfo = redis-cli info server | Select-String "redis_version"
        Write-Host "✅ $redisInfo" -ForegroundColor Green
    } else {
        Write-Host "❌ Redis ping failed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Redis connection failed: $_" -ForegroundColor Red
}

Write-Host "`n🚀 Database connections tested!" -ForegroundColor Cyan