Write-Host "=== Fixing QuantumTrade AI Services ===" -ForegroundColor Green

Write-Host "1. Stopping all services..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml down

Write-Host "2. Removing volumes to start fresh..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml down -v

Write-Host "3. Starting services with updated configuration..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml up -d

Write-Host "4. Waiting 45 seconds for initialization..." -ForegroundColor Yellow
Start-Sleep 45

Write-Host "5. Checking service status:" -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml ps

Write-Host "6. Testing connections..." -ForegroundColor Yellow

# Test ClickHouse
Write-Host "Testing ClickHouse..." -ForegroundColor Cyan
for ($i = 1; $i -le 3; $i++) {
    try {
        $result = Invoke-RestMethod -Uri "http://localhost:8123/ping" -TimeoutSec 5
        Write-Host "✅ ClickHouse: $result" -ForegroundColor Green
        break
    } catch {
        Write-Host "⏳ ClickHouse attempt $i failed, retrying..." -ForegroundColor Yellow
        Start-Sleep 5
    }
}

# Test Redis
Write-Host "Testing Redis..." -ForegroundColor Cyan
try {
    $redisResult = docker exec quantumtrade-redis redis-cli ping
    Write-Host "✅ Redis: $redisResult" -ForegroundColor Green
} catch {
    Write-Host "❌ Redis: Failed" -ForegroundColor Red
}

Write-Host "`n=== Services Fixed! ===" -ForegroundColor Green
Write-Host "ClickHouse: http://localhost:8123" -ForegroundColor White
Write-Host "Redis: localhost:6379" -ForegroundColor White