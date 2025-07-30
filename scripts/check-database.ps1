Write-Host "🔍 QuantumTrade AI Database Check" -ForegroundColor Cyan

# Check if databases are running
Write-Host "🔍 Checking database status..." -ForegroundColor Yellow
try {
    $clickhouseTest = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%201" -Method Get -TimeoutSec 5
    if ($clickhouseTest -eq "1`n") {
        Write-Host "✅ ClickHouse: Ready" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ ClickHouse is not running. Please start databases first with: .\scripts\start-databases.ps1" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📊 Checking database tables..." -ForegroundColor Yellow

# Check tables using direct HTTP query
try {
    $tables = Invoke-RestMethod -Uri "http://localhost:8123/?query=SHOW%20TABLES%20FROM%20quantumtrade" -Method Get -TimeoutSec 10
    Write-Host "✅ Tables found:" -ForegroundColor Green
    $tables -split "`n" | Where-Object { $_ -ne "" } | ForEach-Object {
        Write-Host "  📋 $_" -ForegroundColor White
    }
} catch {
    Write-Host "❌ Failed to retrieve tables: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "📊 Checking migrations table..." -ForegroundColor Yellow

# Check migrations table
try {
    $migrations = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%20*%20FROM%20__migrations%20ORDER%20BY%20version" -Method Get -TimeoutSec 10
    if ($migrations -and $migrations.Trim() -ne "") {
        Write-Host "✅ Migrations table contains:" -ForegroundColor Green
        $migrations -split "`n" | Where-Object { $_ -ne "" } | ForEach-Object {
            Write-Host "  📋 $_" -ForegroundColor White
        }
    } else {
        Write-Host "⚠️  Migrations table is empty" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Failed to retrieve migrations: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 Database check complete!" -ForegroundColor Cyan 