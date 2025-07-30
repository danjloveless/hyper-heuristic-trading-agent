param(
    [switch]$Status,
    [switch]$Reset,
    [switch]$DryRun
)

Write-Host "🔄 QuantumTrade AI Database Migration Runner" -ForegroundColor Cyan

# Check if databases are running
Write-Host "🔍 Checking database status..." -ForegroundColor Yellow
try {
    $clickhouseTest = Invoke-RestMethod -Uri "http://localhost:8123/?query=SELECT%201" -Method Get -TimeoutSec 5
    if ($clickhouseTest -ne "1`n") {
        Write-Host "❌ ClickHouse is not responding correctly" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ ClickHouse: Ready" -ForegroundColor Green
} catch {
    Write-Host "❌ ClickHouse is not running. Please start databases first with: .\scripts\start-databases.ps1" -ForegroundColor Red
    exit 1
}

# Build the migration runner
Write-Host "🔨 Building migration runner..." -ForegroundColor Yellow
cargo build --package database-abstraction --example run_migrations

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build migration runner" -ForegroundColor Red
    exit 1
}

# Run migrations
Write-Host "🚀 Running database migrations..." -ForegroundColor Green

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
    cargo run --package database-abstraction --example run_migrations -- --dry-run
} elseif ($Reset) {
    Write-Host "⚠️  RESET MODE - This will drop and recreate all tables!" -ForegroundColor Red
    $confirmation = Read-Host "Are you sure you want to reset the database? (y/N)"
    if ($confirmation -eq "y" -or $confirmation -eq "Y") {
        cargo run --package database-abstraction --example run_migrations -- --reset
    } else {
        Write-Host "❌ Migration reset cancelled" -ForegroundColor Yellow
        exit 0
    }
} elseif ($Status) {
    Write-Host "📊 Checking migration status..." -ForegroundColor Yellow
    cargo run --package database-abstraction --example run_migrations -- --status
} else {
    cargo run --package database-abstraction --example run_migrations
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Migrations completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Migration failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Database setup complete!" -ForegroundColor Cyan
Write-Host "   You can now run the application with: cargo run --example basic_usage" -ForegroundColor White 