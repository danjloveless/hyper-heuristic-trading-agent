# Performance benchmarking script (PowerShell)

Write-Host "📊 Running Performance Benchmarks" -ForegroundColor Green

# Build in release mode
Write-Host "🔨 Building in release mode..." -ForegroundColor Yellow
cargo build --release

# Run benchmarks
Write-Host "🏃 Running benchmarks..." -ForegroundColor Yellow
cargo bench

# Performance tests
Write-Host "🧪 Running performance tests..." -ForegroundColor Yellow
cargo test --release test_performance --features bench

# Memory usage test
Write-Host "💾 Running memory efficiency tests..." -ForegroundColor Yellow
cargo test --release test_memory_efficiency

# High volume test
Write-Host "📈 Running high volume tests..." -ForegroundColor Yellow
cargo test --release test_high_volume_batch_processing

Write-Host "✅ Benchmarks complete!" -ForegroundColor Green 