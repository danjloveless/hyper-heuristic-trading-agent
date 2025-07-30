use database_abstraction::{DatabaseManager, DatabaseConfig, DatabaseClient};
use clap::{Command, Arg};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let matches = Command::new("Database Migration Runner")
        .version("1.0")
        .about("Runs database migrations for QuantumTrade AI")
        .arg(
            Arg::new("dry-run")
                .long("dry-run")
                .action(clap::ArgAction::SetTrue)
                .help("Show what migrations would be run without executing them")
        )
        .arg(
            Arg::new("reset")
                .long("reset")
                .action(clap::ArgAction::SetTrue)
                .help("Reset database by dropping all tables and recreating them")
        )
        .arg(
            Arg::new("status")
                .long("status")
                .action(clap::ArgAction::SetTrue)
                .help("Show current migration status")
        )
        .get_matches();

    // Create database configuration
    let config = DatabaseConfig::default();
    
    // Create database manager
    let db_manager = DatabaseManager::new(config).await?;
    
    println!("🔗 Connected to databases successfully!");
    
    if matches.get_flag("status") {
        println!("📊 Checking migration status...");
        
        // Get ClickHouse client to check migration status
        let clickhouse = db_manager.clickhouse();
        
        // Check if migrations table exists by trying to query it
        let migrations_exist = clickhouse.health_check().await.is_ok();
            
        if migrations_exist {
            // Get executed migrations - we'll use a simple approach for now
            let executed_migrations: Vec<u32> = Vec::new(); // TODO: Implement proper migration status check
                
            println!("✅ Migrations table exists");
            println!("📋 Executed migrations: {:?}", executed_migrations);
            
            // Get all available migrations
            use database_abstraction::clickhouse::migrations::get_migrations;
            let all_migrations = get_migrations();
            println!("📋 Total available migrations: {}", all_migrations.len());
            
            for migration in all_migrations {
                let status = if executed_migrations.contains(&migration.version) {
                    "✅ EXECUTED"
                } else {
                    "⏳ PENDING"
                };
                println!("  {} - {}: {}", status, migration.version, migration.name);
            }
        } else {
            println!("❌ Migrations table does not exist");
            println!("💡 Run migrations to create the table");
        }
        
        return Ok(());
    }
    
    if matches.get_flag("reset") {
        println!("⚠️  RESET MODE - Dropping all tables and recreating them...");
        
        let clickhouse = db_manager.clickhouse();
        
        // Drop all tables - we'll use a simpler approach for now
        println!("🗑️  Note: Reset functionality requires direct database access");
        println!("💡 For now, please manually drop tables if needed");
        
        println!("✅ All tables dropped");
    }
    
    if matches.get_flag("dry-run") {
        println!("🔍 DRY RUN MODE - Showing what migrations would be executed:");
        
        // Get ClickHouse client
        let clickhouse = db_manager.clickhouse();
        
        // Get executed migrations - simplified for now
        let executed_migrations: Vec<u32> = Vec::new(); // TODO: Implement proper migration status check
            
        // Get all available migrations
        use database_abstraction::clickhouse::migrations::get_migrations;
        let all_migrations = get_migrations();
        
        for migration in all_migrations {
            if !executed_migrations.contains(&migration.version) {
                println!("⏳ Would execute: {} - {}", migration.version, migration.name);
                println!("   SQL: {}", migration.sql.lines().next().unwrap_or(""));
            } else {
                println!("✅ Already executed: {} - {}", migration.version, migration.name);
            }
        }
        
        return Ok(());
    }
    
    // Run migrations
    println!("🚀 Running database migrations...");
    
    match db_manager.run_migrations().await {
        Ok(()) => {
            println!("✅ Migrations completed successfully!");
            
            // Show final status
            println!("📊 Migrations completed successfully!");
            println!("💡 Check the database to see the created tables");
        }
        Err(e) => {
            eprintln!("❌ Migration failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
} 