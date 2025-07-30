use market_data_ingestion::config_provider::CoreConfigurationProvider;
use core_traits::ConfigurationProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Configuration Access");
    println!("===============================");
    
    // Create the configuration provider
    let config_provider = CoreConfigurationProvider::new().await?;
    
    // Test accessing the API key
    println!("Testing API key access...");
    match config_provider.get_secret("ALPHA_VANTAGE_API_KEY").await {
        Ok(api_key) => {
            println!("✅ API Key found: {}", if api_key.len() > 8 { 
                format!("{}...{}", &api_key[..4], &api_key[api_key.len()-4..]) 
            } else { 
                "***".to_string() 
            });
        }
        Err(e) => {
            println!("❌ Failed to get API key: {:?}", e);
        }
    }
    
    // Test accessing other environment variables
    println!("\nTesting other environment variables...");
    let test_keys = ["CLICKHOUSE_URL", "REDIS_URL", "DATABASE_NAME"];
    
    for key in &test_keys {
        match config_provider.get_string(key).await {
            Ok(value) => {
                println!("✅ {}: {}", key, value);
            }
            Err(e) => {
                println!("❌ {}: {:?}", key, e);
            }
        }
    }
    
    println!("\n✅ Configuration access test completed!");
    Ok(())
} 