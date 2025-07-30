use market_data_ingestion::config_provider::CoreConfigurationProvider;
use core_traits::ConfigurationProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug Configuration Loading");
    println!("==============================");
    
    // Test 1: Direct environment variable access
    println!("1. Testing direct environment variable access...");
    match std::env::var("ALPHA_VANTAGE_API_KEY") {
        Ok(api_key) => {
            println!("✅ Direct env var access: {}", if api_key.len() > 8 { 
                format!("{}...{}", &api_key[..4], &api_key[api_key.len()-4..]) 
            } else { 
                "***".to_string() 
            });
        }
        Err(e) => {
            println!("❌ Direct env var access failed: {:?}", e);
        }
    }
    
    // Test 2: Load .env file manually
    println!("\n2. Testing manual .env file loading...");
    match dotenv::from_path("../.env") {
        Ok(_) => {
            println!("✅ .env file loaded successfully");
            
            // Try again after loading .env
            match std::env::var("ALPHA_VANTAGE_API_KEY") {
                Ok(api_key) => {
                    println!("✅ API Key after .env load: {}", if api_key.len() > 8 { 
                        format!("{}...{}", &api_key[..4], &api_key[api_key.len()-4..]) 
                    } else { 
                        "***".to_string() 
                    });
                }
                Err(e) => {
                    println!("❌ API Key still not found: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to load .env file: {:?}", e);
        }
    }
    
    // Test 3: Try to create configuration provider
    println!("\n3. Testing configuration provider creation...");
    match CoreConfigurationProvider::new().await {
        Ok(provider) => {
            println!("✅ Configuration provider created successfully");
            
            // Test getting the API key through the provider
            match provider.get_secret("ALPHA_VANTAGE_API_KEY").await {
                Ok(api_key) => {
                    println!("✅ API Key through provider: {}", if api_key.len() > 8 { 
                        format!("{}...{}", &api_key[..4], &api_key[api_key.len()-4..]) 
                    } else { 
                        "***".to_string() 
                    });
                }
                Err(e) => {
                    println!("❌ Failed to get API key through provider: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to create configuration provider: {:?}", e);
        }
    }
    
    println!("\n✅ Debug test completed!");
    Ok(())
} 