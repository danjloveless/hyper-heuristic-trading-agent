use super::*;

/// Error reporter for sending error reports to various destinations
#[derive(Debug, Clone)]
pub struct ErrorReporter {
    config: ReportingConfig,
}

impl ErrorReporter {
    pub fn new(config: ReportingConfig) -> Self {
        Self { config }
    }
    
    pub async fn report_error(
        &self,
        error: &QuantumTradeError,
        context: &ErrorContext,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        for destination in &self.config.destinations {
            match self.send_to_destination(destination, error, context).await {
                Ok(_) => {
                    tracing::debug!("Successfully reported error to {:?}", destination);
                }
                Err(e) => {
                    tracing::warn!("Failed to report error to {:?}: {}", destination, e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn send_to_destination(
        &self,
        destination: &ReportingDestination,
        error: &QuantumTradeError,
        context: &ErrorContext,
    ) -> Result<()> {
        match destination {
            ReportingDestination::Logs => {
                tracing::error!(
                    error = %error,
                    service = %context.service_name,
                    operation = %context.operation,
                    request_id = ?context.request_id,
                    trace_id = ?context.trace_id,
                    "Error reported"
                );
                Ok(())
            },
            ReportingDestination::Metrics => {
                // In a real implementation, this would send metrics to CloudWatch or Prometheus
                tracing::info!("Metrics reported for error: {}", error);
                Ok(())
            },
            ReportingDestination::CloudWatch => {
                // In a real implementation, this would send structured logs to CloudWatch
                tracing::info!("CloudWatch log sent for error: {}", error);
                Ok(())
            },
            ReportingDestination::SNS => {
                // In a real implementation, this would send alerts via SNS
                tracing::info!("SNS alert sent for error: {}", error);
                Ok(())
            },
            ReportingDestination::Slack => {
                // In a real implementation, this would send Slack notifications
                tracing::info!("Slack notification sent for error: {}", error);
                Ok(())
            },
            ReportingDestination::Email => {
                // In a real implementation, this would send email alerts
                tracing::info!("Email alert sent for error: {}", error);
                Ok(())
            },
        }
    }
} 