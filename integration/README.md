# Real-World Integration Framework

This module provides seamless integration between FBA-Bench simulations and real marketplace platforms, enabling agents to transition from simulated training to live marketplace operations with safety guarantees and validation mechanisms.

## Overview

The integration framework consists of three core components:

- **[`RealWorldAdapter`](real_world_adapter.py)**: Core abstraction layer for marketplace interactions
- **[`Marketplace APIs`](marketplace_apis/)**: Platform-specific API implementations
- **[`IntegrationValidator`](integration_validator.py)**: Safety and consistency validation system

## Key Features

### 🔌 Universal Marketplace Abstraction
- **Unified Interface**: Single API for multiple marketplace platforms (Amazon, eBay, Shopify, etc.)
- **Mode Switching**: Seamless transitions between simulation, sandbox, and live environments
- **Action Translation**: Automatic conversion between simulation actions and platform-specific API calls
- **State Synchronization**: Real-time sync between simulation state and marketplace reality

### 🛡️ Safety and Validation
- **Pre-execution Validation**: Safety checks before executing actions in live environments
- **Consistency Verification**: Ensure simulation and real-world state alignment
- **Rollback Mechanisms**: Automated rollback of dangerous or failed operations
- **Performance Monitoring**: Track integration performance and detect anomalies

### 🏢 Platform Support
- **Amazon Seller Central**: Complete integration with Amazon's seller APIs
- **Multi-platform Architecture**: Extensible design for additional marketplaces
- **Rate Limiting**: Automatic handling of API rate limits and quotas
- **Error Recovery**: Robust error handling and retry mechanisms

## Quick Start

### Basic Integration Setup

```python
from integration.real_world_adapter import RealWorldAdapter
from integration.marketplace_apis.marketplace_factory import MarketplaceFactory

# Initialize real-world adapter
adapter = RealWorldAdapter(mode="sandbox")  # Start with sandbox mode

# Configure marketplace connection
marketplace_config = {
    "platform": "amazon",
    "credentials": {
        "access_key": "your_access_key",
        "secret_key": "your_secret_key",
        "marketplace_id": "ATVPDKIKX0DER"
    }
}

await adapter.initialize_marketplace_connection(marketplace_config)
```

### Action Execution

```python
# Translate simulation action to real-world action
sim_action = {
    "type": "set_price",
    "product_sku": "FBA-PRODUCT-001",
    "new_price": 29.99
}

real_action = await adapter.translate_simulation_action(sim_action)
print(f"Real action: {real_action}")

# Execute action safely
try:
    result = await adapter.execute_action(real_action)
    print(f"Action executed: {result}")
except Exception as e:
    print(f"Action failed: {e}")
```

### State Synchronization

```python
# Sync simulation state with real marketplace
real_state = await adapter.sync_state_from_real_world()
print(f"Current inventory: {real_state['inventory']}")
print(f"Active listings: {real_state['listings']}")

# Update simulation with real-world changes
simulation_updates = await adapter.get_state_updates_for_simulation()
```

## Integration Modes

### 1. Simulation Mode
Pure simulation environment with no real-world connections.

```python
adapter = RealWorldAdapter(mode="simulation")
# All actions are simulated, no API calls made
```

### 2. Sandbox Mode
Integration with marketplace sandbox/testing environments.

```python
adapter = RealWorldAdapter(mode="sandbox")
# Actions executed against sandbox APIs
# Safe for testing without real consequences
```

### 3. Live Mode
Full integration with production marketplace APIs.

```python
adapter = RealWorldAdapter(mode="live")
# Actions executed against live APIs
# Real money and inventory at stake
```

## Marketplace API Integration

### Amazon Seller Central

```python
from integration.marketplace_apis.amazon_seller_central import AmazonSellerCentralAPI

# Initialize Amazon API client
amazon_api = AmazonSellerCentralAPI(
    access_key="your_access_key",
    secret_key="your_secret_key",
    marketplace_id="ATVPDKIKX0DER"
)

# Product operations
await amazon_api.update_product_price("SKU123", 19.99)
await amazon_api.update_inventory_quantity("SKU123", 100)

# Order management
orders = await amazon_api.get_recent_orders()
await amazon_api.confirm_shipment("ORDER123", "TRACKING456")

# Financial data
financial_events = await amazon_api.get_financial_events()
```

### Adding New Marketplaces

```python
from integration.marketplace_apis.base_marketplace_api import BaseMarketplaceAPI

class CustomMarketplaceAPI(BaseMarketplaceAPI):
    async def update_product_price(self, sku: str, price: float) -> Dict[str, Any]:
        # Implement platform-specific price update
        pass
    
    async def get_inventory_status(self, sku: str) -> Dict[str, Any]:
        # Implement platform-specific inventory check
        pass

# Register with factory
MarketplaceFactory.register_marketplace("custom", CustomMarketplaceAPI)
```

## Validation and Safety

### Integration Validation

```python
from integration.integration_validator import IntegrationValidator

validator = IntegrationValidator(adapter)

# Validate action consistency
sim_action = {"type": "set_price", "value": 25.00}
real_action = {"api_call": "update_price", "price": 25.00}

is_consistent, report = await validator.validate_action_consistency(
    sim_action, real_action
)

if not is_consistent:
    print(f"Action inconsistency detected: {report}")
```

### Safety Constraints

```python
# Test safety constraints
dangerous_actions = [
    {"type": "set_price", "value": 0.01},  # Extremely low price
    {"type": "bulk_delete", "count": 1000}  # Mass deletion
]

safety_results = await validator.test_safety_constraints(dangerous_actions)
for action, result in zip(dangerous_actions, safety_results):
    if not result["safe"]:
        print(f"Unsafe action blocked: {action} - {result['reason']}")
```

### Performance Validation

```python
# Run performance comparison
comparison_result = await validator.compare_simulation_vs_real_performance(
    simulation_results=sim_results,
    real_world_results=real_results,
    tolerance_config=tolerance_config
)

print(f"Performance difference: {comparison_result['performance_delta']}")
print(f"Within tolerance: {comparison_result['within_tolerance']}")
```

## CLI Integration

The integration framework seamlessly integrates with the experiment CLI:

```bash
# Run simulation with real-world integration
python experiment_cli.py run sweep.yaml --real-world-mode sandbox

# Validate integration before going live
python experiment_cli.py analyze results/ --validate-integration

# Test in different modes
python experiment_cli.py run sweep.yaml --real-world-mode simulation  # Safe testing
python experiment_cli.py run sweep.yaml --real-world-mode sandbox     # API testing
python experiment_cli.py run sweep.yaml --real-world-mode live        # Production
```

## Advanced Configuration

### Rate Limiting and Throttling

```python
# Configure API rate limits
adapter.configure_rate_limiting({
    "requests_per_second": 10,
    "burst_capacity": 50,
    "backoff_strategy": "exponential",
    "max_retries": 3
})
```

### Error Handling and Recovery

```python
# Configure error recovery
adapter.configure_error_handling({
    "retry_on_errors": ["RateLimitExceeded", "TemporaryFailure"],
    "max_retry_attempts": 3,
    "retry_delay_base": 1.0,
    "circuit_breaker_threshold": 5
})
```

### State Management

```python
# Configure state synchronization
adapter.configure_state_sync({
    "sync_frequency": 300,  # Sync every 5 minutes
    "sync_on_actions": True,  # Sync after each action
    "conflict_resolution": "real_world_wins",
    "state_validation": True
})
```

## Monitoring and Observability

### Performance Metrics

The integration framework tracks key performance indicators:

- **API Response Times**: Monitor marketplace API latency
- **Success Rates**: Track action execution success rates
- **State Sync Accuracy**: Measure simulation-reality alignment
- **Error Frequencies**: Monitor and categorize integration errors

### Logging and Debugging

```python
import logging

# Enable integration debugging
logging.getLogger("integration").setLevel(logging.DEBUG)

# Access integration logs
adapter.get_integration_logs(
    level="ERROR",
    time_range="last_hour"
)
```

### Health Checks

```python
# Check integration health
health_status = await adapter.get_health_status()
print(f"Overall health: {health_status['status']}")
print(f"API connectivity: {health_status['api_connectivity']}")
print(f"State sync status: {health_status['state_sync']}")
```

## Security and Compliance

### Credential Management

```python
# Secure credential storage
from integration.security import CredentialManager

cred_manager = CredentialManager()
cred_manager.store_credentials("amazon", {
    "access_key": "encrypted_access_key",
    "secret_key": "encrypted_secret_key"
})

# Use credentials securely
credentials = cred_manager.get_credentials("amazon")
```

### Data Privacy

```python
# Configure data handling
adapter.configure_privacy_settings({
    "log_sensitive_data": False,
    "encrypt_state_data": True,
    "data_retention_days": 30,
    "anonymize_customer_data": True
})
```

### Compliance Checks

```python
# Ensure compliance with marketplace policies
compliance_checker = adapter.get_compliance_checker()
compliance_status = await compliance_checker.validate_operations()

if not compliance_status["compliant"]:
    print(f"Compliance issues: {compliance_status['violations']}")
```

## Best Practices

### 1. Progressive Integration
- Start with simulation mode for development
- Move to sandbox mode for integration testing
- Graduate to live mode only after thorough validation

### 2. Safety First
- Always enable safety constraints in live environments
- Implement comprehensive rollback mechanisms
- Monitor for unexpected behavior patterns
- Maintain human oversight for critical operations

### 3. Performance Optimization
- Implement efficient caching strategies
- Use batch operations where possible
- Monitor and optimize API usage patterns
- Implement smart retry mechanisms

### 4. Error Handling
- Design for failure - expect API errors
- Implement graceful degradation
- Log all errors with sufficient context
- Provide clear error messages to users

### 5. Testing Strategy
- Test all integration paths thoroughly
- Use sandbox environments extensively
- Implement integration tests with real APIs
- Validate state consistency regularly

## Troubleshooting

### Common Issues

**API Connection Failures**
- Verify API credentials and permissions
- Check network connectivity and firewall settings
- Validate API endpoint URLs and versions
- Review rate limiting and quota status

**State Synchronization Problems**
- Check sync frequency configuration
- Verify state mapping logic
- Review conflict resolution settings
- Monitor for data format inconsistencies

**Performance Issues**
- Optimize API call patterns
- Implement proper caching strategies
- Review rate limiting settings
- Monitor network latency

**Safety Constraint Violations**
- Review safety rule configurations
- Check action validation logic
- Verify constraint threshold settings
- Monitor for edge cases

## Extension Points

The integration framework provides several extension points:

### Custom Marketplace APIs
Implement the [`BaseMarketplaceAPI`](marketplace_apis/base_marketplace_api.py) interface to add support for new marketplaces.

### Custom Validators
Extend the [`IntegrationValidator`](integration_validator.py) class to add domain-specific validation rules.

### Custom Safety Constraints
Implement custom safety constraint handlers for specific business requirements.

### Custom State Mappers
Create specialized state mapping logic for complex marketplace data structures.

## Research Applications

The integration framework enables various research directions:

- **Sim-to-Real Transfer**: Study how simulated strategies perform in real environments
- **Real-World Validation**: Validate simulation accuracy against real marketplace data
- **Live Learning**: Enable agents to learn from real marketplace interactions
- **Risk Assessment**: Evaluate the safety of learned strategies in live environments
- **Market Impact Analysis**: Study the impact of AI agents on real marketplace dynamics

For more examples and detailed implementation guides, see the [`examples/`](../examples/) directory.