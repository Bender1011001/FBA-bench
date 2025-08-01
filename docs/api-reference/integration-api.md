# Real-World Integration API Reference

This document provides a comprehensive API reference for interacting with FBA-Bench's real-world integration components, primarily `RealWorldAdapter` and `MarketplaceAPI` implementations. These APIs facilitate safe and controlled deployment of agents to live marketplaces.

## 1. `RealWorldAdapter`

The primary interface for submitting agent-generated actions to real-world marketplaces, with built-in safety constraints.

-   **Module**: [`integration/real_world_adapter.py`](integration/real_world_adapter.py)
-   **Class**: `RealWorldAdapter`

### Constructor

`__init__(self, config: dict, marketplace_factory: MarketplaceFactory = None)`

-   **`config`**: (`dict`, required) Configuration for real-world integration, including enabled marketplaces, safety constraints, and API credentials.
-   **`marketplace_factory`**: (`MarketplaceFactory`, optional) An instance of `MarketplaceFactory` to create specific marketplace API clients. If `None`, a default factory is used.

### Key Methods

#### `handle_agent_action(self, agent_action: Event) -> bool`
Processes an agent-proposed action, applies safety constraints, and if valid, translates and dispatches it to the relevant real-world marketplace API.

-   **`agent_action`**: (`Event`, required) An FBA-Bench `Event` object representing an action proposed by an agent (e.g., `PriceUpdateEvent`, `InventoryRestockEvent`).
-   **Returns**: `bool` - `True` if the action was successfully processed and sent (or queued) to the marketplace, `False` otherwise (e.g., if it violated a safety constraint).
-   **Raises**: `SafetyConstraintViolationError`, `MarketplaceAPIError`

#### `set_dry_run_mode(self, enabled: bool)`
Toggles dry-run mode. In dry-run mode, actions are validated but not actually sent to external APIs.

-   **`enabled`**: (`bool`, required) Set to `True` for dry-run, `False` for live.

## 2. `MarketplaceAPI` (Abstract Base Class)

The foundational abstract class for all real-world marketplace API integrations. Specific marketplace clients (e.g., Amazon Seller Central) must inherit from this.

-   **Module**: [`integration/marketplace_apis/marketplace_factory.py`](integration/marketplace_apis/marketplace_factory.py)
-   **Class**: `MarketplaceAPI`

### Abstract Methods (Must be implemented by subclasses)

#### `submit_price_update(self, product_id: str, new_price: float) -> dict`
Submits a price update for a given product to the marketplace.

-   **`product_id`**: (`str`, required) The ID of the product.
-   **`new_price`**: (`float`, required) The new price for the product.
-   **Returns**: `dict` - A dictionary containing the API response.

#### `submit_inventory_update(self, product_id: str, quantity: int) -> dict`
Submits an inventory update for a given product.

-   **`product_id`**: (`str`, required) The ID of the product.
-   **`quantity`**: (`int`, required) The new inventory quantity or quantity to add/remove.
-   **Returns**: `dict` - A dictionary containing the API response.

#### `get_recent_orders(self, lookback_days: int = 7) -> list[dict]`
Retrieves recent order information from the marketplace.

-   **`lookback_days`**: (`int`, optional) Number of days to look back for orders.
-   **Returns**: `list[dict]` - A list of order dictionaries.

### Example Concrete Implementation: `AmazonSellerCentralAPI`

-   **Module**: [`integration/marketplace_apis/amazon_seller_central.py`](integration/marketplace/amazon_seller_central.py)
-   **Class**: `AmazonSellerCentralAPI` (inherits from `MarketplaceAPI`)

This class implements the abstract methods of `MarketplaceAPI` using specific Amazon MWS or Selling Partner API calls.

## 3. `IntegrationValidator`

Provides utility functions to validate the setup and configuration of real-world integration.

-   **Module**: [`integration/integration_validator.py`](integration/integration_validator.py)
-   **Class**: `IntegrationValidator`

### Constructor

`__init__(self, config: dict = None)`

-   **`config`**: (`dict`, optional) The integration configuration to validate against.

### Key Methods

#### `validate_setup(self) -> bool`
Performs pre-flight checks to ensure all required API credentials are available and configured correctly for enabled marketplaces. Does not make live API calls for this check.

-   **Returns**: `bool` - `True` if the setup is valid, `False` otherwise. Logs specific issues.

#### `test_connection(self, marketplace_name: str) -> bool`
Attempts a minimal, safe API call (e.g., getting account info) to validate live connection to a specified marketplace.

-   **`marketplace_name`**: (`str`, required) The name of the marketplace (e.g., "amazon_seller_central").
-   **Returns**: `bool` - `True` if the connection is successful, `False` otherwise.
-   **Raises**: `MarketplaceAPIError` on connection failure.