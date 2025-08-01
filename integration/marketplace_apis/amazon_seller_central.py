import logging
from typing import Dict, Any, List, Optional
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmazonSellerCentralAPI:
    """
    A mock Amazon Seller Central API wrapper.
    This class simulates interactions with various Seller Central functionalities
    like inventory, pricing, orders, and reports.
    """
    
    def __init__(self, api_key: str = "mock_api_key", secret_key: str = "mock_secret_key", sandbox_mode: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.sandbox_mode = sandbox_mode
        self.base_url = "https://api.amazon.com/sellercentral" if not sandbox_mode else "https://sandbox.api.amazon.com/sellercentral"
        logging.info(f"Initialized AmazonSellerCentralAPI in {'sandbox' if sandbox_mode else 'live'} mode. Base URL: {self.base_url}")
        
        # Internal mock data
        self._mock_inventory: Dict[str, int] = {"FBA-SKU-123": 100, "FBA-SKU-456": 50}
        self._mock_prices: Dict[str, float] = {"FBA-SKU-123": 25.00, "FBA-SKU-456": 15.00}
        self._mock_orders: List[Dict[str, Any]] = []
        self._mock_campaigns: Dict[str, Any] = {}
        self._mock_revenue = 0.0

    async def _mock_api_call(self, endpoint: str, data: Dict[str, Any] = None, delay: float = 0.1) -> Dict[str, Any]:
        """Simulates an asynchronous API call with a delay."""
        await asyncio.sleep(delay)
        logging.debug(f"Mock API call to {endpoint} with data {data}")
        return {"status": "success", "endpoint": endpoint, "data_received": data, "mode": "sandbox" if self.sandbox_mode else "live"}

    async def get_inventory_summary(self, sku: Optional[str] = None) -> Dict[str, int]:
        """
        Simulates retrieving inventory information.
        :param sku: Optional SKU to get specific inventory, returns all if None.
        :return: Dictionary of SKUs to their quantities.
        """
        logging.info(f"Retrieving inventory summary for SKU: {sku if sku else 'all'}")
        await self._mock_api_call("/inventory/summary")
        if sku:
            return {sku: self._mock_inventory.get(sku, 0)}
        return self._mock_inventory.copy()

    async def update_inventory(self, sku: str, quantity: int) -> Dict[str, Any]:
        """
        Simulates updating inventory for a given SKU.
        :param sku: The product SKU.
        :param quantity: The new quantity for the SKU.
        """
        logging.info(f"Updating inventory for {sku} to {quantity}")
        if quantity < 0:
            logging.error(f"Cannot set negative inventory for {sku}. Quantity: {quantity}")
            return {"status": "error", "message": "Quantity cannot be negative."}
        
        self._mock_inventory[sku] = quantity
        return await self._mock_api_call(f"/inventory/{sku}", {"quantity": quantity})

    async def get_product_price(self, sku: str) -> float:
        """
        Simulates retrieving the price of a product.
        :param sku: The product SKU.
        :return: The current price of the product.
        """
        logging.info(f"Retrieving price for {sku}")
        await self._mock_api_call(f"/pricing/{sku}")
        return self._mock_prices.get(sku, 0.0)

    async def update_product_price(self, sku: str, new_price: float) -> Dict[str, Any]:
        """
        Simulates updating the price of a product.
        :param sku: The product SKU.
        :param new_price: The new price for the product.
        """
        logging.info(f"Updating price for {sku} to {new_price}")
        if new_price <= 0:
            logging.error(f"Cannot set non-positive price for {sku}. Price: {new_price}")
            return {"status": "error", "message": "Price must be positive."}

        self._mock_prices[sku] = new_price
        return await self._mock_api_call(f"/pricing/{sku}", {"new_price": new_price})

    async def create_marketing_campaign(self, campaign_name: str, budget: float, duration_days: int) -> Dict[str, Any]:
        """
        Simulates creating a marketing campaign.
        :param campaign_name: Name of the campaign.
        :param budget: Total budget for the campaign.
        :param duration_days: Duration of the campaign in days.
        """
        logging.info(f"Creating marketing campaign: {campaign_name} with budget {budget}")
        if budget <= 0:
            logging.error("Campaign budget must be positive.")
            return {"status": "error", "message": "Campaign budget must be positive."}
        
        campaign_id = f"CAMP-{len(self._mock_campaigns) + 1}"
        self._mock_campaigns[campaign_id] = {
            "name": campaign_name,
            "budget": budget,
            "duration": duration_days,
            "status": "active"
        }
        return await self._mock_api_call("/campaigns", {"campaign_id": campaign_id, "name": campaign_name})

    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Simulates retrieving order information.
        :param status: Filter orders by status (e.g., "pending", "shipped").
        :return: List of order dictionaries.
        """
        logging.info(f"Retrieving orders with status: {status if status else 'all'}")
        await self._mock_api_call("/orders")
        if status:
            return [order for order in self._mock_orders if order.get("status") == status]
        return self._mock_orders.copy()

    async def process_order(self, order_id: str) -> Dict[str, Any]:
        """
        Simulates processing an order (e.g., marking as shipped).
        :param order_id: The ID of the order to process.
        """
        logging.info(f"Processing order: {order_id}")
        for order in self._mock_orders:
            if order["order_id"] == order_id:
                order["status"] = "shipped"
                return await self._mock_api_call(f"/orders/{order_id}/process", {"status": "shipped"})
        return {"status": "error", "message": f"Order {order_id} not found."}

    async def get_performance_report(self, report_type: str = "daily_sales") -> Dict[str, Any]:
        """
        Simulates retrieving various performance reports.
        :param report_type: Type of report requested (e.g., "daily_sales", "revenue_analytics").
        :return: Dictionary containing report data.
        """
        logging.info(f"Retrieving performance report: {report_type}")
        await self._mock_api_call(f"/reports/{report_type}")
        if report_type == "daily_sales":
            return {"date": "2025-01-01", "sales_volume": 120, "net_revenue": 2500.00}
        elif report_type == "revenue_analytics":
            return {"total_revenue": self._mock_revenue, "revenue_last_7_days": self._mock_revenue * 0.2}
        return {"status": "error", "message": f"Report type {report_type} not supported."}

    async def send_customer_message(self, order_id: str, message: str) -> Dict[str, Any]:
        """
        Simulates sending a message to a customer regarding an order.
        :param order_id: The associated order ID.
        :param message: The message content.
        """
        logging.info(f"Sending message for order {order_id}: {message[:50]}...")
        return await self._mock_api_call(f"/orders/{order_id}/message", {"message": message})

    def _simulate_new_order(self, sku: str, quantity: int, price: float):
        """Internal helper to simulate a new order coming in."""
        order_id = f"ORDER-{len(self._mock_orders) + 1}"
        total_price = quantity * price
        self._mock_orders.append({
            "order_id": order_id,
            "sku": sku,
            "quantity": quantity,
            "price_per_unit": price,
            "total_price": total_price,
            "status": "pending",
            "customer_id": "CUST-XYZ"
        })
        self._mock_inventory[sku] = max(0, self._mock_inventory.get(sku, 0) - quantity)
        self._mock_revenue += total_price
        logging.info(f"Simulated new order: {order_id} for {quantity} of {sku}")

# Example usage (for testing purposes outside the main framework flow)
async def _main():
    api = AmazonSellerCentralAPI(sandbox_mode=True)

    # Inventory management
    inventory = await api.get_inventory_summary()
    print(f"Initial Inventory: {inventory}")
    await api.update_inventory("FBA-SKU-123", 90)
    inventory = await api.get_inventory_summary("FBA-SKU-123")
    print(f"Updated Inventory for FBA-SKU-123: {inventory}")

    # Pricing updates
    price = await api.get_product_price("FBA-SKU-123")
    print(f"Initial Price for FBA-SKU-123: {price}")
    await api.update_product_price("FBA-SKU-123", 26.50)
    price = await api.get_product_price("FBA-SKU-123")
    print(f"Updated Price for FBA-SKU-123: {price}")

    # Marketing campaigns
    campaign_result = await api.create_marketing_campaign("Summer Sale", 500.00, 7)
    print(f"Campaign creation result: {campaign_result}")

    # Order processing
    api._simulate_new_order("FBA-SKU-123", 5, 26.50) # Simulate an incoming order
    orders = await api.get_orders()
    print(f"Current Orders: {orders}")
    if orders:
        process_result = await api.process_order(orders[0]["order_id"])
        print(f"Order processing result: {process_result}")
        orders_after_process = await api.get_orders()
        print(f"Orders after processing: {orders_after_process}")

    # Performance analytics
    sales_report = await api.get_performance_report("daily_sales")
    print(f"Daily Sales Report: {sales_report}")
    revenue_report = await api.get_performance_report("revenue_analytics")
    print(f"Revenue Analytics Report: {revenue_report}")

    # Customer communication
    if orders:
        await api.send_customer_message(orders[0]["order_id"], "Your order has been shipped!")

if __name__ == "__main__":
    asyncio.run(_main())