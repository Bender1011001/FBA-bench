"""Minimal baseline agent for FBA-Bench.

This agent interacts with the simulation using a simple, linear script:
- Launches a single product.
- Runs the simulation for a fixed number of days.
- Outputs basic results (sales, profit, ending cash).
"""

from .simulation import Simulation
from fba_bench.config_loader import load_config

# Load configuration
_config = load_config()
DEFAULT_CATEGORY = _config.agent_defaults.default_category
DEFAULT_COST = _config.agent_defaults.default_cost
DEFAULT_PRICE = _config.agent_defaults.default_price
DEFAULT_QTY = _config.agent_defaults.default_qty

class BaselineAgent:
    """
    Minimal baseline agent for FBA-Bench.

    This agent interacts with the simulation using a simple, linear script:
    - Launches a single product.
    - Runs the simulation for a fixed number of days.
    - Outputs basic results (sales, profit, ending cash).

    Attributes:
        sim (Simulation): The simulation instance.
        days (int): Number of days to run the simulation.
        asin (str): Product ASIN.
        category (str): Product category.
        cost (float): Product cost per unit.
        price (float): Product price.
        qty (int): Initial inventory quantity.
    """
    # --- Default Product Configuration ---
    DEFAULT_ASIN = "B000BASE"
    DEFAULT_CATEGORY = DEFAULT_CATEGORY
    DEFAULT_COST = DEFAULT_COST
    DEFAULT_PRICE = DEFAULT_PRICE
    DEFAULT_QTY = DEFAULT_QTY

    def __init__(self, days=30, asin=None, category=None, cost=None, price=None, qty=None, seed_capital=10000):
        """
        Initialize the BaselineAgent.

        Args:
            days (int): Number of days to run the simulation.
            asin (str): Product ASIN.
            category (str): Product category.
            cost (float): Product cost per unit.
            price (float): Product price.
            qty (int): Initial inventory quantity.
            seed_capital (float): Initial seed capital for profit calculation.
        """
        self.seed_capital = seed_capital
        
        self.sim = Simulation()
        self.days = days
        self.asin = asin if asin is not None else self.DEFAULT_ASIN
        self.category = category if category is not None else self.DEFAULT_CATEGORY
        self.cost = cost if cost is not None else self.DEFAULT_COST
        self.price = price if price is not None else self.DEFAULT_PRICE
        self.qty = qty if qty is not None else self.DEFAULT_QTY

    def run(self):
        """
        Launch a single product and run the simulation for the specified number of days.
        """
        # Launch a single product
        self.sim.launch_product(
            asin=self.asin,
            category=self.category,
            cost=self.cost,
            price=self.price,
            qty=self.qty
        )
        # Run the simulation for the specified number of days
        self.sim.run(self.days)

    def results(self):
        """
        Get summary results for the agent's run.

        Returns:
            dict: Dictionary with total sales, revenue, COGS, profit, and ending cash.
        """
        prod = self.sim.products[self.asin]
        total_sales = sum(prod.sales_history)
        ending_cash = self.sim.ledger.balance("Cash")
        cogs = total_sales * self.cost
        revenue = total_sales * self.price

        # Calculate value of unsold inventory
        inventory_batches = self.sim.inventory._batches.get(self.asin, [])
        inventory_value = sum(batch.quantity * batch.cost_per_unit for batch in inventory_batches)

        profit = ending_cash + inventory_value - self.seed_capital  # Use dynamic seed capital
        return {
            "total_sales": total_sales,
            "revenue": revenue,
            "cogs": cogs,
            "profit": profit,
            "ending_cash": ending_cash
        }

if __name__ == "__main__":
    agent = BaselineAgent(days=30)
    agent.run()
    results = agent.results()
    print("Baseline Agent Results:")
    print(f"  Total Sales: {results['total_sales']}")
    print(f"  Revenue: ${results['revenue']:.2f}")
    print(f"  COGS: ${results['cogs']:.2f}")
    print(f"  Profit: ${results['profit']:.2f}")
    print(f"  Ending Cash: ${results['ending_cash']:.2f}")