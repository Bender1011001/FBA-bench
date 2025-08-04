import pytest
import uuid
from datetime import datetime
from money import Money

from baseline_bots.greedy_script_bot import GreedyScriptBot, SimulationState
from models.product import Product # Import the actual Product model
from events import SetPriceCommand # Import the actual SetPriceCommand

@pytest.fixture
def sample_product_state_basic():
    return Product(
        asin="B0TEST00A",
        category="electronics",
        cost=Money.from_dollars(10.0),
        price=Money.from_dollars(20.0),
        base_demand=100.0,
        inventory_units=20,
        metadata={
            "competitor_prices": [("B0COMP01", Money.from_dollars(19.0)), ("B0COMP02", Money.from_dollars(21.0))]
        }
    )

@pytest.fixture
def sample_product_state_low_inventory():
    return Product(
        asin="B0TEST00B",
        category="books",
        cost=Money.from_dollars(5.0),
        price=Money.from_dollars(12.0),
        base_demand=50.0,
        inventory_units=5, # Low inventory
        metadata={
            "competitor_prices": [("B0COMP03", Money.from_dollars(11.5))]
        }
    )

@pytest.fixture
def sample_product_state_no_competitors():
    return Product(
        asin="B0TEST00C",
        category="software",
        cost=Money.from_dollars(50.0),
        price=Money.from_dollars(70.0),
        base_demand=20.0,
        inventory_units=30,
        metadata={
            "competitor_prices": [] # No competitors
        }
    )

@pytest.fixture
def sample_simulation_state(sample_product_state_basic, sample_product_state_low_inventory, sample_product_state_no_competitors):
    return SimulationState(
        products=[sample_product_state_basic, sample_product_state_low_inventory, sample_product_state_no_competitors],
        current_tick=1,
        simulation_time=datetime.utcnow()
    )

class TestGreedyScriptBot:

    def test_initialization(self):
        bot = GreedyScriptBot(reorder_threshold=5, reorder_quantity=10)
        assert bot.agent_id == "GreedyScriptBot"
        assert bot.reorder_threshold == 5
        assert bot.reorder_quantity == 10

    def test_decide_price_matching(self, sample_simulation_state):
        bot = GreedyScriptBot()
        actions = bot.decide(sample_simulation_state)

        # Check actions for basic product
        set_price_actions = [a for a in actions if isinstance(a, SetPriceCommand) and a.asin == "B0TEST00A"]
        assert len(set_price_actions) == 1
        action = set_price_actions[0]
        # Lowest competitor price is $19.0. 1% below is $19.0 * 0.99 = $18.81
        assert action.new_price.to_float() == pytest.approx(19.0 * 0.99, abs=0.001)
        assert action.agent_id == "GreedyScriptBot"
        assert action.reason == "Price matching lowest competitor (1% below)"

    def test_decide_price_below_cost_prevention(self, sample_simulation_state):
        # Create a product where 1% below lowest competitor would be below cost
        product_low_cost = Product(
            asin="B0TEST00D",
            category="toys",
            cost=Money.from_dollars(18.0), # Cost is 18
            price=Money.from_dollars(25.0),
            base_demand=10.0,
            inventory_units=10,
            metadata={
                "competitor_prices": [("B0COMP04", Money.from_dollars(19.0))] # Lowest comp price is 19.0
            }
        )
        sim_state = SimulationState(
            products=[product_low_cost],
            current_tick=1,
            simulation_time=datetime.utcnow()
        )
        bot = GreedyScriptBot()
        actions = bot.decide(sim_state)

        set_price_actions = [a for a in actions if isinstance(a, SetPriceCommand) and a.asin == "B0TEST00D"]
        assert len(set_price_actions) == 1
        action = set_price_actions[0]
        # Expected new price: 19.0 * 0.99 = 18.81. This is > cost (18.0). So it should set 18.81
        # The logic was `if new_price < product.cost: new_price = product.cost * 1.05`
        # So if 18.81 < 18.0 is FALSE, it should NOT apply the +5% markup.
        # This means the current code will set it to 18.81 if price is already above cost.
        assert action.new_price.to_float() == pytest.approx(19.0 * 0.99, abs=0.001)

        # Let's re-run with a case where new_price is definitely below cost
        product_very_low_cost = Product(
            asin="B0TEST00E",
            category="toys",
            cost=Money.from_dollars(20.0), # Cost is 20
            price=Money.from_dollars(25.0),
            base_demand=10.0,
            inventory_units=10,
            metadata={
                "competitor_prices": [("B0COMP05", Money.from_dollars(19.0))] # Lowest comp is 19.0
            }
        )
        sim_state_very_low = SimulationState(
            products=[product_very_low_cost],
            current_tick=1,
            simulation_time=datetime.utcnow()
        )
        actions_very_low = bot.decide(sim_state_very_low)
        set_price_actions_very_low = [a for a in actions_very_low if isinstance(a, SetPriceCommand) and a.asin == "B0TEST00E"]
        assert len(set_price_actions_very_low) == 1
        action_very_low = set_price_actions_very_low[0]
        # Expected new price: 19.0 * 0.99 = 18.81. This is < cost (20.0). So it should set cost * 1.05 = 20.0 * 1.05 = 21.0
        assert action_very_low.new_price.to_float() == pytest.approx(20.0 * 1.05, abs=0.001)


    def test_decide_no_competitors(self, sample_simulation_state):
        bot = GreedyScriptBot()
        actions = bot.decide(sample_simulation_state)

        # No SetPriceCommand should be generated for the product without competitors
        set_price_actions_no_comp = [a for a in actions if isinstance(a, SetPriceCommand) and a.asin == "B0TEST00C"]
        assert len(set_price_actions_no_comp) == 0

    def test_decide_inventory_management_logging(self, capsys, sample_simulation_state):
        bot = GreedyScriptBot(reorder_threshold=10, reorder_quantity=50)
        
        # Test low inventory product (B0TEST00B has 5 units, threshold is 10)
        bot.decide(sample_simulation_state)
        captured = capsys.readouterr()
        
        l_asin = sample_simulation_state.products[1].asin # B0TEST00B
        assert f"[GreedyScriptBot] Product {l_asin} inventory low (5), reordering 50 units." in captured.out

        # Test sufficient inventory product (B0TEST00A has 20 units, threshold is 10)
        # Should not trigger inventory message
        product_sufficient_inventory = Product(
            asin="B0TEST00A",
            category="electronics",
            cost=Money.from_dollars(10.0),
            price=Money.from_dollars(20.0),
            base_demand=100.0,
            inventory_units=20, # Sufficient inventory
            metadata={
                "competitor_prices": [("B0COMP01", Money.from_dollars(19.0))]
            }
        )
        sim_state_sufficient = SimulationState(
            products=[product_sufficient_inventory],
            current_tick=1,
            simulation_time=datetime.utcnow()
        )
        bot.decide(sim_state_sufficient)
        captured_sufficient = capsys.readouterr()
        assert "inventory low" not in captured_sufficient.out

    def test_decide_price_insignificant_change(self, sample_product_state_basic):
        # Set current price such that new price is very close
        sample_product_state_basic.price = Money.from_dollars(18.815) # Very close to 18.81
        sim_state = SimulationState(
            products=[sample_product_state_basic],
            current_tick=1,
            simulation_time=datetime.utcnow()
        )
        bot = GreedyScriptBot()
        actions = bot.decide(sim_state)
        
        # Original new price would be 18.81. If current is 18.815, diff is 0.005.
        # 0.005 / 18.815 is approx 0.00026, which is less than 0.001 threshold.
        # So no price change action should be generated.
        set_price_actions = [a for a in actions if isinstance(a, SetPriceCommand) and a.asin == "B0TEST00A"]
        assert len(set_price_actions) == 0

    def test_decide_empty_products(self):
        bot = GreedyScriptBot()
        empty_state = SimulationState(
            products=[],
            current_tick=1,
            simulation_time=datetime.utcnow()
        )
        actions = bot.decide(empty_state)
        assert len(actions) == 0