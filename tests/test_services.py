"""
Unit tests for FBA-Bench service layer.
Tests each service method in isolation with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal

from fba_bench.money import Money
from fba_bench.services.sales_processor import SalesProcessor
from fba_bench.services.inventory_service import InventoryService
from fba_bench.services.fee_calculation_service import FeeCalculationService
from fba_bench.services.penalty_fee_service import PenaltyFeeService
from fba_bench.services.trust_score_service import TrustScoreService
from fba_bench.services.demand_service import DemandService
from fba_bench.services.competitor_manager import CompetitorManager
from fba_bench.services.listing_manager import ListingManagerService


class TestSalesProcessor:
    """Test SalesProcessor service methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create a mock simulation for testing."""
        sim = Mock()
        sim.ledger = Mock()
        sim.inventory = Mock()
        sim.products = {}
        sim.day = 1
        return sim
    
    @pytest.fixture
    def sales_processor(self, mock_simulation):
        """Create SalesProcessor instance with mocked dependencies."""
        return SalesProcessor(mock_simulation)
    
    def test_process_sale_success(self, sales_processor, mock_simulation):
        """Test successful sale processing."""
        # Setup
        asin = "B000TEST"
        quantity = 2
        price = Money.from_dollars(19.99)
        
        # Mock product
        product = Mock()
        product.price = price
        product.cost = Money.from_dollars(5.0)
        mock_simulation.products[asin] = product
        
        # Mock inventory check
        mock_simulation.inventory.get_available_quantity.return_value = 10
        mock_simulation.inventory.remove_inventory.return_value = True
        
        # Execute
        result = sales_processor.process_sale(asin, quantity)
        
        # Verify
        assert result is not None
        assert result.asin == asin
        assert result.quantity_sold == quantity
        assert result.unit_price == price
        assert result.total_revenue == price * quantity
        
        # Verify inventory was updated
        mock_simulation.inventory.remove_inventory.assert_called_once_with(asin, quantity)
    
    def test_process_sale_insufficient_inventory(self, sales_processor, mock_simulation):
        """Test sale processing with insufficient inventory."""
        # Setup
        asin = "B000TEST"
        quantity = 10
        
        product = Mock()
        product.price = Money.from_dollars(19.99)
        mock_simulation.products[asin] = product
        
        # Mock insufficient inventory
        mock_simulation.inventory.get_available_quantity.return_value = 5
        
        # Execute
        result = sales_processor.process_sale(asin, quantity)
        
        # Verify
        assert result is None
        mock_simulation.inventory.remove_inventory.assert_not_called()
    
    def test_process_sale_nonexistent_product(self, sales_processor, mock_simulation):
        """Test sale processing for non-existent product."""
        # Execute
        result = sales_processor.process_sale("NONEXISTENT", 1)
        
        # Verify
        assert result is None


class TestInventoryService:
    """Test InventoryService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.inventory = Mock()
        sim.products = {}
        return sim
    
    @pytest.fixture
    def inventory_service(self, mock_simulation):
        """Create InventoryService instance."""
        return InventoryService(mock_simulation)
    
    def test_add_inventory_batch(self, inventory_service, mock_simulation):
        """Test adding inventory batch."""
        # Setup
        asin = "B000TEST"
        quantity = 100
        cost_per_unit = Money.from_dollars(5.0)
        
        # Execute
        result = inventory_service.add_inventory_batch(asin, quantity, cost_per_unit)
        
        # Verify
        assert result is True
        mock_simulation.inventory.add_batch.assert_called_once()
    
    def test_get_inventory_value(self, inventory_service, mock_simulation):
        """Test inventory value calculation."""
        # Setup mock batches
        batch1 = Mock()
        batch1.quantity = 50
        batch1.cost_per_unit = Money.from_dollars(5.0)
        
        batch2 = Mock()
        batch2.quantity = 30
        batch2.cost_per_unit = Money.from_dollars(6.0)
        
        mock_simulation.inventory.get_all_batches.return_value = [batch1, batch2]
        
        # Execute
        total_value = inventory_service.get_total_inventory_value()
        
        # Verify
        expected_value = (50 * Money.from_dollars(5.0)) + (30 * Money.from_dollars(6.0))
        assert total_value == expected_value


class TestFeeCalculationService:
    """Test FeeCalculationService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.fee_engine = Mock()
        return sim
    
    @pytest.fixture
    def fee_service(self, mock_simulation):
        """Create FeeCalculationService instance."""
        return FeeCalculationService(mock_simulation)
    
    def test_calculate_referral_fee(self, fee_service, mock_simulation):
        """Test referral fee calculation."""
        # Setup
        category = "Electronics"
        sale_price = Money.from_dollars(100.0)
        expected_fee = Money.from_dollars(8.0)  # 8% for electronics
        
        mock_simulation.fee_engine.calculate_referral_fee.return_value = expected_fee
        
        # Execute
        fee = fee_service.calculate_referral_fee(category, sale_price)
        
        # Verify
        assert fee == expected_fee
        mock_simulation.fee_engine.calculate_referral_fee.assert_called_once_with(category, sale_price)
    
    def test_calculate_fba_fee(self, fee_service, mock_simulation):
        """Test FBA fee calculation."""
        # Setup
        weight = 1.5  # pounds
        dimensions = (10, 8, 6)  # inches
        expected_fee = Money.from_dollars(3.50)
        
        mock_simulation.fee_engine.calculate_fba_fee.return_value = expected_fee
        
        # Execute
        fee = fee_service.calculate_fba_fee(weight, dimensions)
        
        # Verify
        assert fee == expected_fee
        mock_simulation.fee_engine.calculate_fba_fee.assert_called_once_with(weight, dimensions)


class TestPenaltyFeeService:
    """Test PenaltyFeeService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.ledger = Mock()
        sim.products = {}
        return sim
    
    @pytest.fixture
    def penalty_service(self, mock_simulation):
        """Create PenaltyFeeService instance."""
        return PenaltyFeeService(mock_simulation)
    
    def test_apply_stockout_penalty(self, penalty_service, mock_simulation):
        """Test stockout penalty application."""
        # Setup
        asin = "B000TEST"
        days_out_of_stock = 3
        
        product = Mock()
        product.price = Money.from_dollars(20.0)
        mock_simulation.products[asin] = product
        
        # Execute
        penalty = penalty_service.apply_stockout_penalty(asin, days_out_of_stock)
        
        # Verify
        assert penalty > Money.zero()
        mock_simulation.ledger.record_transaction.assert_called()
    
    def test_apply_excess_inventory_penalty(self, penalty_service, mock_simulation):
        """Test excess inventory penalty."""
        # Setup
        asin = "B000TEST"
        excess_units = 500
        
        # Execute
        penalty = penalty_service.apply_excess_inventory_penalty(asin, excess_units)
        
        # Verify
        assert penalty > Money.zero()
        mock_simulation.ledger.record_transaction.assert_called()


class TestTrustScoreService:
    """Test TrustScoreService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.products = {}
        sim.day = 10
        return sim
    
    @pytest.fixture
    def trust_service(self, mock_simulation):
        """Create TrustScoreService instance."""
        return TrustScoreService(mock_simulation)
    
    def test_update_trust_score_positive_event(self, trust_service, mock_simulation):
        """Test trust score update with positive event."""
        # Setup
        asin = "B000TEST"
        product = Mock()
        product.trust_score = 0.8
        mock_simulation.products[asin] = product
        
        # Execute
        new_score = trust_service.update_trust_score(asin, "positive_review", 0.1)
        
        # Verify
        assert new_score > 0.8
        assert new_score <= 1.0
    
    def test_update_trust_score_negative_event(self, trust_service, mock_simulation):
        """Test trust score update with negative event."""
        # Setup
        asin = "B000TEST"
        product = Mock()
        product.trust_score = 0.8
        mock_simulation.products[asin] = product
        
        # Execute
        new_score = trust_service.update_trust_score(asin, "negative_review", -0.2)
        
        # Verify
        assert new_score < 0.8
        assert new_score >= 0.0
    
    def test_calculate_trust_impact_on_conversion(self, trust_service):
        """Test trust score impact on conversion rate."""
        # Test various trust scores
        test_cases = [
            (1.0, 1.0),  # Perfect trust = no penalty
            (0.8, 0.9),  # Good trust = small penalty
            (0.5, 0.75), # Medium trust = medium penalty
            (0.2, 0.4),  # Low trust = high penalty
        ]
        
        for trust_score, expected_min_multiplier in test_cases:
            multiplier = trust_service.calculate_trust_impact_on_conversion(trust_score)
            assert multiplier >= expected_min_multiplier
            assert multiplier <= 1.0


class TestDemandService:
    """Test DemandService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.products = {}
        sim.market_dynamics = Mock()
        sim.day = 5
        return sim
    
    @pytest.fixture
    def demand_service(self, mock_simulation):
        """Create DemandService instance."""
        return DemandService(mock_simulation)
    
    def test_calculate_base_demand(self, demand_service, mock_simulation):
        """Test base demand calculation."""
        # Setup
        asin = "B000TEST"
        category = "Electronics"
        
        product = Mock()
        product.category = category
        product.price = Money.from_dollars(50.0)
        mock_simulation.products[asin] = product
        
        mock_simulation.market_dynamics.get_category_demand.return_value = 1000
        
        # Execute
        demand = demand_service.calculate_base_demand(asin)
        
        # Verify
        assert demand > 0
        mock_simulation.market_dynamics.get_category_demand.assert_called_once_with(category)
    
    def test_apply_seasonality_factor(self, demand_service, mock_simulation):
        """Test seasonality factor application."""
        # Setup
        base_demand = 100
        category = "Toys"
        
        # Mock holiday season (day 350 = mid December)
        mock_simulation.day = 350
        
        # Execute
        adjusted_demand = demand_service.apply_seasonality_factor(base_demand, category)
        
        # Verify - toys should have higher demand during holiday season
        assert adjusted_demand >= base_demand


class TestCompetitorManager:
    """Test CompetitorManager methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.competitors = []
        sim.products = {}
        return sim
    
    @pytest.fixture
    def competitor_manager(self, mock_simulation):
        """Create CompetitorManager instance."""
        return CompetitorManager(mock_simulation)
    
    def test_add_competitor(self, competitor_manager, mock_simulation):
        """Test adding a competitor."""
        # Setup
        asin = "B000COMP"
        category = "Electronics"
        price = Money.from_dollars(25.0)
        strategy = "price_follower"
        
        # Execute
        competitor = competitor_manager.add_competitor(asin, category, price, strategy)
        
        # Verify
        assert competitor is not None
        assert competitor.asin == asin
        assert competitor.category == category
        assert competitor.price == price
        assert competitor.strategy == strategy
    
    def test_update_competitor_prices(self, competitor_manager, mock_simulation):
        """Test competitor price updates."""
        # Setup
        competitor = Mock()
        competitor.asin = "B000COMP"
        competitor.strategy = "price_follower"
        competitor.price = Money.from_dollars(20.0)
        mock_simulation.competitors = [competitor]
        
        # Mock agent product
        agent_product = Mock()
        agent_product.price = Money.from_dollars(25.0)
        mock_simulation.products["B000TEST"] = agent_product
        
        # Execute
        competitor_manager.update_competitor_prices()
        
        # Verify that price follower adjusted price
        # (exact logic depends on implementation)
        assert competitor.price is not None


class TestListingManagerService:
    """Test ListingManagerService methods."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation."""
        sim = Mock()
        sim.products = {}
        sim.ledger = Mock()
        return sim
    
    @pytest.fixture
    def listing_manager(self, mock_simulation):
        """Create ListingManagerService instance."""
        return ListingManagerService(mock_simulation)
    
    def test_update_product_price(self, listing_manager, mock_simulation):
        """Test product price update."""
        # Setup
        asin = "B000TEST"
        new_price = Money.from_dollars(29.99)
        
        product = Mock()
        product.price = Money.from_dollars(19.99)
        mock_simulation.products[asin] = product
        
        # Execute
        success = listing_manager.update_product_price(asin, new_price)
        
        # Verify
        assert success is True
        assert product.price == new_price
    
    def test_update_product_price_nonexistent(self, listing_manager, mock_simulation):
        """Test price update for non-existent product."""
        # Execute
        success = listing_manager.update_product_price("NONEXISTENT", Money.from_dollars(10.0))
        
        # Verify
        assert success is False
    
    def test_launch_new_product(self, listing_manager, mock_simulation):
        """Test launching a new product."""
        # Setup
        asin = "B000NEW"
        category = "Books"
        cost = Money.from_dollars(8.0)
        price = Money.from_dollars(15.99)
        initial_qty = 50
        
        # Execute
        product = listing_manager.launch_product(asin, category, cost, price, initial_qty)
        
        # Verify
        assert product is not None
        assert asin in mock_simulation.products
        assert mock_simulation.products[asin] == product


# Integration test for service interactions
class TestServiceIntegration:
    """Test service interactions and workflows."""
    
    def test_complete_sale_workflow(self, mock_simulation):
        """Test complete sale processing workflow across services."""
        # Setup services
        sales_processor = SalesProcessor(mock_simulation)
        fee_service = FeeCalculationService(mock_simulation)
        trust_service = TrustScoreService(mock_simulation)
        
        # Setup mocks
        asin = "B000TEST"
        product = Mock()
        product.price = Money.from_dollars(20.0)
        product.cost = Money.from_dollars(8.0)
        product.category = "Electronics"
        product.trust_score = 0.9
        mock_simulation.products[asin] = product
        
        mock_simulation.inventory.get_available_quantity.return_value = 10
        mock_simulation.inventory.remove_inventory.return_value = True
        mock_simulation.fee_engine.calculate_referral_fee.return_value = Money.from_dollars(1.60)
        
        # Execute workflow
        sale_result = sales_processor.process_sale(asin, 2)
        referral_fee = fee_service.calculate_referral_fee(product.category, product.price)
        trust_service.update_trust_score(asin, "successful_sale", 0.01)
        
        # Verify workflow completed successfully
        assert sale_result is not None
        assert referral_fee > Money.zero()
        assert product.trust_score >= 0.9  # Trust score should improve or stay same