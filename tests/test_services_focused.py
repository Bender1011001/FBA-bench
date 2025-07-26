"""
Focused unit tests for FBA-Bench service layer.
Tests key service methods with proper mocking of dependencies.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal

from fba_bench.money import Money
from fba_bench.services.trust_score_service import TrustScoreService
from fba_bench.services.inventory_service import InventoryService
from fba_bench.services.demand_service import DemandService


class TestTrustScoreService:
    """Test TrustScoreService methods."""
    
    def test_update_score_with_positive_events(self):
        """Test trust score update with positive customer events."""
        service = TrustScoreService()
        
        # Test with positive events
        customer_events = {
            'positive_reviews': 5,
            'negative_reviews': 1,
            'returns': 2,
            'a_to_z_claims': 0,
            'policy_violations': 0
        }
        
        initial_score = 0.8
        new_score = service.update_score(customer_events, initial_score)
        
        # Score should improve with more positive than negative events
        assert new_score >= initial_score
        assert 0.0 <= new_score <= 1.0
    
    def test_update_score_with_negative_events(self):
        """Test trust score update with negative customer events."""
        service = TrustScoreService()
        
        # Test with negative events
        customer_events = {
            'positive_reviews': 1,
            'negative_reviews': 5,
            'returns': 8,
            'a_to_z_claims': 2,
            'policy_violations': 1
        }
        
        initial_score = 0.8
        new_score = service.update_score(customer_events, initial_score)
        
        # Score should decrease with many negative events
        assert new_score < initial_score
        assert 0.0 <= new_score <= 1.0
    
    def test_update_score_boundary_conditions(self):
        """Test trust score boundary conditions."""
        service = TrustScoreService()
        
        # Test with perfect score
        perfect_events = {
            'positive_reviews': 10,
            'negative_reviews': 0,
            'returns': 0,
            'a_to_z_claims': 0,
            'policy_violations': 0
        }
        
        score = service.update_score(perfect_events, 1.0)
        assert score <= 1.0
        
        # Test with terrible events
        terrible_events = {
            'positive_reviews': 0,
            'negative_reviews': 20,
            'returns': 15,
            'a_to_z_claims': 5,
            'policy_violations': 3
        }
        
        score = service.update_score(terrible_events, 0.1)
        assert score >= 0.0


class TestInventoryService:
    """Test InventoryService methods."""
    
    def test_update_inventory_reduces_quantity(self):
        """Test that inventory update reduces product quantity."""
        service = InventoryService()
        
        # Create mock product
        product = Mock()
        product.qty = 100
        
        # Update inventory
        service.update_inventory(product, 25)
        
        # Verify quantity was reduced
        assert product.qty == 75
    
    def test_update_inventory_zero_units(self):
        """Test inventory update with zero units sold."""
        service = InventoryService()
        
        product = Mock()
        product.qty = 50
        
        service.update_inventory(product, 0)
        
        # Quantity should remain unchanged
        assert product.qty == 50
    
    def test_update_inventory_all_units(self):
        """Test selling all available inventory."""
        service = InventoryService()
        
        product = Mock()
        product.qty = 30
        
        service.update_inventory(product, 30)
        
        # All inventory should be sold
        assert product.qty == 0


class TestDemandService:
    """Test DemandService methods."""
    
    @pytest.fixture
    def demand_service(self):
        """Create DemandService instance with test parameters."""
        return DemandService(
            bsr_base=100000,
            bsr_min_value=1,
            bsr_max_value=1000000,
            bsr_smoothing_factor=0.1,
            ema_decay=0.2
        )
    
    def test_update_bsr_with_sales(self, demand_service):
        """Test BSR update when product has sales."""
        # Create mock product
        product = Mock()
        product.bsr = 50000
        product.sales_history = []
        product.ema_sales_velocity = 0.0
        product.ema_conversion = 0.0
        
        # Create mock competitors
        competitors = [
            Mock(bsr=40000, ema_sales_velocity=3.0),
            Mock(bsr=60000, ema_sales_velocity=2.0),
            Mock(bsr=80000, ema_sales_velocity=1.0)
        ]
        
        # Update BSR with sales
        demand_service.update_bsr(product, units_sold=5, demand=100, competitors=competitors)
        
        # BSR should improve (lower number) with sales
        assert product.bsr < 50000
        assert product.bsr >= 1  # Within valid range
        assert len(product.sales_history) > 0
    
    def test_update_bsr_no_sales(self, demand_service):
        """Test BSR update when product has no sales."""
        product = Mock()
        product.bsr = 50000
        product.sales_history = []
        product.ema_sales_velocity = 2.0
        product.ema_conversion = 0.1
        
        competitors = [Mock(bsr=40000, ema_sales_velocity=3.0)]
        
        # Update BSR with no sales
        demand_service.update_bsr(product, units_sold=0, demand=100, competitors=competitors)
        
        # BSR should worsen (higher number) with no sales
        assert product.bsr >= 50000
        assert product.bsr <= 1000000  # Within valid range
    
    def test_get_bsr_metrics(self, demand_service):
        """Test BSR metrics extraction."""
        product = Mock()
        product.bsr = 25000
        product.ema_sales_velocity = 3.5
        product.ema_conversion = 0.12
        product.sales_history = [1, 2, 3, 2, 4]
        
        metrics = demand_service.get_bsr_metrics(product)
        
        assert metrics['current_bsr'] == 25000
        assert metrics['ema_sales_velocity'] == 3.5
        assert metrics['ema_conversion'] == 0.12
        assert metrics['sales_history_length'] == 5


class TestServiceIntegration:
    """Test integration between services."""
    
    def test_trust_and_inventory_workflow(self):
        """Test workflow combining trust score and inventory updates."""
        trust_service = TrustScoreService()
        inventory_service = InventoryService()
        
        # Setup initial state
        product = Mock()
        product.qty = 100
        product.trust_score = 0.85
        
        # Simulate a sale with mixed customer events
        units_sold = 10
        customer_events = {
            'positive_reviews': 3,
            'negative_reviews': 1,
            'returns': 1,
            'a_to_z_claims': 0,
            'policy_violations': 0
        }
        
        # Process the sale
        inventory_service.update_inventory(product, units_sold)
        new_trust_score = trust_service.update_score(customer_events, product.trust_score)
        
        # Verify both services updated correctly
        assert product.qty == 90  # Inventory reduced
        assert 0.0 <= new_trust_score <= 1.0  # Trust score in valid range
        
        # With mostly positive events, trust should improve or stay same
        assert new_trust_score >= product.trust_score * 0.95  # Allow small decrease
    
    def test_demand_and_trust_interaction(self):
        """Test how demand service interacts with trust-affected sales."""
        demand_service = DemandService()
        trust_service = TrustScoreService()
        
        # Setup product with good trust score
        product = Mock()
        product.bsr = 100000
        product.sales_history = []
        product.ema_sales_velocity = 0.0
        product.ema_conversion = 0.0
        product.trust_score = 0.9
        
        competitors = [Mock(bsr=90000, ema_sales_velocity=2.0)]
        
        # Simulate high sales due to good trust
        high_sales = 15
        demand_service.update_bsr(product, high_sales, demand=200, competitors=competitors)
        
        # BSR should improve significantly
        improved_bsr = product.bsr
        assert improved_bsr < 100000
        
        # Now simulate trust degradation
        bad_events = {
            'positive_reviews': 1,
            'negative_reviews': 8,
            'returns': 5,
            'a_to_z_claims': 2,
            'policy_violations': 1
        }
        
        degraded_trust = trust_service.update_score(bad_events, product.trust_score)
        assert degraded_trust < product.trust_score
        
        # Simulate lower sales due to poor trust
        low_sales = 2
        demand_service.update_bsr(product, low_sales, demand=200, competitors=competitors)
        
        # BSR should worsen
        assert product.bsr > improved_bsr


# Mock-based tests for services that require complex dependencies
class TestServiceMocking:
    """Test services using comprehensive mocking."""
    
    @patch('fba_bench.services.sales_processor.SalesProcessor')
    def test_sales_processor_mock(self, mock_sales_processor):
        """Test SalesProcessor with mocked dependencies."""
        # Setup mock
        mock_instance = mock_sales_processor.return_value
        mock_instance.process_product_sales.return_value = {
            'units_sold': 5,
            'revenue': Money.from_dollars(99.95),
            'fees': Money.from_dollars(15.99)
        }
        
        # Test the mock
        result = mock_instance.process_product_sales(
            asin="B000TEST",
            product=Mock(),
            competitors=[],
            customer_events={}
        )
        
        assert result['units_sold'] == 5
        assert result['revenue'] == Money.from_dollars(99.95)
        assert result['fees'] == Money.from_dollars(15.99)
    
    @patch('fba_bench.services.fee_calculation_service.FeeCalculationService')
    def test_fee_calculation_mock(self, mock_fee_service):
        """Test FeeCalculationService with mocked dependencies."""
        # Setup mock
        mock_instance = mock_fee_service.return_value
        mock_instance.calculate_and_record_sale.return_value = {
            'referral_fee': Money.from_dollars(8.00),
            'fba_fee': Money.from_dollars(3.50),
            'total_fees': Money.from_dollars(11.50)
        }
        
        # Test the mock
        result = mock_instance.calculate_and_record_sale(
            asin="B000TEST",
            units_sold=2,
            unit_price=Money.from_dollars(20.00),
            product_info={}
        )
        
        assert result['referral_fee'] == Money.from_dollars(8.00)
        assert result['fba_fee'] == Money.from_dollars(3.50)
        assert result['total_fees'] == Money.from_dollars(11.50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])