"""
Unit tests for BSRCalculationService.

These tests isolate the BSR calculation logic to verify correctness of:
- EMA (Exponential Moving Average) calculations
- Sales velocity and conversion tracking
- Relative index calculations
- BSR formula application
- Edge cases and boundary conditions
"""

import unittest
from unittest.mock import Mock, MagicMock
from decimal import Decimal

from fba_bench.services.bsr_calculation_service import BSRCalculationService
from fba_bench.money import Money
from fba_bench.models.competitor import Competitor


class TestBSRCalculationService(unittest.TestCase):
    """Test suite for BSRCalculationService."""
    
    def setUp(self):
        """Set up test fixtures with standard configuration."""
        self.config = {
            'ema_decay': 0.1,
            'bsr_smoothing_factor': 1.0,
            'bsr_base': 1000000,
            'bsr_min_value': 1,
            'bsr_max_value': 10000000
        }
        self.service = BSRCalculationService(self.config)
    
    def create_mock_product(self, **kwargs):
        """Create a mock product with default attributes."""
        product = Mock()
        product.price = kwargs.get('price', Money.from_dollars(25.99))
        product.bsr = kwargs.get('bsr', 100000)
        
        # Create mock lists that support append and len operations
        product.sales_history = Mock()
        product.sales_history.append = Mock()
        sales_history_data = kwargs.get('sales_history', [])
        product.sales_history.__len__ = Mock(return_value=len(sales_history_data))
        
        product.demand_history = Mock()
        product.demand_history.append = Mock()
        demand_history_data = kwargs.get('demand_history', [])
        product.demand_history.__len__ = Mock(return_value=len(demand_history_data))
        
        product.conversion_history = Mock()
        product.conversion_history.append = Mock()
        conversion_history_data = kwargs.get('conversion_history', [])
        product.conversion_history.__len__ = Mock(return_value=len(conversion_history_data))
        
        # Set EMA attributes directly, not as mocks, for direct value access
        # Set EMA attributes directly, not as mocks, for direct value access
        # Set EMA attributes directly, not as mocks, for direct value access
        # Set EMA attributes directly, not as mocks, for direct value access
        product.ema_sales_velocity = kwargs.get('ema_sales_velocity', 0.0)
        product.ema_conversion = kwargs.get('ema_conversion', 0.0)
        return product
    
    def create_mock_competitor(self, sales_velocity=10.0, price=25.99):
        """Create a mock competitor with specified attributes."""
        competitor = Mock(spec=Competitor)
        competitor.sales_velocity = sales_velocity
        competitor.price = Money.from_dollars(price)
        return competitor

    def test_initialization_with_config(self):
        """Test service initializes correctly with provided configuration."""
        custom_config = {
            'ema_decay': 0.2,
            'bsr_smoothing_factor': 2.0,
            'bsr_base': 2000000,
            'bsr_min_value': 5,
            'bsr_max_value': 5000000
        }
        service = BSRCalculationService(custom_config)
        
        self.assertEqual(service.ema_decay, 0.2)
        self.assertEqual(service.bsr_smoothing_factor, 2.0)
        self.assertEqual(service.bsr_base, 2000000)
        self.assertEqual(service.bsr_min_value, 5)
        self.assertEqual(service.bsr_max_value, 5000000)
    
    def test_initialization_with_defaults(self):
        """Test service uses default values for missing config parameters."""
        minimal_config = {}
        service = BSRCalculationService(minimal_config)
        
        self.assertEqual(service.ema_decay, 0.1)
        self.assertEqual(service.bsr_smoothing_factor, 1.0)
        self.assertEqual(service.bsr_base, 1000000)
        self.assertEqual(service.bsr_min_value, 1)
        self.assertEqual(service.bsr_max_value, 10000000)

    def test_sales_history_update(self):
        """Test that sales and demand history are properly updated."""
        product = self.create_mock_product()
        units_sold = 15
        demand = 50.0
        
        self.service._update_sales_history(product, units_sold, demand)
        
        product.sales_history.append.assert_called_once_with(15)
        product.demand_history.append.assert_called_once_with(50.0)
        product.conversion_history.append.assert_called_once_with(0.3)  # 15/50
    
    def test_sales_history_update_zero_demand(self):
        """Test conversion calculation when demand is zero."""
        product = self.create_mock_product()
        units_sold = 10
        demand = 0.0
        
        self.service._update_sales_history(product, units_sold, demand)
        
        product.conversion_history.append.assert_called_once_with(0.0)

    def test_ema_initialization_first_data_point(self):
        """Test EMA initialization with first data point."""
        # Create product with empty sales history to trigger first data point logic
        product = self.create_mock_product(sales_history=[])
        product.sales_history.__len__ = Mock(return_value=1)  # After append, length will be 1
        units_sold = 20
        demand = 100.0
        
        self.service._update_ema_metrics(product, units_sold, demand)
        
        self.assertEqual(product.ema_sales_velocity, 20)
        self.assertEqual(product.ema_conversion, 0.2)  # 20/100

    def test_ema_update_subsequent_data_points(self):
        """Test EMA calculation for subsequent data points."""
        product = self.create_mock_product(
            sales_history=[10, 15],  # Non-empty = not first point
            ema_sales_velocity=10.0,
            ema_conversion=0.1
        )
        units_sold = 30
        demand = 100.0
        expected_conversion = 0.3  # 30/100
        
        self.service._update_ema_metrics(product, units_sold, demand)
        
        # EMA formula: (1 - decay) * old_value + decay * new_value
        # decay = 0.1
        expected_ema_sales = (1 - 0.1) * 10.0 + 0.1 * 30  # 9.0 + 3.0 = 12.0
        expected_ema_conversion = (1 - 0.1) * 0.1 + 0.1 * 0.3  # 0.09 + 0.03 = 0.12
        
        self.assertEqual(product.ema_sales_velocity, expected_ema_sales)
        self.assertEqual(product.ema_conversion, expected_ema_conversion)

    def test_bsr_calculation_higher_sales_velocity_lower_bsr(self):
        """Test that higher sales velocity results in lower (better) BSR."""
        # Create two identical products except for sales velocity
        # Use values > smoothing_factor (1.0) to trigger calculation
        product_high_sales = self.create_mock_product(
            ema_sales_velocity=25.0,
            ema_conversion=2.0  # Above smoothing factor
        )
        product_low_sales = self.create_mock_product(
            ema_sales_velocity=5.0,
            ema_conversion=2.0  # Above smoothing factor
        )
        
        competitors = [self.create_mock_competitor(sales_velocity=10.0)]
        
        bsr_high_sales = self.service._calculate_bsr(product_high_sales, competitors)
        bsr_low_sales = self.service._calculate_bsr(product_low_sales, competitors)
        
        self.assertLess(bsr_high_sales, bsr_low_sales)

    def test_bsr_calculation_higher_conversion_lower_bsr(self):
        """Test that higher conversion rate results in lower (better) BSR."""
        # Use values > smoothing_factor (1.0) to trigger calculation
        product_high_conversion = self.create_mock_product(
            ema_sales_velocity=2.0,
            ema_conversion=2.0  # Above smoothing factor
        )
        product_low_conversion = self.create_mock_product(
            ema_sales_velocity=2.0,
            ema_conversion=1.5  # Above smoothing factor but lower
        )
        
        competitors = [self.create_mock_competitor()]
        
        bsr_high_conversion = self.service._calculate_bsr(product_high_conversion, competitors)
        bsr_low_conversion = self.service._calculate_bsr(product_low_conversion, competitors)
        
        self.assertLess(bsr_high_conversion, bsr_low_conversion)

    def test_bsr_calculation_no_competitors(self):
        """Test BSR calculation when no competitors are present."""
        product = self.create_mock_product(
            ema_sales_velocity=15.0,
            ema_conversion=0.15
        )
        
        bsr = self.service._calculate_bsr(product, [])
        
        # Should use smoothing factor for competitor averages
        self.assertIsInstance(bsr, int)
        self.assertGreaterEqual(bsr, self.config['bsr_min_value'])
        self.assertLessEqual(bsr, self.config['bsr_max_value'])

    def test_bsr_calculation_with_competitors(self):
        """Test BSR calculation with multiple competitors."""
        product = self.create_mock_product(
            ema_sales_velocity=20.0,
            ema_conversion=0.2,
            price=Money.from_dollars(25.00)
        )
        
        competitors = [
            self.create_mock_competitor(sales_velocity=10.0, price=30.00),
            self.create_mock_competitor(sales_velocity=15.0, price=20.00),
            self.create_mock_competitor(sales_velocity=5.0, price=35.00)
        ]
        
        bsr = self.service._calculate_bsr(product, competitors)
        
        self.assertIsInstance(bsr, int)
        self.assertGreaterEqual(bsr, self.config['bsr_min_value'])
        self.assertLessEqual(bsr, self.config['bsr_max_value'])

    def test_bsr_calculation_low_performance_returns_base(self):
        """Test that low performance metrics return base BSR."""
        product = self.create_mock_product(
            ema_sales_velocity=0.5,  # Below smoothing factor
            ema_conversion=0.5       # Below smoothing factor
        )
        
        bsr = self.service._calculate_bsr(product, [])
        
        self.assertEqual(bsr, self.config['bsr_base'])

    def test_bsr_bounds_enforcement(self):
        """Test that BSR is properly bounded by min/max values."""
        # Test minimum bound
        product_excellent = self.create_mock_product(
            ema_sales_velocity=1000.0,  # Extremely high
            ema_conversion=0.9          # Extremely high
        )
        
        bsr_min = self.service._calculate_bsr(product_excellent, [])
        self.assertGreaterEqual(bsr_min, self.config['bsr_min_value'])
        
        # Test maximum bound (harder to trigger, but bounds should work)
        product_poor = self.create_mock_product(
            ema_sales_velocity=1.1,  # Just above smoothing factor
            ema_conversion=1.1       # Just above smoothing factor
        )
        
        bsr_max = self.service._calculate_bsr(product_poor, [])
        self.assertLessEqual(bsr_max, self.config['bsr_max_value'])

    def test_relative_price_index_calculation(self):
        """Test relative price index calculation with competitors."""
        product = self.create_mock_product(
            price=Money.from_dollars(20.00),
            ema_sales_velocity=2.0,  # Above smoothing factor
            ema_conversion=2.0       # Above smoothing factor
        )
        
        # Competitors with higher average price should improve our relative position
        expensive_competitors = [
            self.create_mock_competitor(price=30.00),
            self.create_mock_competitor(price=40.00)
        ]
        
        cheap_competitors = [
            self.create_mock_competitor(price=10.00),
            self.create_mock_competitor(price=15.00)
        ]
        
        bsr_vs_expensive = self.service._calculate_bsr(product, expensive_competitors)
        bsr_vs_cheap = self.service._calculate_bsr(product, cheap_competitors)
        
        # Lower price relative to expensive competitors should give better BSR
        self.assertLess(bsr_vs_expensive, bsr_vs_cheap)

    def test_update_product_bsr_integration(self):
        """Test the main update_product_bsr method integrates all components."""
        product = self.create_mock_product(sales_history=[])
        units_sold = 25
        demand = 100.0
        competitors = [self.create_mock_competitor()]
        
        original_bsr = product.bsr
        self.service.update_product_bsr(product, units_sold, demand, competitors)
        
        # Verify history was updated
        product.sales_history.append.assert_called_with(25)
        product.demand_history.append.assert_called_with(100.0)
        product.conversion_history.append.assert_called_with(0.25)
        
        # Verify BSR was updated
        self.assertNotEqual(product.bsr, original_bsr)
        self.assertIsInstance(product.bsr, int)

    def test_get_bsr_metrics(self):
        """Test BSR metrics retrieval."""
        product = self.create_mock_product(
            bsr=50000,
            ema_sales_velocity=15.5,
            ema_conversion=0.12,
            sales_history=[10, 15, 20],
            demand_history=[50, 75, 100],
            conversion_history=[0.2, 0.2, 0.2]
        )
        
        metrics = self.service.get_bsr_metrics(product)
        
        expected_metrics = {
            'current_bsr': 50000,
            'ema_sales_velocity': 15.5,
            'ema_conversion': 0.12,
            'sales_history_length': 3,
            'demand_history_length': 3,
            'conversion_history_length': 3
        }
        
        self.assertEqual(metrics, expected_metrics)

    def test_get_bsr_metrics_missing_attributes(self):
        """Test BSR metrics with missing product attributes."""
        product = Mock()
        product.bsr = 75000
        
        # Don't set ema_sales_velocity and ema_conversion so getattr returns default
        # Ensure these attributes are mocked as lists or have __len__
        product.sales_history = Mock()
        product.sales_history.__len__ = Mock(return_value=0)
        product.demand_history = Mock()
        product.demand_history.__len__ = Mock(return_value=0)
        product.conversion_history = Mock()
        product.conversion_history.__len__ = Mock(return_value=0)
        
        metrics = self.service.get_bsr_metrics(product)
        
        self.assertEqual(metrics['current_bsr'], 75000)
        self.assertEqual(metrics['ema_sales_velocity'], 0)
        self.assertEqual(metrics['ema_conversion'], 0)
        self.assertEqual(metrics['sales_history_length'], 0)
        self.assertEqual(metrics['demand_history_length'], 0)
        self.assertEqual(metrics['conversion_history_length'], 0)

    def test_money_arithmetic_in_price_calculations(self):
        """Test that Money objects are handled correctly in price calculations."""
        product = self.create_mock_product(
            price=Money.from_dollars(25.50),
            ema_sales_velocity=10.0,
            ema_conversion=0.1
        )
        
        competitors = [
            self.create_mock_competitor(price=30.25),
            self.create_mock_competitor(price=20.75)
        ]
        
        # Should not raise any Money arithmetic errors
        bsr = self.service._calculate_bsr(product, competitors)
        self.assertIsInstance(bsr, int)

    def test_edge_case_zero_denominator_protection(self):
        """Test protection against division by zero in BSR calculation."""
        product = self.create_mock_product(
            ema_sales_velocity=0.0,  # Could cause division issues
            ema_conversion=0.0,
            price=Money.zero()       # Could cause division issues
        )
        
        competitors = [self.create_mock_competitor(sales_velocity=0.0, price=0.01)]
        
        # Should handle gracefully without raising exceptions
        bsr = self.service._calculate_bsr(product, competitors)
        self.assertEqual(bsr, self.config['bsr_base'])  # Should return base BSR


if __name__ == '__main__':
    unittest.main()