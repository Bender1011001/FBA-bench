"""
Unit tests for DemandService.

Covers:
- BSR update logic with/without competitors
- EMA and history updates
- Edge cases: zero demand, min/max BSR, price/velocity effects
"""

import unittest
from unittest.mock import Mock
from fba_bench.services.demand_service import DemandService
from fba_bench.money import Money

class TestDemandService(unittest.TestCase):
    def setUp(self):
        self.service = DemandService(
            bsr_base=1000,
            bsr_min_value=10,
            bsr_max_value=5000,
            bsr_smoothing_factor=0.1,
            ema_decay=0.2
        )
        self.product = self._make_product(price=Money.from_dollars(20.0))

    def _make_product(self, price):
        p = Mock()
        p.price = price
        p.sales_history = []
        p.demand_history = []
        p.conversion_history = []
        p.ema_sales_velocity = 0
        p.ema_conversion = 0
        p.bsr = None
        return p

    def _make_competitor(self, price, sales_velocity):
        c = Mock()
        c.price = price
        c.sales_velocity = sales_velocity
        return c

    def test_update_bsr_no_competitors(self):
        # First update: should set EMA to current sales/demand, BSR to base
        self.service.update_bsr(self.product, units_sold=10, demand=100, competitors=[])
        self.assertEqual(self.product.sales_history, [10])
        self.assertEqual(self.product.demand_history, [100])
        self.assertAlmostEqual(self.product.conversion_history[0], 0.1)
        self.assertEqual(self.product.ema_sales_velocity, 10)
        self.assertAlmostEqual(self.product.ema_conversion, 0.1)
        self.assertEqual(self.product.bsr, 1000)  # base, since EMA <= smoothing

        # Second update: EMA should update, BSR should recalculate
        self.service.update_bsr(self.product, units_sold=20, demand=200, competitors=[])
        self.assertEqual(self.product.sales_history, [10, 20])
        self.assertEqual(self.product.demand_history, [100, 200])
        self.assertAlmostEqual(self.product.conversion_history[1], 0.1)
        # EMA: (1-0.2)*10 + 0.2*20 = 8 + 4 = 12
        self.assertAlmostEqual(self.product.ema_sales_velocity, 12)
        self.assertAlmostEqual(self.product.ema_conversion, 0.1*0.8 + 0.1*0.2)
        # BSR: still base, since EMA <= smoothing

    def test_update_bsr_with_competitors(self):
        comp1 = self._make_competitor(Money.from_dollars(18.0), 15)
        comp2 = self._make_competitor(Money.from_dollars(22.0), 25)
        competitors = [comp1, comp2]

        # Prime product with some history
        self.service.update_bsr(self.product, units_sold=50, demand=100, competitors=[])
        self.service.update_bsr(self.product, units_sold=60, demand=100, competitors=[])
        # Now update with competitors
        self.service.update_bsr(self.product, units_sold=70, demand=100, competitors=competitors)
        # EMA should be > smoothing, so BSR should be calculated
        self.assertTrue(self.product.ema_sales_velocity > self.service.bsr_smoothing_factor)
        self.assertTrue(self.product.ema_conversion > self.service.bsr_smoothing_factor)
        # BSR should be clamped between min and max
        self.assertGreaterEqual(self.product.bsr, self.service.bsr_min_value)
        self.assertLessEqual(self.product.bsr, self.service.bsr_max_value)

    def test_update_bsr_zero_demand(self):
        # Zero demand should not crash, conversion = 0
        self.service.update_bsr(self.product, units_sold=10, demand=0, competitors=[])
        self.assertEqual(self.product.conversion_history[0], 0.0)
        self.assertEqual(self.product.ema_conversion, 0.0)

    def test_bsr_min_max_clamping(self):
        # Set up product with very high EMA to force BSR below min
        self.product.ema_sales_velocity = 1e6
        self.product.ema_conversion = 1e6
        self.product.price = Money.from_dollars(20.0)
        competitors = [self._make_competitor(Money.from_dollars(20.0), 1)]
        bsr = self.service._calculate_bsr(self.product, competitors)
        self.assertEqual(bsr, self.service.bsr_min_value)

        # Set up product with very low EMA: BSR should be set to bsr_base, not bsr_max_value
        self.product.ema_sales_velocity = 0.00001
        self.product.ema_conversion = 0.00001
        bsr = self.service._calculate_bsr(self.product, competitors)
        self.assertEqual(bsr, self.service.bsr_base)

    def test_get_bsr_metrics(self):
        self.service.update_bsr(self.product, units_sold=5, demand=50, competitors=[])
        metrics = self.service.get_bsr_metrics(self.product)
        self.assertEqual(metrics['current_bsr'], self.product.bsr)
        self.assertEqual(metrics['ema_sales_velocity'], self.product.ema_sales_velocity)
        self.assertEqual(metrics['ema_conversion'], self.product.ema_conversion)
        self.assertEqual(metrics['sales_history_length'], len(self.product.sales_history))
        self.assertEqual(metrics['demand_history_length'], len(self.product.demand_history))
        self.assertEqual(metrics['conversion_history_length'], len(self.product.conversion_history))

if __name__ == '__main__':
    unittest.main()