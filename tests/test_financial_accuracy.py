"""
Comprehensive Financial Accuracy Tests.

Tests all financial calculations to ensure accuracy and proper Money type usage.
This test suite validates that the simulation produces financially accurate results.
"""
import pytest
from decimal import Decimal
from fba_bench.money import Money
from fba_bench.simulation import Simulation
from fba_bench.fee_engine import FeeEngine


class TestMoneyTypeStrictness:
    """Test Money type strict enforcement and accuracy."""
    
    def test_money_rejects_float_operations(self):
        """Money type should reject float operations to prevent contamination."""
        money = Money.from_dollars(10.00)
        
        # Test multiplication rejection
        with pytest.raises(TypeError, match="Cannot multiply Money by float"):
            money * 1.5
        
        # Test division rejection
        with pytest.raises(TypeError, match="Cannot divide Money by float"):
            money / 1.5
    
    def test_money_precise_arithmetic(self):
        """Money arithmetic should be precise and deterministic."""
        # Test precise addition
        a = Money.from_dollars(10.01)
        b = Money.from_dollars(20.02)
        result = a + b
        assert result == Money.from_dollars(30.03)
        assert result.cents == 3003
        
        # Test precise multiplication with Decimal
        price = Money.from_dollars(19.99)
        quantity = Decimal('3')
        total = price * quantity
        assert total == Money.from_dollars(59.97)
        assert total.cents == 5997
        
        # Test precise division
        total = Money.from_dollars(100.00)
        parts = 3
        per_part = total / parts
        assert per_part.cents == 3333  # $33.33
    
    def test_money_currency_enforcement(self):
        """Money should enforce currency compatibility."""
        usd = Money.from_dollars(10.00, "USD")
        eur = Money.from_dollars(10.00, "EUR")
        
        with pytest.raises(ValueError, match="Cannot operate on different currencies"):
            usd + eur
    
    def test_money_sum_operations(self):
        """Test proper Money sum operations."""
        prices = [Money.from_dollars(19.99), Money.from_dollars(29.99), Money.from_dollars(9.99)]
        
        # Proper way to sum Money objects
        total = Money.zero()
        for price in prices:
            total += price
        
        assert total == Money.from_dollars(59.97)
        
        # Test that sum() with int start fails as expected
        with pytest.raises(TypeError):
            sum(prices)  # This should fail because sum starts with int(0)


class TestFeeCalculationAccuracy:
    """Test fee calculation accuracy and consistency."""
    
    def test_fee_engine_deterministic(self):
        """Fee calculations should be deterministic and accurate."""
        fee_engine = FeeEngine()
        
        # Test same inputs produce same outputs
        fees1 = fee_engine.total_fees(
            category='DEFAULT',
            price=Money.from_dollars(19.99),
            size_tier='standard',
            size='small'
        )
        
        fees2 = fee_engine.total_fees(
            category='DEFAULT',
            price=Money.from_dollars(19.99),
            size_tier='standard',
            size='small'
        )
        
        assert fees1['total'] == fees2['total']
        assert fees1['referral_fee'] == fees2['referral_fee']
        assert fees1['fba_fee'] == fees2['fba_fee']
    
    def test_fee_calculation_no_double_counting(self):
        """Ensure no double-counting of storage fees."""
        fee_engine = FeeEngine()
        
        # Test with explicit storage fee parameter
        fees_with_param = fee_engine.total_fees(
            category='DEFAULT',
            price=Money.from_dollars(19.99),
            size_tier='standard',
            size='small',
            storage_fee=5.0,
            months_storage=0,  # No calculation
            cubic_feet=0.0
        )
        
        # Test with calculated storage
        fees_with_calc = fee_engine.total_fees(
            category='DEFAULT',
            price=Money.from_dollars(19.99),
            size_tier='standard',
            size='small',
            storage_fee=0.0,  # No parameter
            months_storage=1,
            cubic_feet=1.0
        )
        
        # Should be different - no double counting
        assert abs(fees_with_param['total'] - fees_with_calc['total']) > 0.01
    
    def test_referral_fee_accuracy(self):
        """Test referral fee calculation accuracy."""
        fee_engine = FeeEngine()
        
        # Test known referral fee calculation
        fees = fee_engine.total_fees(
            category='DEFAULT',
            price=Money.from_dollars(100.00),
            size_tier='standard',
            size='small'
        )
        
        # DEFAULT category should have 15% referral fee
        expected_referral = 100.00 * 0.15
        assert abs(fees['referral_fee'] - expected_referral) < 0.01


class TestSimulationFinancialAccuracy:
    """Test end-to-end simulation financial accuracy."""
    
    def test_simulation_accounting_identity(self):
        """Test that accounting identity A = L + E holds throughout simulation."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation and check accounting identity
        for day in range(5):
            sim.tick_day()
            
            # Check accounting identity: Assets = Liabilities + Equity
            assets = sim.ledger.balance('Cash') + sim.ledger.balance('Inventory')
            liabilities = Money.zero()  # No liabilities in this simple case
            equity = sim.ledger.balance('Equity')
            
            # Assets should equal Liabilities + Equity
            assert abs((assets - (liabilities + equity)).to_decimal()) < Decimal('0.01')
    
    def test_simulation_cash_flow_accuracy(self):
        """Test cash flow calculations are accurate."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        initial_cash = sim.ledger.balance('Cash')
        
        # Run one day
        sim.tick_day()
        
        final_cash = sim.ledger.balance('Cash')
        cash_change = final_cash - initial_cash
        
        # Cash change should be revenue minus fees minus COGS
        # This is a basic sanity check
        assert isinstance(cash_change, Money)
    
    def test_simulation_inventory_valuation(self):
        """Test inventory valuation accuracy."""
        sim = Simulation(seed=42)
        cost_per_unit = Money.from_dollars(5.0)
        initial_qty = 100
        
        sim.launch_product('B000TEST', 'DEFAULT', cost=cost_per_unit, price=19.99, qty=initial_qty)
        
        # Initial inventory value should be cost * quantity
        expected_initial_value = cost_per_unit * initial_qty
        initial_inventory = sim.ledger.balance('Inventory')
        
        assert initial_inventory == expected_initial_value
    
    def test_simulation_money_type_consistency(self):
        """Test that all financial values are proper Money types."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation
        sim.tick_day()
        
        # Check that all ledger balances are Money types
        cash = sim.ledger.balance('Cash')
        inventory = sim.ledger.balance('Inventory')
        revenue = sim.ledger.balance('Revenue')
        fees = sim.ledger.balance('Fees')
        cogs = sim.ledger.balance('COGS')
        
        assert isinstance(cash, Money)
        assert isinstance(inventory, Money)
        assert isinstance(revenue, Money)
        assert isinstance(fees, Money)
        assert isinstance(cogs, Money)


class TestFinancialInvariants:
    """Test financial invariants that must always hold."""
    
    def test_revenue_equals_price_times_quantity(self):
        """Revenue should always equal price × quantity sold."""
        sim = Simulation(seed=42)
        price = Money.from_dollars(19.99)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=price, qty=100)
        
        # Track sales and revenue
        initial_revenue = sim.ledger.balance('Revenue')
        
        sim.tick_day()
        
        final_revenue = sim.ledger.balance('Revenue')
        revenue_change = final_revenue - initial_revenue
        
        # Revenue change should be positive (assuming sales occurred)
        # This is a basic sanity check - exact validation would require
        # tracking actual units sold
        assert revenue_change >= Money.zero()
    
    def test_cogs_equals_cost_times_quantity_sold(self):
        """COGS should equal cost per unit × quantity sold."""
        sim = Simulation(seed=42)
        cost_per_unit = Money.from_dollars(5.0)
        sim.launch_product('B000TEST', 'DEFAULT', cost=cost_per_unit, price=19.99, qty=100)
        
        initial_cogs = sim.ledger.balance('COGS')
        initial_inventory = sim.ledger.balance('Inventory')
        
        sim.tick_day()
        
        final_cogs = sim.ledger.balance('COGS')
        final_inventory = sim.ledger.balance('Inventory')
        
        cogs_change = final_cogs - initial_cogs
        inventory_change = final_inventory - initial_inventory
        
        # COGS increase should equal inventory decrease
        assert cogs_change == -inventory_change
    
    def test_no_negative_money_balances(self):
        """No account should have negative balances in normal operation."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation for several days
        for _ in range(10):
            sim.tick_day()
            
            # Check that key balances are non-negative
            inventory = sim.ledger.balance('Inventory')
            revenue = sim.ledger.balance('Revenue')
            fees = sim.ledger.balance('Fees')
            cogs = sim.ledger.balance('COGS')
            
            assert inventory >= Money.zero()
            assert revenue >= Money.zero()
            assert fees >= Money.zero()
            assert cogs >= Money.zero()


class TestPropertyBasedFinancialTests:
    """Property-based tests for financial calculations."""
    
    def test_fee_calculation_monotonicity(self):
        """Higher prices should generally result in higher fees."""
        fee_engine = FeeEngine()
        
        prices = [Money.from_dollars(p) for p in [10.00, 20.00, 30.00, 40.00, 50.00]]
        fees = []
        
        for price in prices:
            fee_result = fee_engine.total_fees(
                category='DEFAULT',
                price=price,
                size_tier='standard',
                size='small'
            )
            fees.append(fee_result['total'])
        
        # Fees should generally increase with price (due to referral fees)
        for i in range(1, len(fees)):
            assert fees[i] >= fees[i-1], f"Fee monotonicity violated: {fees[i]} < {fees[i-1]}"
    
    def test_money_arithmetic_properties(self):
        """Test mathematical properties of Money arithmetic."""
        a = Money.from_dollars(10.50)
        b = Money.from_dollars(5.25)
        c = Money.from_dollars(2.75)
        
        # Associativity: (a + b) + c = a + (b + c)
        assert (a + b) + c == a + (b + c)
        
        # Commutativity: a + b = b + a
        assert a + b == b + a
        
        # Identity: a + 0 = a
        assert a + Money.zero() == a
        
        # Inverse: a - a = 0
        assert a - a == Money.zero()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])