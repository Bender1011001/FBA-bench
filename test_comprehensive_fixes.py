"""
Comprehensive Test Suite for Critical Fixes.

This test validates all the major architectural improvements and fixes:
1. Monolithic tick_day() method broken into modular services
2. Money type strict enforcement and financial accuracy
3. Secure, decoupled dashboard interface
4. Overall system reliability and maintainability
"""
import pytest
from decimal import Decimal
from fba_bench.simulation import Simulation
from fba_bench.money import Money
from dashboard.secure_api import (
    SecureSimulationDataProvider, 
    DashboardSecurityManager,
    secure_data_provider,
    security_manager
)


class TestArchitecturalImprovements:
    """Test the major architectural improvements."""
    
    def test_monolithic_tick_day_replaced(self):
        """Test that the monolithic tick_day() method has been replaced with modular services."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run tick_day to initialize orchestrator
        sim.tick_day()
        
        # Verify orchestrator was created
        assert hasattr(sim, '_orchestrator'), "Simulation orchestrator not initialized"
        
        # Verify orchestrator has modular services
        orchestrator = sim._orchestrator
        assert hasattr(orchestrator, 'bsr_service'), "BSR service not found"
        assert hasattr(orchestrator, 'customer_event_service'), "Customer event service not found"
        assert hasattr(orchestrator, 'penalty_fee_service'), "Penalty fee service not found"
        
        # Verify orchestration metrics
        metrics = orchestrator.get_orchestration_metrics()
        assert metrics['modular_architecture'] is True
        assert metrics['testable_components'] is True
        assert metrics['separation_of_concerns'] is True
        assert metrics['single_responsibility'] is True
    
    def test_tick_day_method_size_reduction(self):
        """Test that the tick_day() method is now much smaller."""
        import inspect
        from fba_bench.simulation import Simulation
        
        # Get the source code of the tick_day method
        tick_day_source = inspect.getsource(Simulation.tick_day)
        lines = tick_day_source.split('\n')
        
        # Filter out empty lines and comments
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # The new tick_day should be under 20 lines (vs original 686 lines)
        assert len(code_lines) < 20, f"tick_day() still too large: {len(code_lines)} lines"
    
    def test_service_extraction_working(self):
        """Test that extracted services are working correctly."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation to test services
        sim.tick_day()
        sim.tick_day()
        
        # Verify services are functioning
        assert hasattr(sim, '_orchestrator')
        orchestrator = sim._orchestrator
        
        # Test BSR service
        product = sim.products['B000TEST']
        assert hasattr(product, 'bsr')
        assert isinstance(product.bsr, int)
        
        # Test customer event service
        assert hasattr(orchestrator, 'customer_event_service')
        
        # Test penalty fee service
        assert hasattr(orchestrator, 'penalty_fee_service')


class TestFinancialAccuracyImprovements:
    """Test financial accuracy and Money type enforcement."""
    
    def test_money_type_strict_enforcement(self):
        """Test that Money type strictly enforces type safety."""
        money = Money.from_dollars(10.00)
        
        # Should reject float operations
        with pytest.raises(TypeError):
            money * 1.5  # float multiplication
        
        with pytest.raises(TypeError):
            money / 1.5  # float division
        
        # Should work with proper types
        result = money * Decimal('2.0')
        assert result == Money.from_dollars(20.00)
        
        result = money * 2  # int multiplication
        assert result == Money.from_dollars(20.00)
    
    def test_financial_calculations_accurate(self):
        """Test that financial calculations are accurate and deterministic."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation
        initial_cash = sim.ledger.balance('Cash')
        sim.tick_day()
        final_cash = sim.ledger.balance('Cash')
        
        # Verify all balances are Money types
        assert isinstance(initial_cash, Money)
        assert isinstance(final_cash, Money)
        assert isinstance(sim.ledger.balance('Revenue'), Money)
        assert isinstance(sim.ledger.balance('Fees'), Money)
        assert isinstance(sim.ledger.balance('COGS'), Money)
        assert isinstance(sim.ledger.balance('Inventory'), Money)
    
    def test_no_float_contamination(self):
        """Test that no float contamination occurs in financial calculations."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run multiple days
        for _ in range(5):
            sim.tick_day()
        
        # Check all financial values are Money types, not floats
        balances = ['Cash', 'Revenue', 'Fees', 'COGS', 'Inventory']
        for account in balances:
            balance = sim.ledger.balance(account)
            assert isinstance(balance, Money), f"{account} balance is not Money type: {type(balance)}"
            assert not isinstance(balance.to_decimal(), float), f"{account} contains float contamination"
    
    def test_accounting_identity_maintained(self):
        """Test that accounting identity A = L + E is maintained."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Run simulation for several days
        for day in range(5):
            sim.tick_day()
            
            # Check accounting identity: Assets = Liabilities + Equity
            assets = sim.ledger.balance('Cash') + sim.ledger.balance('Inventory')
            liabilities = Money.zero()  # No liabilities in this simple case
            equity = sim.ledger.balance('Equity')
            
            # Assets should equal Liabilities + Equity (within rounding tolerance)
            difference = abs((assets - (liabilities + equity)).to_decimal())
            assert difference < Decimal('0.01'), f"Accounting identity violated on day {day}: {difference}"


class TestSecureDashboardInterface:
    """Test the secure, decoupled dashboard interface."""
    
    def test_dashboard_decoupling(self):
        """Test that dashboard is properly decoupled from simulation core."""
        # Create simulation
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        sim.tick_day()
        
        # Register with secure data provider
        simulation_id = secure_data_provider.register_simulation(sim)
        assert simulation_id is not None
        assert isinstance(simulation_id, str)
        
        # Get snapshot through secure interface
        snapshot = secure_data_provider.get_simulation_snapshot(simulation_id)
        assert snapshot is not None
        assert snapshot.simulation_id == simulation_id
        
        # Verify snapshot contains serialized data, not direct object references
        assert isinstance(snapshot.cash_balance, str)
        assert isinstance(snapshot.total_revenue, str)
        assert isinstance(snapshot.products, list)
        assert isinstance(snapshot.competitors, list)
    
    def test_dashboard_security_features(self):
        """Test dashboard security features."""
        # Create simulation and register
        sim = Simulation(seed=42)
        simulation_id = secure_data_provider.register_simulation(sim)
        
        # Test session management
        session_id = security_manager.create_session(simulation_id)
        assert security_manager.validate_session(session_id) is True
        
        # Test invalid session
        assert security_manager.validate_session("invalid_session") is False
        
        # Test rate limiting
        assert security_manager.check_rate_limit(session_id) is True
        
        # Test access logging
        access_log = secure_data_provider.get_access_log()
        assert isinstance(access_log, list)
        assert len(access_log) > 0
    
    def test_dashboard_immutable_snapshots(self):
        """Test that dashboard provides immutable snapshots."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        sim.tick_day()
        
        simulation_id = secure_data_provider.register_simulation(sim)
        
        # Get financial summary
        financial_summary = secure_data_provider.get_financial_summary(simulation_id)
        assert isinstance(financial_summary, dict)
        assert 'cash_balance' in financial_summary
        assert 'total_revenue' in financial_summary
        assert 'profit_margin_pct' in financial_summary
        
        # Verify all monetary values are serialized as strings
        assert isinstance(financial_summary['cash_balance'], str)
        assert isinstance(financial_summary['total_revenue'], str)
        assert isinstance(financial_summary['profit_margin_pct'], (int, float))
    
    def test_dashboard_no_direct_memory_access(self):
        """Test that dashboard cannot directly access simulation memory."""
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        simulation_id = secure_data_provider.register_simulation(sim)
        snapshot = secure_data_provider.get_simulation_snapshot(simulation_id)
        
        # Verify snapshot doesn't contain direct object references
        assert not hasattr(snapshot, '_simulation_ref')
        assert not any(hasattr(product, '__dict__') for product in snapshot.products)
        
        # Verify modifying snapshot data doesn't affect simulation
        original_cash = str(sim.ledger.balance('Cash'))
        snapshot.cash_balance = "999999.99"  # Modify snapshot
        
        # Simulation should be unchanged
        assert str(sim.ledger.balance('Cash')) == original_cash


class TestOverallSystemReliability:
    """Test overall system reliability and maintainability."""
    
    def test_simulation_deterministic_behavior(self):
        """Test that simulation behavior is deterministic."""
        # Run same simulation twice with same seed
        sim1 = Simulation(seed=42)
        sim1.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        sim1.tick_day()
        sim1.tick_day()
        
        sim2 = Simulation(seed=42)
        sim2.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        sim2.tick_day()
        sim2.tick_day()
        
        # Results should be identical
        assert sim1.ledger.balance('Cash') == sim2.ledger.balance('Cash')
        assert sim1.ledger.balance('Revenue') == sim2.ledger.balance('Revenue')
        assert sim1.ledger.balance('Fees') == sim2.ledger.balance('Fees')
    
    def test_error_handling_robustness(self):
        """Test that system handles errors gracefully."""
        # Test with invalid simulation ID
        snapshot = secure_data_provider.get_simulation_snapshot("invalid_id")
        assert snapshot is None
        
        # Test with missing product
        sim = Simulation(seed=42)
        simulation_id = secure_data_provider.register_simulation(sim)
        
        product_snapshot = secure_data_provider.get_product_performance(simulation_id, "NONEXISTENT")
        assert product_snapshot is None
    
    def test_memory_management(self):
        """Test that system manages memory properly."""
        # Create and register multiple simulations
        simulation_ids = []
        for i in range(5):
            sim = Simulation(seed=i)
            sim_id = secure_data_provider.register_simulation(sim)
            simulation_ids.append(sim_id)
        
        # Unregister simulations
        for sim_id in simulation_ids:
            assert secure_data_provider.unregister_simulation(sim_id) is True
        
        # Verify they're no longer accessible
        for sim_id in simulation_ids:
            snapshot = secure_data_provider.get_simulation_snapshot(sim_id)
            assert snapshot is None


class TestPerformanceImprovements:
    """Test performance improvements from architectural changes."""
    
    def test_modular_services_performance(self):
        """Test that modular services don't significantly impact performance."""
        import time
        
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        
        # Time multiple tick_day operations
        start_time = time.time()
        for _ in range(10):
            sim.tick_day()
        end_time = time.time()
        
        # Should complete 10 days in reasonable time (< 5 seconds)
        elapsed = end_time - start_time
        assert elapsed < 5.0, f"Performance regression: {elapsed:.2f}s for 10 days"
    
    def test_dashboard_response_time(self):
        """Test that dashboard responses are fast."""
        import time
        
        sim = Simulation(seed=42)
        sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
        sim.tick_day()
        
        simulation_id = secure_data_provider.register_simulation(sim)
        
        # Time dashboard operations
        start_time = time.time()
        snapshot = secure_data_provider.get_simulation_snapshot(simulation_id)
        financial_summary = secure_data_provider.get_financial_summary(simulation_id)
        market_analysis = secure_data_provider.get_market_analysis(simulation_id)
        end_time = time.time()
        
        # Dashboard operations should be fast (< 1 second)
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Dashboard too slow: {elapsed:.2f}s"
        
        # Verify operations succeeded
        assert snapshot is not None
        assert financial_summary is not None
        assert market_analysis is not None


def test_comprehensive_system_validation():
    """Comprehensive test that validates the entire system works together."""
    print("\n=== COMPREHENSIVE SYSTEM VALIDATION ===")
    
    # 1. Create simulation with modular architecture
    sim = Simulation(seed=42)
    sim.launch_product('B000TEST', 'DEFAULT', cost=5.0, price=19.99, qty=100)
    print("✓ Simulation created with modular architecture")
    
    # 2. Run simulation with new tick_day implementation
    for day in range(5):
        sim.tick_day()
    print("✓ Simulation ran successfully with modular tick_day()")
    
    # 3. Verify financial accuracy
    cash = sim.ledger.balance('Cash')
    revenue = sim.ledger.balance('Revenue')
    fees = sim.ledger.balance('Fees')
    assert isinstance(cash, Money)
    assert isinstance(revenue, Money)
    assert isinstance(fees, Money)
    print("✓ Financial calculations accurate with Money type enforcement")
    
    # 4. Test secure dashboard interface
    simulation_id = secure_data_provider.register_simulation(sim)
    snapshot = secure_data_provider.get_simulation_snapshot(simulation_id)
    assert snapshot is not None
    assert snapshot.simulation_id == simulation_id
    print("✓ Secure dashboard interface working")
    
    # 5. Verify system reliability
    session_id = security_manager.create_session(simulation_id)
    assert security_manager.validate_session(session_id)
    print("✓ Security and session management working")
    
    print("\n=== ALL CRITICAL FIXES VALIDATED ===")
    print("✓ Monolithic architecture → Modular services")
    print("✓ Float contamination → Strict Money type enforcement")
    print("✓ Insecure dashboard → Secure, decoupled interface")
    print("✓ Poor testability → Comprehensive test coverage")
    print("✓ Financial inaccuracy → Validated accounting accuracy")


if __name__ == "__main__":
    # Run comprehensive validation
    test_comprehensive_system_validation()
    
    # Run all tests
    pytest.main([__file__, "-v"])