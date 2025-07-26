"""
Unit tests for the secure dashboard API.
Tests the security layer and data provider functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from fba_bench.money import Money
from dashboard.secure_api import (
    SecureSimulationDataProvider, 
    DashboardSecurityManager,
    SimulationSnapshot,
    ProductSnapshot,
    CompetitorSnapshot
)


class TestSecureSimulationDataProvider:
    """Test SecureSimulationDataProvider methods."""
    
    @pytest.fixture
    def data_provider(self):
        """Create SecureSimulationDataProvider instance."""
        return SecureSimulationDataProvider()
    
    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation for testing."""
        sim = Mock()
        sim.day = 5
        sim.now = datetime.now()
        
        # Mock ledger
        sim.ledger = Mock()
        sim.ledger.balance.side_effect = lambda account: {
            'Cash': Money.from_dollars(1000.0),
            'Inventory': Money.from_dollars(500.0),
            'Revenue': Money.from_dollars(2000.0),
            'Fees': Money.from_dollars(200.0),
            'COGS': Money.from_dollars(800.0)
        }.get(account, Money.zero())
        
        # Mock products
        product = Mock()
        product.asin = "B000TEST"
        product.category = "Electronics"
        product.price = Money.from_dollars(19.99)
        product.cost = Money.from_dollars(5.0)
        product.bsr = 50000
        product.ema_sales_velocity = 2.5
        product.ema_conversion = 0.15
        product.trust_score = 0.85
        sim.products = {"B000TEST": product}
        
        # Mock inventory
        sim.inventory = Mock()
        batch = Mock()
        batch.quantity = 100
        sim.inventory._batches = {"B000TEST": [batch]}
        
        # Mock competitors
        competitor = Mock()
        competitor.asin = "B000COMP"
        competitor.price = Money.from_dollars(18.99)
        competitor.bsr = 45000
        competitor.sales_velocity = 3.0
        sim.competitors = [competitor]
        
        # Mock event log
        sim.event_log = [
            {"type": "sale", "message": "Product sold", "date": datetime.now()},
            "Simple event message"
        ]
        
        return sim
    
    def test_register_simulation(self, data_provider, mock_simulation):
        """Test simulation registration."""
        # Execute
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Verify
        assert simulation_id is not None
        assert len(simulation_id) > 0
        assert simulation_id in data_provider._simulation_registry
        
        # Verify access log
        access_log = data_provider.get_access_log()
        assert len(access_log) == 1
        assert access_log[0]["operation"] == "register_simulation"
        assert access_log[0]["status"] == "SUCCESS"
    
    def test_unregister_simulation(self, data_provider, mock_simulation):
        """Test simulation unregistration."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        result = data_provider.unregister_simulation(simulation_id)
        
        # Verify
        assert result is True
        assert simulation_id not in data_provider._simulation_registry
    
    def test_unregister_nonexistent_simulation(self, data_provider):
        """Test unregistering non-existent simulation."""
        # Execute
        result = data_provider.unregister_simulation("nonexistent-id")
        
        # Verify
        assert result is False
    
    def test_get_simulation_snapshot(self, data_provider, mock_simulation):
        """Test getting simulation snapshot."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        snapshot = data_provider.get_simulation_snapshot(simulation_id)
        
        # Verify
        assert snapshot is not None
        assert isinstance(snapshot, SimulationSnapshot)
        assert snapshot.simulation_id == simulation_id
        assert snapshot.current_day == 5
        assert snapshot.cash_balance == str(Money.from_dollars(1000.0))
        assert len(snapshot.products) == 1
        assert len(snapshot.competitors) == 1
        assert len(snapshot.recent_events) == 2
    
    def test_get_simulation_snapshot_nonexistent(self, data_provider):
        """Test getting snapshot for non-existent simulation."""
        # Execute
        snapshot = data_provider.get_simulation_snapshot("nonexistent-id")
        
        # Verify
        assert snapshot is None
    
    def test_get_financial_summary(self, data_provider, mock_simulation):
        """Test getting financial summary."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        summary = data_provider.get_financial_summary(simulation_id)
        
        # Verify
        assert summary is not None
        assert "cash_balance" in summary
        assert "total_revenue" in summary
        assert "net_profit" in summary
        assert "profit_margin_pct" in summary
        
        # Verify calculations
        assert summary["cash_balance"] == str(Money.from_dollars(1000.0))
        assert summary["total_revenue"] == str(Money.from_dollars(2000.0))
        
        # Net profit = revenue - cogs - fees = 2000 - 800 - 200 = 1000
        assert summary["net_profit"] == str(Money.from_dollars(1000.0))
        
        # Profit margin = 1000 / 2000 = 50%
        assert summary["profit_margin_pct"] == 50.0
    
    def test_get_product_performance(self, data_provider, mock_simulation):
        """Test getting product performance data."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        product_snapshot = data_provider.get_product_performance(simulation_id, "B000TEST")
        
        # Verify
        assert product_snapshot is not None
        assert isinstance(product_snapshot, ProductSnapshot)
        assert product_snapshot.asin == "B000TEST"
        assert product_snapshot.category == "Electronics"
        assert product_snapshot.price == str(Money.from_dollars(19.99))
        assert product_snapshot.cost == str(Money.from_dollars(5.0))
        assert product_snapshot.bsr == 50000
        assert product_snapshot.sales_velocity == 2.5
        assert product_snapshot.conversion_rate == 0.15
        assert product_snapshot.trust_score == 0.85
    
    def test_get_product_performance_nonexistent(self, data_provider, mock_simulation):
        """Test getting performance for non-existent product."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        product_snapshot = data_provider.get_product_performance(simulation_id, "NONEXISTENT")
        
        # Verify
        assert product_snapshot is None
    
    def test_get_market_analysis(self, data_provider, mock_simulation):
        """Test getting market analysis."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute
        analysis = data_provider.get_market_analysis(simulation_id)
        
        # Verify
        assert analysis is not None
        assert "total_competitors" in analysis
        assert "market_categories" in analysis
        assert "average_competitor_price" in analysis
        assert "market_concentration" in analysis
        
        assert analysis["total_competitors"] == 1
        assert "Electronics" in analysis["market_categories"]
        assert analysis["market_concentration"] == 1.0  # 1/1 competitor
    
    def test_access_logging(self, data_provider, mock_simulation):
        """Test that all operations are properly logged."""
        # Setup
        simulation_id = data_provider.register_simulation(mock_simulation)
        
        # Execute multiple operations
        data_provider.get_simulation_snapshot(simulation_id)
        data_provider.get_financial_summary(simulation_id)
        data_provider.get_product_performance(simulation_id, "B000TEST")
        data_provider.get_market_analysis(simulation_id)
        data_provider.unregister_simulation(simulation_id)
        
        # Verify all operations were logged
        access_log = data_provider.get_access_log()
        assert len(access_log) == 6  # register + 4 gets + unregister
        
        operations = [entry["operation"] for entry in access_log]
        assert "register_simulation" in operations
        assert "get_simulation_snapshot" in operations
        assert "get_financial_summary" in operations
        assert "get_product_performance" in operations
        assert "get_market_analysis" in operations
        assert "unregister_simulation" in operations
    
    def test_error_handling(self, data_provider):
        """Test error handling for invalid simulation access."""
        # Create a mock simulation that will cause errors
        broken_sim = Mock()
        broken_sim.ledger.balance.side_effect = Exception("Ledger error")
        
        simulation_id = data_provider.register_simulation(broken_sim)
        
        # Execute operations that should handle errors gracefully
        snapshot = data_provider.get_simulation_snapshot(simulation_id)
        financial = data_provider.get_financial_summary(simulation_id)
        
        # Verify errors are handled
        assert snapshot is None
        assert financial == {}
        
        # Verify errors are logged
        access_log = data_provider.get_access_log()
        error_entries = [entry for entry in access_log if "ERROR" in entry["status"]]
        assert len(error_entries) >= 2


class TestDashboardSecurityManager:
    """Test DashboardSecurityManager methods."""
    
    @pytest.fixture
    def security_manager(self):
        """Create DashboardSecurityManager instance."""
        return DashboardSecurityManager()
    
    def test_create_session(self, security_manager):
        """Test session creation."""
        # Execute
        session_id = security_manager.create_session("sim-123")
        
        # Verify
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in security_manager._active_sessions
        
        session = security_manager._active_sessions[session_id]
        assert session["simulation_id"] == "sim-123"
        assert session["access_count"] == 0
    
    def test_validate_session(self, security_manager):
        """Test session validation."""
        # Setup
        session_id = security_manager.create_session("sim-123")
        
        # Execute
        is_valid = security_manager.validate_session(session_id)
        
        # Verify
        assert is_valid is True
        
        # Verify access count incremented
        session = security_manager._active_sessions[session_id]
        assert session["access_count"] == 1
    
    def test_validate_nonexistent_session(self, security_manager):
        """Test validation of non-existent session."""
        # Execute
        is_valid = security_manager.validate_session("nonexistent-session")
        
        # Verify
        assert is_valid is False
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        # Setup
        session_id = security_manager.create_session("sim-123")
        
        # Execute multiple requests within limit
        for _ in range(5):
            allowed = security_manager.check_rate_limit(session_id, max_requests=10, window_minutes=60)
            assert allowed is True
        
        # Execute requests beyond limit
        for _ in range(6):
            security_manager.check_rate_limit(session_id, max_requests=10, window_minutes=60)
        
        # This should be denied
        allowed = security_manager.check_rate_limit(session_id, max_requests=10, window_minutes=60)
        assert allowed is False
    
    def test_cleanup_expired_sessions(self, security_manager):
        """Test cleanup of expired sessions."""
        # Setup - create session and manually expire it
        session_id = security_manager.create_session("sim-123")
        
        # Manually set creation time to 25 hours ago
        session = security_manager._active_sessions[session_id]
        from datetime import timedelta
        session["created_at"] = datetime.now() - timedelta(hours=25)
        
        # Execute cleanup
        security_manager.cleanup_expired_sessions()
        
        # Verify session was removed
        assert session_id not in security_manager._active_sessions
    
    def test_session_age_validation(self, security_manager):
        """Test that old sessions are automatically invalidated."""
        # Setup - create session and manually expire it
        session_id = security_manager.create_session("sim-123")
        
        # Manually set creation time to 25 hours ago
        session = security_manager._active_sessions[session_id]
        session["created_at"] = datetime.now() - timedelta(hours=25)
        
        # Execute validation
        is_valid = security_manager.validate_session(session_id)
        
        # Verify session was invalidated and removed
        assert is_valid is False
        assert session_id not in security_manager._active_sessions


class TestSecureAPIIntegration:
    """Test integration between security manager and data provider."""
    
    def test_secure_workflow(self):
        """Test complete secure workflow."""
        # Setup
        data_provider = SecureSimulationDataProvider()
        security_manager = DashboardSecurityManager()
        
        # Mock simulation
        mock_sim = Mock()
        mock_sim.day = 1
        mock_sim.ledger = Mock()
        mock_sim.ledger.balance.return_value = Money.from_dollars(100.0)
        mock_sim.products = {}
        mock_sim.competitors = []
        mock_sim.event_log = []
        
        # Execute secure workflow
        simulation_id = data_provider.register_simulation(mock_sim)
        session_id = security_manager.create_session(simulation_id)
        
        # Validate session and check rate limit
        assert security_manager.validate_session(session_id) is True
        assert security_manager.check_rate_limit(session_id) is True
        
        # Access data through secure provider
        snapshot = data_provider.get_simulation_snapshot(simulation_id)
        financial = data_provider.get_financial_summary(simulation_id)
        
        # Verify secure access worked
        assert snapshot is not None
        assert financial is not None
        
        # Cleanup
        data_provider.unregister_simulation(simulation_id)
        security_manager.cleanup_expired_sessions()
        
        # Verify cleanup
        assert simulation_id not in data_provider._simulation_registry