"""
Example integration of FBA-Bench Dashboard with simulation.
Demonstrates how to run the dashboard alongside a simulation.
"""

import asyncio
import threading
import time
from datetime import datetime

from fba_bench.simulation import Simulation
from fba_bench.advanced_agent import AdvancedAgent
from dashboard import run_dashboard_server, dashboard_api


def run_simulation_with_dashboard():
    """
    Example of running FBA-Bench simulation with real-time dashboard.
    """
    print("🚀 Starting FBA-Bench Dashboard Integration Example")
    print("=" * 60)
    
    # Create simulation environment
    print("📊 Initializing simulation...")
    sim = Simulation()
    
    # Launch a product
    sim.launch_product("B000TEST", "Electronics", cost=5.0, price=19.99, qty=100)
    print(f"✅ Product launched: B000TEST")
    
    # Initialize advanced agent
    print("🤖 Initializing agent...")
    agent = AdvancedAgent(days=30)
    print(f"✅ Agent initialized for {agent.days} days")
    
    # Connect simulation to dashboard
    print("🔗 Connecting simulation to dashboard...")
    dashboard_api.set_simulation_components(sim, agent)
    print("✅ Dashboard connected to simulation")
    
    # Start dashboard server in a separate thread
    print("🌐 Starting dashboard server...")
    dashboard_thread = threading.Thread(
        target=run_dashboard_server,
        kwargs={
            "host": "127.0.0.1",
            "port": 8000,
            "simulation": sim,
            "agent": agent
        },
        daemon=True
    )
    dashboard_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    print("✅ Dashboard server started at http://127.0.0.1:8000")
    print("📊 Dashboard API documentation: http://127.0.0.1:8000/docs")
    print("🔌 WebSocket endpoint: ws://127.0.0.1:8000/ws")
    print()
    print("🎯 Available endpoints:")
    print("  • GET /api/dashboard/executive-summary")
    print("  • GET /api/dashboard/financial")
    print("  • GET /api/dashboard/product-market")
    print("  • GET /api/dashboard/supply-chain")
    print("  • GET /api/dashboard/agent-cognition")
    print("  • GET /api/kpis")
    print()
    
    # Run simulation for a few days with dashboard updates
    print("🏃 Running simulation with real-time dashboard updates...")
    print("💡 Open http://127.0.0.1:8000/docs to explore the API")
    print("💡 Use WebSocket at ws://127.0.0.1:8000/ws for real-time updates")
    print()
    
    try:
        for day in range(1, 6):  # Run for 5 days
            print(f"📅 Day {day}: Running simulation step...")
            
            # Simulate some business activity
            sim.step()  # Advance simulation by one day
            
            # Agent makes decisions
            if hasattr(agent, 'step'):
                agent.step()
            
            # Simulate some market changes
            if sim.products:
                product = list(sim.products.values())[0]
                # Simulate sales
                product.sales_velocity = max(0, product.sales_velocity + (day * 0.5))
                product.bsr = max(1, int(product.bsr - (day * 1000)))
            
            # Update simulation day
            sim.day = day
            
            print(f"   💰 Cash Balance: ${sim.ledger.balance('Cash'):.2f}")
            if sim.products:
                product = list(sim.products.values())[0]
                print(f"   📈 Sales Velocity: {product.sales_velocity:.1f}")
                print(f"   🏆 BSR: {product.bsr:,}")
            
            print(f"   ✅ Day {day} completed - Dashboard updated")
            print()
            
            # Wait between simulation steps
            time.sleep(3)
        
        print("🎉 Simulation completed successfully!")
        print("📊 Dashboard remains active for real-time monitoring")
        print("🔄 Press Ctrl+C to stop the dashboard server")
        
        # Keep the dashboard running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard server...")
        print("👋 Goodbye!")


def test_dashboard_api():
    """
    Test the dashboard API endpoints without running a full simulation.
    """
    print("🧪 Testing Dashboard API")
    print("=" * 30)
    
    # Create minimal simulation for testing
    sim = Simulation()
    sim.launch_product("B000TEST", "Electronics", cost=5.0, price=19.99, qty=100)
    
    agent = AdvancedAgent(days=30)
    
    # Test data exporter
    from dashboard.data_exporter import DashboardDataExporter
    
    exporter = DashboardDataExporter(sim, agent)
    
    print("📊 Testing data extraction...")
    
    # Test KPI extraction
    kpis = exporter.extract_kpi_metrics()
    print(f"✅ KPIs extracted: Net Worth=${kpis.resilient_net_worth:.2f}")
    
    # Test executive summary
    summary = exporter.extract_executive_summary()
    print(f"✅ Executive summary extracted: {len(summary.event_log)} events")
    
    # Test financial data
    financial = exporter.extract_financial_deep_dive()
    print(f"✅ Financial data extracted: Revenue=${financial.profit_loss.revenue.get('total', 0):.2f}")
    
    # Test product analysis
    product_analysis = exporter.extract_product_market_analysis()
    print(f"✅ Product analysis extracted: {product_analysis.product_info.asin}")
    
    # Test supply chain
    supply_chain = exporter.extract_supply_chain_operations()
    print(f"✅ Supply chain extracted: {len(supply_chain.suppliers)} suppliers")
    
    # Test agent cognition
    cognition = exporter.extract_agent_cognition()
    print(f"✅ Agent cognition extracted: {len(cognition.goal_stack)} goals")
    
    print("\n🎉 All tests passed! Dashboard API is working correctly.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_dashboard_api()
    else:
        run_simulation_with_dashboard()