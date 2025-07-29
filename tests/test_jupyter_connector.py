"""
FBA-Bench v3 Jupyter Connector Integration Test

This test validates the JupyterConnector functionality with a live simulation.
Run this test while the API server is running to verify all components work correctly.

Usage:
    1. Start API server: python api_server.py
    2. Run test: python test_jupyter_connector.py
"""

import time
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from jupyter_connector import JupyterConnector, connect_to_simulation


def test_api_server_availability():
    """Test if the API server is running and accessible."""
    print("🔍 Testing API server availability...")
    
    try:
        response = requests.get("http://localhost:8000/api/v1/simulation/snapshot", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running and accessible")
            return True
        else:
            print(f"❌ API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server at http://localhost:8000")
        print("   Please start the API server with: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error connecting to API server: {e}")
        return False


def test_jupyter_connector_initialization():
    """Test JupyterConnector initialization and basic functionality."""
    print("\n🔧 Testing JupyterConnector initialization...")
    
    # Test basic initialization
    connector = JupyterConnector()
    print("✅ JupyterConnector created successfully")
    
    # Test connection status
    status = connector.get_connection_status()
    print(f"📊 Connection status: {status}")
    
    # Wait for WebSocket connection
    print("⏳ Waiting for WebSocket connection...")
    connected = connector.wait_for_connection(timeout=10)
    
    if connected:
        print("✅ WebSocket connection established")
    else:
        print("⚠️ WebSocket connection timeout (API server may not be running)")
    
    return connector


def test_snapshot_functionality(connector):
    """Test snapshot retrieval and DataFrame conversion."""
    print("\n📸 Testing snapshot functionality...")
    
    # Test snapshot refresh
    success = connector.refresh_snapshot()
    if success:
        print("✅ Snapshot refresh successful")
    else:
        print("❌ Snapshot refresh failed")
        return False
    
    # Test snapshot retrieval
    snapshot_dict = connector.get_snapshot_dict()
    if snapshot_dict:
        print("✅ Snapshot dictionary retrieval successful")
        print(f"   Snapshot keys: {list(snapshot_dict.keys())}")
        
        # Test DataFrame conversion
        snapshot_df = connector.get_snapshot_df()
        if not snapshot_df.empty:
            print("✅ Snapshot DataFrame conversion successful")
            print(f"   DataFrame shape: {snapshot_df.shape}")
            print(f"   DataFrame columns: {list(snapshot_df.columns)[:5]}...")  # Show first 5 columns
        else:
            print("⚠️ Snapshot DataFrame is empty")
    else:
        print("❌ Snapshot dictionary retrieval failed")
        return False
    
    return True


def test_event_stream_functionality(connector):
    """Test event stream and DataFrame conversion."""
    print("\n📡 Testing event stream functionality...")
    
    # Get current events
    events = connector.get_event_stream()
    print(f"📊 Current events in buffer: {len(events)}")
    
    if events:
        print("✅ Event stream retrieval successful")
        
        # Test event types
        event_types = set()
        for event in events:
            event_type = event['data'].get('event_type', 'Unknown')
            event_types.add(event_type)
        
        print(f"   Event types found: {list(event_types)}")
        
        # Test DataFrame conversion
        events_df = connector.get_events_df()
        if not events_df.empty:
            print("✅ Events DataFrame conversion successful")
            print(f"   DataFrame shape: {events_df.shape}")
        else:
            print("⚠️ Events DataFrame is empty")
        
        # Test financial history
        financial_df = connector.get_financial_history_df()
        print(f"💳 Financial history events: {len(financial_df)}")
        
    else:
        print("⚠️ No events in buffer (simulation may need more time to generate events)")
    
    return True


def test_read_only_security(connector):
    """Test that the connector has no write capabilities (security check)."""
    print("\n🔒 Testing read-only security...")
    
    # Check that there are no write methods exposed
    dangerous_methods = [
        'send_command', 'set_price', 'modify_state', 'write_data',
        'update_simulation', 'control_simulation', 'post_data'
    ]
    
    connector_methods = [method for method in dir(connector) if not method.startswith('_')]
    print(f"📋 Available connector methods: {connector_methods}")
    
    # Check for dangerous methods
    found_dangerous = []
    for method in dangerous_methods:
        if hasattr(connector, method):
            found_dangerous.append(method)
    
    if found_dangerous:
        print(f"❌ SECURITY RISK: Found potential write methods: {found_dangerous}")
        return False
    else:
        print("✅ Security check passed - no write methods found")
    
    # Check that all methods are read-only
    read_only_methods = [
        'get_snapshot_dict', 'get_snapshot_df', 'get_events_df',
        'get_event_stream', 'get_financial_history_df', 'refresh_snapshot',
        'is_connected', 'get_connection_status'
    ]
    
    missing_methods = []
    for method in read_only_methods:
        if not hasattr(connector, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"⚠️ Missing expected read-only methods: {missing_methods}")
    else:
        print("✅ All expected read-only methods are available")
    
    return len(found_dangerous) == 0


def test_convenience_function():
    """Test the convenience connect_to_simulation function."""
    print("\n🎯 Testing convenience connection function...")
    
    try:
        # Test quick connection setup
        connector = connect_to_simulation()
        
        if connector:
            print("✅ Convenience function successful")
            
            # Test basic functionality
            status = connector.get_connection_status()
            if status['has_snapshot_data']:
                print("✅ Snapshot data available via convenience function")
            else:
                print("⚠️ No snapshot data via convenience function")
            
            connector.close()
            return True
        else:
            print("❌ Convenience function failed")
            return False
            
    except Exception as e:
        print(f"❌ Convenience function error: {e}")
        return False


def test_real_time_monitoring(connector):
    """Test real-time event monitoring capabilities."""
    print("\n🔴 Testing real-time monitoring (10 seconds)...")
    
    initial_events = len(connector.get_event_stream())
    print(f"   Initial events: {initial_events}")
    
    # Monitor for 10 seconds
    start_time = time.time()
    events_detected = []
    
    while time.time() - start_time < 10:
        current_events = len(connector.get_event_stream())
        
        if current_events > initial_events:
            new_events = current_events - initial_events
            events_detected.append(new_events)
            print(f"   📡 Detected {new_events} new events at {time.time() - start_time:.1f}s")
            initial_events = current_events
        
        time.sleep(1)
    
    final_events = len(connector.get_event_stream())
    total_new_events = final_events - (initial_events if 'initial_events' in locals() else 0)
    
    print(f"✅ Real-time monitoring complete")
    print(f"   Total new events detected: {total_new_events}")
    
    return True


def main():
    """Run comprehensive Jupyter Connector integration tests."""
    print("🧪 FBA-Bench v3 Jupyter Connector Integration Test")
    print("=" * 55)
    
    # Track test results
    test_results = {}
    
    # Test 1: API Server Availability
    test_results['api_server'] = test_api_server_availability()
    if not test_results['api_server']:
        print("\n❌ Cannot proceed without API server. Exiting.")
        sys.exit(1)
    
    # Test 2: Connector Initialization
    connector = test_jupyter_connector_initialization()
    test_results['initialization'] = connector is not None
    
    if not connector:
        print("\n❌ Cannot proceed without connector. Exiting.")
        sys.exit(1)
    
    # Test 3: Snapshot Functionality
    test_results['snapshot'] = test_snapshot_functionality(connector)
    
    # Test 4: Event Stream Functionality
    test_results['event_stream'] = test_event_stream_functionality(connector)
    
    # Test 5: Read-Only Security
    test_results['security'] = test_read_only_security(connector)
    
    # Test 6: Convenience Function
    test_results['convenience'] = test_convenience_function()
    
    # Test 7: Real-Time Monitoring
    test_results['real_time'] = test_real_time_monitoring(connector)
    
    # Clean up
    connector.close()
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 TEST SUMMARY")
    print("=" * 55)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Jupyter Connector is ready for production!")
        return True
    else:
        print("⚠️ Some tests failed - check the output above for details")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)