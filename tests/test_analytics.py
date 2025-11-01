#!/usr/bin/env python3
"""
Test analytics metrics function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_analytics_metrics():
    """Test that analytics metrics work correctly"""
    try:
        # Mock streamlit session state
        class MockSessionState:
            def __init__(self):
                self.data = {
                    'system_initialized': True,
                    'rag_available': False,
                    'query_history': []
                }

            def get(self, key, default=None):
                return self.data.get(key, default)

        import streamlit as st
        st.session_state = MockSessionState()

        # Import and test the metrics function - handle emoji filename
        import importlib.util
        analytics_path = os.path.join(os.path.dirname(__file__), "..", "web_interface", "pages", "4_📊_Analytics.py")
        spec = importlib.util.spec_from_file_location("analytics", analytics_path)
        if spec and spec.loader:
            analytics_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analytics_module)
            get_system_metrics = analytics_module.get_system_metrics
        else:
            raise ImportError("Could not load analytics module")

        metrics = get_system_metrics()

        print("Analytics metrics test:")
        print(f"System initialized: {metrics['system_initialized']}")
        print(f"RAG available: {metrics['rag_available']}")
        print(f"Total queries: {metrics['total_queries']}")
        print(f"Total documents: {metrics['total_documents']}")
        print(f"Embeddings exist: {metrics['embeddings_exist']}")
        print(f"Index exists: {metrics['index_exists']}")
        print(f"Database connected: {metrics.get('database_connected', 'N/A')}")
        print(f"Search connected: {metrics.get('search_connected', 'N/A')}")

        # Check that required keys exist
        required_keys = ['system_initialized', 'rag_available', 'total_queries',
                        'total_documents', 'embeddings_exist', 'index_exists']
        missing_keys = [key for key in required_keys if key not in metrics]

        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        else:
            print("✅ All required metrics keys present")
            return True

    except Exception as e:
        print(f"❌ Error testing analytics: {e}")
        return False

if __name__ == "__main__":
    success = test_analytics_metrics()
    sys.exit(0 if success else 1)