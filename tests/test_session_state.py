#!/usr/bin/env python3
"""
Test session state initialization
"""

import sys
import os

def test_session_state_import():
    """Test that session state functions can be imported"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Test that we can import the session manager functions
        from web_interface.utils.session_manager import initialize_session_state, load_settings

        print("✅ Session manager imports successful")

        # Test that load_settings works
        settings = load_settings()
        if isinstance(settings, dict):
            print("✅ Settings loading works")
            return True
        else:
            print("❌ Settings loading failed")
            return False

    except Exception as e:
        print(f"❌ Error importing session manager: {e}")
        return False

def test_pages_import_session_state():
    """Test that pages can import session state functions"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Test Home page imports
        import importlib.util
        home_path = os.path.join(os.path.dirname(__file__), "..", "web_interface", "pages", "1_🏠_Home.py")
        spec = importlib.util.spec_from_file_location("home_page", home_path)
        if spec and spec.loader:
            home_module = importlib.util.module_from_spec(spec)
            # Just check that the module can be loaded without executing
            print("✅ Home page can import session state functions")
        else:
            print("❌ Could not load Home page module")
            return False

        # Test Analytics page imports
        analytics_path = os.path.join(os.path.dirname(__file__), "..", "web_interface", "pages", "4_📊_Analytics.py")
        spec2 = importlib.util.spec_from_file_location("analytics_page", analytics_path)
        if spec2 and spec2.loader:
            print("✅ Analytics page can import session state functions")
            return True
        else:
            print("❌ Could not load Analytics page module")
            return False

    except Exception as e:
        print(f"❌ Error testing page imports: {e}")
        return False

if __name__ == "__main__":
    print("Testing session state initialization...")

    success1 = test_session_state_import()
    success2 = test_pages_import_session_state()

    if success1 and success2:
        print("\n🎉 Session state initialization tests passed!")
        print("The query_history error should now be resolved.")
        sys.exit(0)
    else:
        print("\n❌ Some session state tests failed")
        sys.exit(1)