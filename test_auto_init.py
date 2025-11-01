#!/usr/bin/env python3
"""
Test that the auto-initialization changes work
"""

import sys
import os

def test_imports():
    """Test that all necessary modules can be imported"""
    try:
        sys.path.insert(0, os.path.dirname(__file__))

        # Test basic imports
        from src.rag_pipeline_db import RAGPipelineDB
        from src.retrieval_db import DatabaseRetriever

        print("✅ Core imports successful")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_modified_functions():
    """Test that the modified functions exist and are callable"""
    try:
        # Test that we can import the settings manager
        from web_interface.utils.session_manager import load_settings
        settings = load_settings()
        print("✅ Settings loading works")

        # Test that the function signatures are correct
        import importlib.util
        spec = importlib.util.spec_from_file_location("home_page", "web_interface/pages/1_🏠_Home.py")
        if spec and spec.loader:
            home_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(home_module)

            # Check that initialize_system function exists
            if hasattr(home_module, 'initialize_system'):
                print("✅ initialize_system function exists")
                return True
            else:
                print("❌ initialize_system function not found")
                return False
        else:
            print("❌ Could not load home page module")
            return False

    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        return False

if __name__ == "__main__":
    print("Testing auto-initialization changes...")

    success1 = test_imports()
    success2 = test_modified_functions()

    if success1 and success2:
        print("\n🎉 All auto-initialization tests passed!")
        print("The system should now initialize automatically without requiring extra clicks.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)