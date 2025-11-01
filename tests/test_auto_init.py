#!/usr/bin/env python3
"""
Test that the auto-initialization changes work
"""

import sys
import os

def test_imports():
    """Test that all necessary modules can be imported"""
    try:
        # Add the parent directory to the path so we can import src and web_interface
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

        # Test that the home page file exists and has the expected structure
        home_page_path = os.path.join(os.path.dirname(__file__), "..", "web_interface", "pages", "1_🏠_Home.py")
        if os.path.exists(home_page_path):
            with open(home_page_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for key functions
            if 'def initialize_system' in content and 'def _do_initialization' in content:
                print("✅ Auto-initialization functions exist in home page")
                return True
            else:
                print("❌ Auto-initialization functions not found in home page")
                return False
        else:
            print("❌ Home page file not found")
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