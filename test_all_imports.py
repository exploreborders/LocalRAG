#!/usr/bin/env python3
"""
Test script to verify all web interface imports work correctly.
"""

import sys
import os
from pathlib import Path

# Set environment variables
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_DB"] = "rag_system"
os.environ["POSTGRES_USER"] = "christianhein"
os.environ["OPENSEARCH_HOST"] = "localhost"
os.environ["OPENSEARCH_PORT"] = "9200"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"

# Set up paths like the web interface does
ROOT = Path(__file__).parent
SRC = ROOT / "src"
WEB = ROOT / "web_interface"

for path in [ROOT, SRC, WEB]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Set PYTHONPATH
current_pythonpath = os.environ.get("PYTHONPATH", "")
paths_to_add = [str(ROOT), str(SRC), str(WEB)]
for path in paths_to_add:
    if path not in current_pythonpath:
        if current_pythonpath:
            current_pythonpath = f"{path}:{current_pythonpath}"
        else:
            current_pythonpath = path
os.environ["PYTHONPATH"] = current_pythonpath


def test_all_imports():
    """Test all imports that the web interface uses."""
    print("🧪 Testing all web interface imports...")

    imports_to_test = [
        # Core modules
        (
            "core.document_manager",
            "UploadProcessor, DocumentProcessor, TagManager, CategoryManager",
        ),
        ("database.models", "SessionLocal, Document, DocumentChunk, DocumentEmbedding"),
        ("core.retrieval", "DatabaseRetriever, RAGPipelineDB"),
        ("src.cache.redis_cache", "RedisCache"),
        ("src.utils.error_handler", "ErrorHandler"),
        ("src.utils.progress_tracker", "ProgressTracker"),
        ("src.utils.rate_limiter", "RateLimiter"),
        # Web interface components
        ("web_interface.components.tag_analytics", "render_tag_suggestions"),
        ("web_interface.components.query_interface", "render_query_input"),
        ("web_interface.components.results_display", "render_results"),
        ("web_interface.utils.session_manager", "load_settings"),
        # External dependencies
        ("streamlit", "st"),
        ("sqlalchemy", "create_engine"),
        ("elasticsearch", "Elasticsearch"),
        ("redis", "Redis"),
        ("networkx", "Graph"),
        ("plotly", "graph_objects"),
        ("wordcloud", "WordCloud"),
        ("matplotlib", "pyplot"),
    ]

    failed_imports = []

    for module_name, items in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[items.split(", ")[0].strip()])
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failed_imports.append((module_name, str(e)))

    if failed_imports:
        print(f"\n❌ {len(failed_imports)} imports failed:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        return False

    print("\n✅ All imports successful! Web interface should work now.")
    return True


if __name__ == "__main__":
    success = test_all_imports()
    if success:
        print("\n🎉 Ready to run the web interface!")
        print("   Run: python run_web.py")
    else:
        print("\n❌ Some imports are still failing.")
    sys.exit(0 if success else 1)
