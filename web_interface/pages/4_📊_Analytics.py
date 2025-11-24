#!/usr/bin/env python3
"""
Analytics Page - Performance Dashboard
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Find project root (two levels up from /web_interface/pages/)
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
WEB = ROOT / "web_interface"

for p in (SRC, WEB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import utilities
from web_interface.utils.session_manager import initialize_session_state

# Page configuration
st.set_page_config(page_title="Local RAG - Analytics", page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .page-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .chart-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_system_metrics():
    """Get comprehensive system metrics from database and services"""
    metrics = {
        "total_queries": len(st.session_state.get("query_history", [])),
        "current_time": datetime.now(),
    }

    # Database-driven document metrics
    try:
        from database.models import (
            SessionLocal,
            Document,
            DocumentChunk,
            DocumentChapter,
            DocumentEmbedding,
            DocumentTopic,
            DocumentTagAssignment,
            DocumentCategoryAssignment,
            ProcessingJob,
        )
        from sqlalchemy import func, distinct

        db = SessionLocal()

        # Core document metrics
        total_docs = db.query(Document).count()
        processed_docs = (
            db.query(Document).filter(Document.status == "processed").count()
        )
        total_chunks = db.query(DocumentChunk).count()
        total_chapters = db.query(DocumentChapter).count()
        total_embeddings = db.query(DocumentEmbedding).count()

        # AI enrichment metrics
        docs_with_summary = (
            db.query(Document).filter(Document.document_summary.isnot(None)).count()
        )
        docs_with_topics = (
            db.query(Document).filter(Document.key_topics.isnot(None)).count()
        )
        docs_with_reading_time = (
            db.query(Document).filter(Document.reading_time_minutes.isnot(None)).count()
        )

        # Document type breakdown
        doc_types = (
            db.query(Document.content_type, func.count(Document.id).label("count"))
            .group_by(Document.content_type)
            .all()
        )

        # Language breakdown
        languages = (
            db.query(Document.detected_language, func.count(Document.id).label("count"))
            .filter(Document.detected_language.isnot(None))
            .group_by(Document.detected_language)
            .all()
        )

        # Topic and tag counts
        total_topics = db.query(DocumentTopic).distinct(DocumentTopic.topic_id).count()
        total_tags = (
            db.query(DocumentTagAssignment)
            .distinct(DocumentTagAssignment.tag_id)
            .count()
        )
        total_categories = (
            db.query(DocumentCategoryAssignment)
            .distinct(DocumentCategoryAssignment.category_id)
            .count()
        )

        # File size calculation (from database if available, fallback to filesystem)
        total_size_bytes = 0
        docs_with_paths = db.query(Document).filter(Document.filepath.isnot(None)).all()
        for doc in docs_with_paths:
            if doc.filepath and os.path.exists(doc.filepath):
                try:
                    total_size_bytes += os.path.getsize(doc.filepath)
                except:
                    pass

        metrics.update(
            {
                "database_connected": True,
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "total_chunks": total_chunks,
                "total_chapters": total_chapters,
                "total_embeddings": total_embeddings,
                "total_doc_size": total_size_bytes,
                # AI enrichment
                "docs_with_summary": docs_with_summary,
                "docs_with_topics": docs_with_topics,
                "docs_with_reading_time": docs_with_reading_time,
                "ai_ready_count": docs_with_summary + docs_with_topics,
                # Content breakdown
                "doc_types": dict(
                    (row.content_type, row.count)
                    for row in doc_types
                    if row.content_type
                ),
                "languages": dict(
                    (row.detected_language, row.count)
                    for row in languages
                    if row.detected_language
                ),
                # Organization
                "total_topics": total_topics,
                "total_tags": total_tags,
                "total_categories": total_categories,
                # System readiness
                "chapters_exist": total_chapters > 0,
                "chunks_exist": total_chunks > 0,
                "embeddings_exist": total_embeddings > 0,
            }
        )

        db.close()

    except Exception as e:
        metrics.update(
            {
                "database_connected": False,
                "total_documents": 0,
                "processed_documents": 0,
                "total_chunks": 0,
                "total_chapters": 0,
                "total_embeddings": 0,
                "total_doc_size": 0,
                "embeddings_exist": False,
                "chapters_exist": False,
                "chunks_exist": False,
            }
        )

    # Vector search connectivity (Elasticsearch/OpenSearch)
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False,
        )
        metrics["search_connected"] = es.ping()

        if metrics["search_connected"]:
            # Get vector index stats for our documents index
            try:
                index_info = es.cat.indices(index="documents", format="json")
                if index_info and len(index_info) > 0:
                    total_vectors = int(index_info[0].get("docs.count", 0))
                    metrics["total_vectors"] = total_vectors
                else:
                    metrics["total_vectors"] = 0
            except Exception:
                metrics["total_vectors"] = 0
        else:
            metrics["total_vectors"] = 0

    except Exception:
        metrics["search_connected"] = False
        metrics["total_vectors"] = 0

    # Set embeddings_exist based on total_vectors (after Elasticsearch check)
    metrics["embeddings_exist"] = metrics.get("total_vectors", 0) > 0

    # Redis cache metrics
    try:
        from cache.redis_cache import RedisCache

        cache = RedisCache()
        cache_stats = cache.get_stats()

        metrics.update(
            {
                "cache_connected": True,
                "cache_enabled": True,  # Redis is always enabled if connected
                "cached_responses": cache_stats.get("total_keys", 0),
                "cache_memory": cache_stats.get("memory_used", "unknown"),
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "cache_uptime_days": cache_stats.get("uptime_days", 0),
            }
        )
    except Exception as e:
        metrics.update(
            {
                "cache_connected": False,
                "cache_enabled": False,
                "cached_responses": 0,
                "cache_memory": f"Error: {str(e)[:20]}...",
                "cache_hit_rate": 0,
                "cache_uptime_days": 0,
            }
        )

    # System initialization status
    metrics["system_initialized"] = st.session_state.get("system_initialized", False)
    metrics["rag_available"] = st.session_state.get("rag_available", False)

    return metrics


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def create_query_history_chart():
    """Create a chart of query history over time"""
    history = st.session_state.get("query_history", [])

    if not history:
        return None

    # Create DataFrame from history
    df_data = []
    for item in history:
        df_data.append(
            {
                "timestamp": item.get("timestamp", datetime.now()),
                "mode": item.get("mode", "unknown"),
                "processing_time": item.get("processing_time", 0),
            }
        )

    df = pd.DataFrame(df_data)

    # Group by hour and mode
    df["hour"] = df["timestamp"].dt.floor("h")
    df_grouped = df.groupby(["hour", "mode"]).size().unstack(fill_value=0)

    return df_grouped


def create_performance_chart():
    """Create a chart of query performance over time"""
    history = st.session_state.get("query_history", [])

    if not history:
        return None

    # Create DataFrame from history
    df_data = []
    for item in history[-20:]:  # Last 20 queries
        df_data.append(
            {
                "timestamp": item.get("timestamp", datetime.now()),
                "processing_time": item.get("processing_time", 0),
                "mode": item.get("mode", "unknown"),
            }
        )

    df = pd.DataFrame(df_data)
    return df


def initialize_system_if_needed():
    """Initialize the RAG system if not already done"""
    if not st.session_state.get("system_initialized", False):
        try:
            # Import system components
            from core.retrieval import DatabaseRetriever, RAGPipelineDB
            from web_interface.utils.session_manager import load_settings

            # Get configured models from settings
            settings = load_settings()
            embedding_model = settings.get("retrieval", {}).get(
                "embedding_model", "nomic-ai/nomic-embed-text-v1.5"
            )
            llm_model = settings.get("generation", {}).get("model", "llama2")
            cache_enabled = settings.get("cache", {}).get("enabled", True)
            cache_settings = settings.get("cache", {})

            # Initialize retriever
            st.session_state.retriever = DatabaseRetriever(embedding_model)

            # Try to initialize RAG pipeline
            try:
                st.session_state.rag_pipeline = RAGPipelineDB(
                    embedding_model,
                    llm_model,
                    cache_enabled=cache_enabled,
                    cache_settings=cache_settings,
                )
                st.session_state.rag_available = True
            except Exception:
                st.session_state.rag_pipeline = None
                st.session_state.rag_available = False

            st.session_state.system_initialized = True

        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            st.session_state.system_initialized = False


def main():
    """Main page content"""
    # Initialize session state
    initialize_session_state()

    # Initialize system if needed
    initialize_system_if_needed()

    st.markdown(
        '<h1 class="page-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True
    )
    st.markdown("Comprehensive performance monitoring for your Local RAG system")

    # Get current metrics
    metrics = get_system_metrics()

    # Core System Metrics
    st.markdown("### 📈 Core System Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_docs = metrics.get("total_documents", 0)
        processed_docs = metrics.get("processed_documents", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total_docs}</div>
            <div class="metric-label">Total Documents</div>
            <div style="font-size: 0.8rem; color: #666;">{processed_docs} processed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        total_chunks = metrics.get("total_chunks", 0)
        total_chapters = metrics.get("total_chapters", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total_chunks:,}</div>
            <div class="metric-label">Content Chunks</div>
            <div style="font-size: 0.8rem; color: #666;">{total_chapters} chapters</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        total_vectors = metrics.get("total_vectors", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total_vectors:,}</div>
            <div class="metric-label">Vector Embeddings</div>
            <div style="font-size: 0.8rem; color: #666;">Search index</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        doc_size = format_file_size(metrics.get("total_doc_size", 0))
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{doc_size}</div>
            <div class="metric-label">Total Size</div>
            <div style="font-size: 0.8rem; color: #666;">On disk</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # AI Enrichment Metrics
    st.markdown("### 🤖 AI Enrichment Status")

    ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)

    with ai_col1:
        docs_with_summary = metrics.get("docs_with_summary", 0)
        total_docs = metrics.get("total_documents", 1)
        st.metric("📝 Summaries", f"{docs_with_summary}/{total_docs}")

    with ai_col2:
        docs_with_topics = metrics.get("docs_with_topics", 0)
        st.metric("🏷️ Topics", f"{docs_with_topics}/{total_docs}")

    with ai_col3:
        docs_with_reading_time = metrics.get("docs_with_reading_time", 0)
        st.metric("⏱️ Reading Time", f"{docs_with_reading_time}/{total_docs}")

    with ai_col4:
        ai_ready = metrics.get("ai_ready_count", 0)
        st.metric("🎯 AI Ready", f"{ai_ready}/{total_docs}")

    # Document Type & Language Breakdown
    if metrics.get("doc_types") or metrics.get("languages"):
        st.markdown("### 📋 Content Analysis")

        breakdown_col1, breakdown_col2 = st.columns(2)

        with breakdown_col1:
            if metrics.get("doc_types"):
                st.markdown("**Document Types**")
                doc_types = metrics["doc_types"]
                for doc_type, count in doc_types.items():
                    if doc_type:  # Skip None values
                        st.write(f"• {doc_type.upper()}: {count}")

        with breakdown_col2:
            if metrics.get("languages"):
                st.markdown("**Detected Languages**")
                languages = metrics["languages"]
                language_names = {
                    "en": "English",
                    "de": "German",
                    "fr": "French",
                    "es": "Spanish",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "nl": "Dutch",
                    "sv": "Swedish",
                    "pl": "Polish",
                    "zh": "Chinese",
                    "ja": "Japanese",
                    "ko": "Korean",
                }
                for lang_code, count in languages.items():
                    lang_name = language_names.get(lang_code, lang_code.upper())
                    st.write(f"• {lang_name}: {count}")

    # Organization Metrics
    org_metrics = []
    if metrics.get("total_topics", 0) > 0:
        org_metrics.append(f"🏷️ {metrics['total_topics']} Topics")
    if metrics.get("total_tags", 0) > 0:
        org_metrics.append(f"🏷️ {metrics['total_tags']} Tags")
    if metrics.get("total_categories", 0) > 0:
        org_metrics.append(f"📂 {metrics['total_categories']} Categories")

    if org_metrics:
        st.markdown("### 📚 Organization")
        st.info(" • ".join(org_metrics))

    # Tag Analytics (detailed breakdown)
    if metrics.get("total_tags", 0) > 0:
        st.markdown("### 🏷️ Tag Analytics")

        try:
            from database.models import SessionLocal, DocumentTag, DocumentTagAssignment
            from sqlalchemy import func

            db = SessionLocal()

            # Get tag usage statistics
            tag_stats = (
                db.query(
                    DocumentTag.name,
                    DocumentTag.color,
                    func.count(DocumentTagAssignment.document_id).label("usage_count"),
                )
                .join(DocumentTagAssignment)
                .group_by(DocumentTag.id, DocumentTag.name, DocumentTag.color)
                .order_by(func.count(DocumentTagAssignment.document_id).desc())
                .all()
            )  # Get all tags for cloud

            db.close()

            if tag_stats:
                st.markdown("**Most Used Tags:**")
                tag_cols = st.columns(
                    min(len(tag_stats[:10]), 5)
                )  # Max 5 tags per row, top 10

                for i, (tag_name, tag_color, usage_count) in enumerate(tag_stats[:10]):
                    col_idx = i % 5
                    with tag_cols[col_idx]:
                        st.markdown(
                            f'<div style="background-color: {tag_color}; color: white; padding: 6px 12px; '
                            f"border-radius: 15px; display: inline-block; font-size: 0.9em; font-weight: 500; "
                            f'text-align: center; margin: 2px;">{tag_name} ({usage_count})</div>',
                            unsafe_allow_html=True,
                        )

                # Tag distribution chart
                if len(tag_stats) > 1:
                    st.markdown("**Tag Usage Distribution:**")
                    chart_data = pd.DataFrame(
                        {
                            "Tag": [
                                stat[0] for stat in tag_stats[:15]
                            ],  # Top 15 for chart
                            "Documents": [stat[2] for stat in tag_stats[:15]],
                        }
                    )
                    st.bar_chart(chart_data.set_index("Tag"), height=200)

                # Tag Cloud Visualization
                if len(tag_stats) > 2:
                    st.markdown("**☁️ Tag Cloud:**")
                    try:
                        # Create word cloud data
                        from wordcloud import WordCloud
                        import matplotlib.pyplot as plt
                        from io import BytesIO
                        import base64

                        # Prepare word frequencies
                        word_freq = {stat[0]: stat[2] for stat in tag_stats}

                        # Generate word cloud
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color="white",
                            colormap="viridis",
                            max_words=50,
                            prefer_horizontal=0.7,
                        ).generate_from_frequencies(word_freq)

                        # Convert to image
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        ax.set_title("Tag Usage Cloud", fontsize=16, fontweight="bold")

                        # Save to buffer
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                        buf.seek(0)
                        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()
                        plt.close(fig)

                        # Display
                        st.markdown(
                            f'<div style="text-align: center;"><img src="data:image/png;base64,{image_base64}" '
                            'style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>',
                            unsafe_allow_html=True,
                        )

                    except ImportError:
                        st.info(
                            "📦 Install wordcloud for tag cloud visualization: `pip install wordcloud matplotlib`"
                        )
                    except Exception as e:
                        st.warning(f"Could not generate tag cloud: {e}")

        except Exception as e:
            st.warning(f"Could not load detailed tag analytics: {e}")

    # Category Analytics (detailed breakdown)
    if metrics.get("total_categories", 0) > 0:
        st.markdown("### 📂 Category Analytics")

        try:
            from core.document_manager import CategoryManager
            from database.models import SessionLocal

            db = SessionLocal()
            cat_manager = CategoryManager(db)

            # Get category usage statistics
            cat_stats = cat_manager.get_category_usage_stats()

            db.close()

            if cat_stats:
                # Sort by document count
                cat_stats.sort(key=lambda x: x["document_count"], reverse=True)

                st.markdown("**Most Used Categories:**")
                cat_cols = st.columns(
                    min(len(cat_stats), 4)
                )  # Max 4 categories per row

                for i, cat in enumerate(cat_stats[:8]):  # Show top 8
                    col_idx = i % 4
                    with cat_cols[col_idx]:
                        doc_count = cat["document_count"]
                        st.metric(
                            cat["name"],
                            f"{doc_count} docs",
                            help=cat.get("description", ""),
                        )

                # Category distribution chart
                if len(cat_stats) > 1:
                    st.markdown("**Category Usage Distribution:**")
                    chart_data = pd.DataFrame(
                        {
                            "Category": [stat["name"] for stat in cat_stats],
                            "Documents": [stat["document_count"] for stat in cat_stats],
                        }
                    )
                    st.bar_chart(chart_data.set_index("Category"), height=250)

                # Category hierarchy overview
                st.markdown("**Category Hierarchy:**")
                try:
                    db = SessionLocal()
                    cat_manager = CategoryManager(db)
                    category_tree = cat_manager.get_category_tree()
                    db.close()

                    if category_tree:

                        def display_tree_summary(categories, level=0):
                            for cat in categories:
                                indent = "  " * level
                                doc_count = cat.get("document_count", 0)
                                st.caption(
                                    f"{indent}📁 {cat['name']} ({doc_count} docs)"
                                )
                                if cat.get("children"):
                                    display_tree_summary(cat["children"], level + 1)

                        display_tree_summary(category_tree)
                    else:
                        st.caption("No hierarchical categories")

                except Exception as e:
                    st.caption(f"Could not load hierarchy: {e}")

        except Exception as e:
            st.warning(f"Could not load detailed category analytics: {e}")

    # Knowledge Graph Visualization
    st.markdown("### 🕸️ Knowledge Graph Analytics")

    try:
        from core.knowledge_graph import KnowledgeGraph
        from database.models import SessionLocal
        import networkx as nx

        db = SessionLocal()
        kg = KnowledgeGraph(db)

        # Get graph statistics
        graph_stats = kg.get_graph_statistics()

        # Build relationships for visualization
        kg.build_relationships_from_cooccurrence(min_occurrences=2)
        kg.infer_tag_category_relationships()

        # Get top relationships for visualization
        relationships_data = []
        tags_with_relationships = set()
        for tag, relations in kg._relationship_cache.items():
            if relations:  # Only count tags that have relationships
                tags_with_relationships.add(tag)
            for rel in relations[:3]:  # Top 3 relationships per tag
                relationships_data.append(
                    {
                        "source": tag,
                        "target": rel["related_tag"],
                        "type": rel["type"],
                        "strength": rel["strength"],
                        "evidence": rel["evidence_count"],
                    }
                )

        # Update statistics with current relationship data
        graph_stats["tags"]["with_relationships"] = len(tags_with_relationships)
        graph_stats["relationships"] = {
            "total": len(relationships_data),
            "tag_relationships_cached": len(kg._relationship_cache)
            if kg._relationship_cache
            else 0,
        }

        db.close()

        # Knowledge Graph Metrics
        kg_col1, kg_col2, kg_col3, kg_col4 = st.columns(4)

        with kg_col1:
            total_tags = graph_stats["tags"]["total"]
            st.metric("🏷️ Total Tags", total_tags)

        with kg_col2:
            tags_with_rels = graph_stats["tags"]["with_relationships"]
            st.metric("🔗 Tags with Relationships", tags_with_rels)

        with kg_col3:
            total_cats = graph_stats["categories"]["total"]
            st.metric("📂 Categories", total_cats)

        with kg_col4:
            total_relationships = len(relationships_data)
            st.metric("⚡ Relationships", total_relationships)

        # Knowledge Graph Network Visualization
        if relationships_data:
            st.markdown("**Knowledge Graph Network:**")

            # Create network data for visualization
            import plotly.graph_objects as go

            # Build network
            G = nx.Graph()

            # Add nodes and edges
            for rel in relationships_data:
                G.add_node(rel["source"], node_type="tag")
                G.add_node(rel["target"], node_type="tag")
                G.add_edge(
                    rel["source"],
                    rel["target"],
                    weight=rel["strength"],
                    relationship_type=rel["type"],
                )

            # Calculate advanced graph metrics
            if len(G.nodes()) > 1:
                # Centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
                closeness_centrality = nx.closeness_centrality(G)

                # Community detection (greedy modularity)
                try:
                    from networkx.algorithms.community import (
                        greedy_modularity_communities,
                    )

                    communities = list(
                        greedy_modularity_communities(G, weight="weight")
                    )
                    community_map = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            community_map[node] = i
                except:
                    community_map = {node: 0 for node in G.nodes()}

                # Calculate node positions
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

                # Create edge traces
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2].get("weight", 1))

                # Normalize edge weights for visualization
                if edge_weights:
                    max_weight = max(edge_weights)
                    min_weight = min(edge_weights)
                    if max_weight > min_weight:
                        edge_widths = [
                            1 + 3 * (w - min_weight) / (max_weight - min_weight)
                            for w in edge_weights
                        ]
                    else:
                        edge_widths = [2] * len(edge_weights)
                else:
                    edge_widths = [2] * len(G.edges())

                # Use average edge width for the trace (Plotly doesn't support per-edge widths in multi-edge traces)
                avg_edge_width = (
                    sum(edge_widths) / len(edge_widths) if edge_widths else 2
                )

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=avg_edge_width, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )

                # Create node traces with advanced coloring
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                node_size = []
                node_hover_text = []

                # Color palette for communities
                colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                ]

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)

                    # Node size based on degree centrality
                    degree_cent = degree_centrality.get(node, 0)
                    node_size.append(max(15, min(40, 15 + degree_cent * 50)))

                    # Color based on community
                    comm_id = community_map.get(node, 0)
                    node_color.append(colors[comm_id % len(colors)])

                    # Hover text with centrality measures
                    hover_text = f"{node}<br>"
                    hover_text += (
                        f"Degree Centrality: {degree_centrality.get(node, 0):.3f}<br>"
                    )
                    hover_text += (
                        f"Betweenness: {betweenness_centrality.get(node, 0):.3f}<br>"
                    )
                    hover_text += (
                        f"Closeness: {closeness_centrality.get(node, 0):.3f}<br>"
                    )
                    hover_text += f"Community: {comm_id}"
                    node_hover_text.append(hover_text)

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    hoverinfo="text",
                    text=node_text,
                    hovertext=node_hover_text,
                    textposition="top center",
                    marker=dict(
                        showscale=False,
                        color=node_color,
                        size=node_size,
                        line_width=2,
                        line_color="white",
                    ),
                )

                # Create figure
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        height=500,
                        title="Knowledge Graph Network (with Centrality & Communities)",
                    ),
                )

                st.plotly_chart(fig, width="stretch")

                # Advanced Analytics Section
                st.markdown("**📊 Advanced Graph Analytics:**")

                analytics_col1, analytics_col2, analytics_col3 = st.columns(3)

                with analytics_col1:
                    st.markdown("**🏆 Top Central Tags:**")
                    # Sort by degree centrality
                    top_central = sorted(
                        degree_centrality.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    for tag, centrality in top_central:
                        st.write(f"• {tag}: {centrality:.3f}")

                with analytics_col2:
                    st.markdown("**🌉 Bridge Tags:**")
                    # Sort by betweenness centrality
                    top_between = sorted(
                        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    for tag, centrality in top_between:
                        st.write(f"• {tag}: {centrality:.3f}")

                with analytics_col3:
                    st.markdown("**👥 Communities:**")
                    st.write(
                        f"• {len(set(community_map.values()))} communities detected"
                    )
                    for comm_id in sorted(set(community_map.values())):
                        comm_size = sum(
                            1
                            for node in community_map
                            if community_map[node] == comm_id
                        )
                        st.write(f"• Community {comm_id}: {comm_size} tags")

            else:
                # Fallback for small graphs
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

                # Create edge traces
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )

                # Create node traces
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                node_size = []

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)

                    # Node size based on degree
                    degree = G.degree(node)
                    node_size.append(max(10, min(30, degree * 3)))

                    # Color based on node type (could be extended)
                    node_color.append("#1f77b4")  # Blue for tags

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    hoverinfo="text",
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=False,
                        color=node_color,
                        size=node_size,
                        line_width=2,
                        line_color="white",
                    ),
                )

                # Create figure
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        height=400,
                    ),
                )

                st.plotly_chart(fig, width="stretch")

                # Relationship Details
                st.markdown("**Top Relationships:**")
                rel_df = pd.DataFrame(relationships_data[:10])  # Show top 10
                if not rel_df.empty:
                    st.dataframe(
                        rel_df[["source", "target", "type", "strength", "evidence"]],
                        width="stretch",
                        column_config={
                            "strength": st.column_config.NumberColumn(
                                "Strength", format="%.2f"
                            ),
                            "evidence": st.column_config.NumberColumn("Evidence Count"),
                        },
                    )

        else:
            # No relationships to visualize
            st.info(
                "📊 No tag relationships found. Add more documents with tags to see the knowledge graph visualization."
            )

    except ImportError:
        st.warning("⚠️ NetworkX not available for knowledge graph visualization")
    except Exception as e:
        st.warning(f"Could not load knowledge graph analytics: {e}")

    # System Health Status
    st.markdown("### 🔧 System Health")

    status_col1, status_col2, status_col3, status_col4, status_col5, status_col6 = (
        st.columns(6)
    )

    with status_col1:
        db_connected = metrics.get("database_connected", False)
        status = "✅ Connected" if db_connected else "❌ Offline"
        color = "#28a745" if db_connected else "#dc3545"
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">PostgreSQL DB</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col2:
        search_connected = metrics.get("search_connected", False)
        status = "✅ Connected" if search_connected else "❌ Offline"
        color = "#28a745" if search_connected else "#dc3545"
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">Vector Search</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col3:
        cache_connected = metrics.get("cache_connected", False)
        status = "✅ Connected" if cache_connected else "❌ Offline"
        color = "#28a745" if cache_connected else "#dc3545"
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">Redis Cache</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col4:
        embeddings_exist = metrics.get("embeddings_exist", False)
        total_vectors = metrics.get("total_vectors", 0)
        status = f"✅ {total_vectors:,}" if embeddings_exist else "❌ Empty"
        color = "#28a745" if embeddings_exist else "#dc3545"
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">Embeddings</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col5:
        chapters_exist = metrics.get("chapters_exist", False)
        total_chapters = metrics.get("total_chapters", 0)
        if chapters_exist:
            status = f"✅ {total_chapters}"
            color = "#28a745"
        else:
            status = "ℹ️ Not detected"
            color = "#17a2b8"  # Info blue instead of error red
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">Chapters</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col6:
        ai_ready = metrics.get("ai_ready_count", 0)
        total_docs = metrics.get("total_documents", 0)
        if ai_ready > 0:
            status = f"✅ {ai_ready}/{total_docs}"
            color = "#28a745"
        else:
            status = "⏳ Processing"
            color = "#ffc107"  # Warning yellow
        st.markdown(
            f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.1rem;">{status}</div>
            <div class="metric-label">AI Enriched</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Cache Performance Metrics
    if metrics.get("cache_connected", False):
        st.markdown("### 🚀 Redis Cache Performance")

        cache_col1, cache_col2, cache_col3, cache_col4 = st.columns(4)

        with cache_col1:
            cached_responses = metrics.get("cached_responses", 0)
            st.metric("📦 Cached Items", cached_responses)

        with cache_col2:
            cache_memory = metrics.get("cache_memory", "unknown")
            st.metric("💾 Memory Used", cache_memory)

        with cache_col3:
            hit_rate = metrics.get("cache_hit_rate", 0)
            st.metric("🎯 Hit Rate", f"{hit_rate:.1%}")

        with cache_col4:
            uptime = metrics.get("cache_uptime_days", 0)
            st.metric("⏱️ Uptime", f"{uptime} days")

        # Cache breakdown
        st.markdown("**Cache Contents:**")
        cache_info_col1, cache_info_col2 = st.columns(2)

        with cache_info_col1:
            st.info("• LLM responses (llm:*)\n• Document metadata (doc_meta:*)")

        with cache_info_col2:
            st.info("• Query results\n• Document summaries")

    else:
        st.markdown("### 🚀 Cache Status")
        st.warning("⚠️ Redis cache not available. Check Redis connection in Settings.")

    # Query Performance (if available)
    history = st.session_state.get("query_history", [])
    if history:
        st.markdown("### 📊 Query Performance")

        # Recent query metrics
        recent_queries = history[-10:]  # Last 10 queries
        if recent_queries:
            avg_time = sum(
                item.get("processing_time", 0) for item in recent_queries
            ) / len(recent_queries)
            total_queries = len(history)

            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

            with perf_col1:
                st.metric("Total Queries", total_queries)

            with perf_col2:
                st.metric("Avg Response", f"{avg_time:.2f}s")

            with perf_col3:
                min_time = min(
                    item.get("processing_time", 0) for item in recent_queries
                )
                st.metric("Fastest", f"{min_time:.2f}s")

            with perf_col4:
                max_time = max(
                    item.get("processing_time", 0) for item in recent_queries
                )
                st.metric("Slowest", f"{max_time:.2f}s")

        # Query mode breakdown
        st.markdown("**Query Types:**")
        mode_counts = {}
        for item in history:
            mode = item.get("mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        mode_chart_data = pd.DataFrame(
            {"Mode": list(mode_counts.keys()), "Count": list(mode_counts.values())}
        ).set_index("Mode")

        if not mode_chart_data.empty:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.bar_chart(mode_chart_data)
            st.markdown("</div>", unsafe_allow_html=True)

    # System Performance Insights
    st.markdown("### 📈 System Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.markdown("**Content Efficiency**")
        total_docs = metrics.get("total_documents", 0)
        total_chunks = metrics.get("total_chunks", 0)
        total_chapters = metrics.get("total_chapters", 0)

        if total_docs > 0:
            avg_chunks_per_doc = total_chunks / total_docs
            avg_chapters_per_doc = total_chapters / total_docs

            st.info(f"""
            • {avg_chunks_per_doc:.1f} chunks per document
            • {avg_chapters_per_doc:.1f} chapters per document
            • {total_chunks + total_chapters:,} total searchable units
            """)

    with insights_col2:
        st.markdown("**AI Enrichment Coverage**")
        ai_ready = metrics.get("ai_ready_count", 0)
        total_docs = metrics.get("total_documents", 0)

        if total_docs > 0:
            ai_coverage = (ai_ready / total_docs) * 100
            st.info(f"""
            • {ai_coverage:.1f}% documents AI-enriched
            • {metrics.get("docs_with_summary", 0)} with summaries
            • {metrics.get("docs_with_topics", 0)} with topics
            • {metrics.get("docs_with_reading_time", 0)} with reading time
            """)

    # Recent Queries Table with Feedback
    st.markdown("### 📋 Recent Queries & Feedback")

    history = st.session_state.get("query_history", [])
    if history:
        # Show last 10 queries
        recent_queries = history[-10:]

        # Create a table with feedback options
        table_data = []
        for i, item in enumerate(reversed(recent_queries)):  # Most recent first
            query_id = f"query_{len(history) - i - 1}"  # Unique ID for each query

            # Get existing feedback if any
            feedback_key = f"feedback_{query_id}"
            existing_feedback = st.session_state.get(feedback_key, {})

            table_data.append(
                {
                    "Time": item.get("timestamp", datetime.now()).strftime("%H:%M:%S"),
                    "Mode": item.get("mode", "unknown").title(),
                    "Query": item.get("query", "")[:50]
                    + ("..." if len(item.get("query", "")) > 50 else ""),
                    "Response Time": f"{item.get('processing_time', 0):.2f}s",
                    "Feedback": existing_feedback.get("rating", "Not rated"),
                }
            )

        st.dataframe(table_data, width="stretch")

        # Feedback Collection Section
        st.markdown("**💬 Provide Feedback on Recent Queries:**")

        # Allow feedback on last 5 queries
        feedback_queries = history[-5:]
        if feedback_queries:
            tabs = st.tabs([f"Query {i + 1}" for i in range(len(feedback_queries))])

            for i, (tab, item) in enumerate(zip(tabs, reversed(feedback_queries))):
                with tab:
                    query_id = f"query_{len(history) - i - 1}"
                    feedback_key = f"feedback_{query_id}"

                    st.markdown(f"**Query:** {item.get('query', '')}")
                    st.markdown(f"**Mode:** {item.get('mode', 'unknown').title()}")
                    st.markdown(
                        f"**Response Time:** {item.get('processing_time', 0):.2f}s"
                    )

                    # Get existing feedback
                    existing = st.session_state.get(feedback_key, {})

                    # Rating
                    rating = st.slider(
                        "Rate this response (1-5):",
                        min_value=1,
                        max_value=5,
                        value=existing.get("rating", 3),
                        key=f"rating_{query_id}",
                    )

                    # Comments
                    comments = st.text_area(
                        "Comments (optional):",
                        value=existing.get("comments", ""),
                        height=60,
                        key=f"comments_{query_id}",
                    )

                    # Save feedback
                    if st.button("💾 Save Feedback", key=f"save_{query_id}"):
                        st.session_state[feedback_key] = {
                            "rating": rating,
                            "comments": comments,
                            "timestamp": datetime.now(),
                            "query": item.get("query", ""),
                            "mode": item.get("mode", ""),
                            "response_time": item.get("processing_time", 0),
                        }
                        st.success("✅ Feedback saved!")

                        # Update the table data with new rating
                        st.rerun()

        # Feedback Analytics
        st.markdown("**📊 Feedback Summary:**")
        feedback_stats = []
        for item in history:
            query_id = f"query_{history.index(item)}"
            feedback_key = f"feedback_{query_id}"
            feedback = st.session_state.get(feedback_key, {})
            if feedback.get("rating"):
                feedback_stats.append(
                    {
                        "rating": feedback["rating"],
                        "mode": item.get("mode", "unknown"),
                        "response_time": item.get("processing_time", 0),
                    }
                )

        if feedback_stats:
            feedback_df = pd.DataFrame(feedback_stats)

            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

            with feedback_col1:
                avg_rating = feedback_df["rating"].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5")

            with feedback_col2:
                total_feedback = len(feedback_stats)
                st.metric("Total Feedback", total_feedback)

            with feedback_col3:
                # Rating distribution
                rating_counts = feedback_df["rating"].value_counts().sort_index()
                most_common = rating_counts.idxmax()
                st.metric("Most Common Rating", f"{most_common}/5")

            # Rating distribution chart
            st.markdown("**Rating Distribution:**")
            rating_chart_data = pd.DataFrame(
                {"Rating": rating_counts.index, "Count": rating_counts.values}
            ).set_index("Rating")
            st.bar_chart(rating_chart_data, height=150)

        else:
            st.info("📝 No feedback provided yet. Rate some queries above!")

    else:
        st.info("📭 No queries recorded yet")

    # Export Data
    st.markdown("---")
    st.markdown("### 📥 Export Analytics Data")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.button("📊 Export Query History", width="stretch"):
            history = st.session_state.get("query_history", [])
            if history:
                # Convert to DataFrame for export
                df_data = []
                for item in history:
                    df_data.append(
                        {
                            "timestamp": item.get("timestamp", datetime.now()),
                            "query": item.get("query", ""),
                            "mode": item.get("mode", ""),
                            "processing_time": item.get("processing_time", 0),
                        }
                    )
                df = pd.DataFrame(df_data)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"rag_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("📭 No query history to export")

    with export_col2:
        if st.button("📈 Export System Metrics", width="stretch"):
            # Create comprehensive metrics export
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "documents": {
                        "total": metrics.get("total_documents", 0),
                        "processed": metrics.get("processed_documents", 0),
                        "chunks": metrics.get("total_chunks", 0),
                        "chapters": metrics.get("total_chapters", 0),
                        "embeddings": metrics.get("total_embeddings", 0),
                        "total_size_bytes": metrics.get("total_doc_size", 0),
                    },
                    "ai_enrichment": {
                        "docs_with_summary": metrics.get("docs_with_summary", 0),
                        "docs_with_topics": metrics.get("docs_with_topics", 0),
                        "docs_with_reading_time": metrics.get(
                            "docs_with_reading_time", 0
                        ),
                        "ai_ready_count": metrics.get("ai_ready_count", 0),
                    },
                    "content_breakdown": {
                        "doc_types": metrics.get("doc_types", {}),
                        "languages": metrics.get("languages", {}),
                    },
                    "organization": {
                        "topics": metrics.get("total_topics", 0),
                        "tags": metrics.get("total_tags", 0),
                        "categories": metrics.get("total_categories", 0),
                    },
                    "system_health": {
                        "database_connected": metrics.get("database_connected", False),
                        "search_connected": metrics.get("search_connected", False),
                        "cache_connected": metrics.get("cache_connected", False),
                        "total_vectors": metrics.get("total_vectors", 0),
                    },
                    "cache_performance": {
                        "cached_responses": metrics.get("cached_responses", 0),
                        "cache_memory": metrics.get("cache_memory", "unknown"),
                        "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                        "cache_uptime_days": metrics.get("cache_uptime_days", 0),
                    },
                },
            }

            import json

            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"rag_system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    with export_col3:
        if st.button("📋 Generate Report", width="stretch"):
            # Generate a human-readable report
            report_lines = [
                "# Local RAG System Analytics Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 📊 Core Metrics",
                f"- Total Documents: {metrics.get('total_documents', 0)}",
                f"- Processed Documents: {metrics.get('processed_documents', 0)}",
                f"- Content Chunks: {metrics.get('total_chunks', 0):,}",
                f"- Chapters: {metrics.get('total_chapters', 0)}",
                f"- Vector Embeddings: {metrics.get('total_vectors', 0):,}",
                f"- Total Size: {format_file_size(metrics.get('total_doc_size', 0))}",
                "",
                "## 🤖 AI Enrichment",
                f"- Documents with Summaries: {metrics.get('docs_with_summary', 0)}",
                f"- Documents with Topics: {metrics.get('docs_with_topics', 0)}",
                f"- Documents with Reading Time: {metrics.get('docs_with_reading_time', 0)}",
                f"- AI Ready Documents: {metrics.get('ai_ready_count', 0)}",
                "",
                "## 🔧 System Health",
                f"- Database: {'✅ Connected' if metrics.get('database_connected') else '❌ Offline'}",
                f"- Vector Search: {'✅ Connected' if metrics.get('search_connected') else '❌ Offline'}",
                f"- Redis Cache: {'✅ Connected' if metrics.get('cache_connected') else '❌ Offline'}",
                "",
                "## 🚀 Performance",
                f"- Cached Responses: {metrics.get('cached_responses', 0)}",
                f"- Cache Memory: {metrics.get('cache_memory', 'unknown')}",
                f"- Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}",
            ]

            report_text = "\n".join(report_lines)

            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"rag_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
