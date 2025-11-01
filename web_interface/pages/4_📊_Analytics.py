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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import utilities
from utils.session_manager import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Local RAG - Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

def get_system_metrics():
    """Get current system metrics"""
    metrics = {
        'total_queries': len(st.session_state.get('query_history', [])),
        'system_initialized': st.session_state.get('system_initialized', False),
        'rag_available': st.session_state.get('rag_available', False),
        'current_time': datetime.now()
    }

    # Document metrics
    data_dir = Path("data")
    if data_dir.exists():
        doc_files = list(data_dir.glob("*"))
        supported_ext = ['.txt', '.pdf', '.docx', '.pptx', '.xlsx']
        supported_files = [f for f in doc_files if f.is_file() and f.suffix.lower() in supported_ext]
        metrics['total_documents'] = len(supported_files)
        metrics['total_doc_size'] = sum(f.stat().st_size for f in supported_files)
    else:
        metrics['total_documents'] = 0
        metrics['total_doc_size'] = 0

    # System readiness metrics
    try:
        from src.database.models import SessionLocal, Document, DocumentChunk
        db = SessionLocal()
        # Check if there are any documents in the database
        doc_count = db.query(Document).count()
        chunk_count = db.query(DocumentChunk).count()
        metrics['embeddings_exist'] = doc_count > 0 and chunk_count > 0
        metrics['index_exists'] = chunk_count > 0  # Chunks in DB indicate indexing is done
        metrics['database_connected'] = True
        db.close()
    except Exception:
        # If database is not accessible, assume not ready
        metrics['embeddings_exist'] = False
        metrics['index_exists'] = False
        metrics['database_connected'] = False

    # Check Elasticsearch/OpenSearch connectivity
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False
        )
        metrics['search_connected'] = es.ping()
    except Exception:
        metrics['search_connected'] = False

    return metrics

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def create_query_history_chart():
    """Create a chart of query history over time"""
    history = st.session_state.get('query_history', [])

    if not history:
        return None

    # Create DataFrame from history
    df_data = []
    for item in history:
        df_data.append({
            'timestamp': item.get('timestamp', datetime.now()),
            'mode': item.get('mode', 'unknown'),
            'processing_time': item.get('processing_time', 0)
        })

    df = pd.DataFrame(df_data)

    # Group by hour and mode
    df['hour'] = df['timestamp'].dt.floor('H')
    df_grouped = df.groupby(['hour', 'mode']).size().unstack(fill_value=0)

    return df_grouped

def create_performance_chart():
    """Create a chart of query performance over time"""
    history = st.session_state.get('query_history', [])

    if not history:
        return None

    # Create DataFrame from history
    df_data = []
    for item in history[-20:]:  # Last 20 queries
        df_data.append({
            'timestamp': item.get('timestamp', datetime.now()),
            'processing_time': item.get('processing_time', 0),
            'mode': item.get('mode', 'unknown')
        })

    df = pd.DataFrame(df_data)
    return df

def main():
    """Main page content"""
    # Initialize session state
    initialize_session_state()

    st.markdown('<h1 class="page-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Monitor your Local RAG system performance and usage")

    # Get current metrics
    metrics = get_system_metrics()

    # Overview Metrics
    st.markdown("### 📈 Overview Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_queries']}</div>
            <div class="metric-label">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_documents']}</div>
            <div class="metric-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_time = 0
        history = st.session_state.get('query_history', [])
        if history:
            avg_time = sum(item.get('processing_time', 0) for item in history) / len(history)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.2f}s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        doc_size = format_file_size(metrics['total_doc_size'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{doc_size}</div>
            <div class="metric-label">Total Size</div>
        </div>
        """, unsafe_allow_html=True)

    # System Status
    st.markdown("### 🔧 System Status")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        status = "✅ Online" if metrics['system_initialized'] else "❌ Offline"
        color = "#28a745" if metrics['system_initialized'] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">System Status</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status = "✅ Available" if metrics['rag_available'] else "❌ Offline"
        color = "#28a745" if metrics['rag_available'] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">RAG Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status = "✅ Ready" if metrics['embeddings_exist'] else "❌ No Documents"
        color = "#28a745" if metrics['embeddings_exist'] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">Document Store</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        status = "✅ Ready" if metrics['index_exists'] else "❌ No Chunks"
        color = "#28a745" if metrics['index_exists'] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">Vector Search</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        status = "✅ Connected" if metrics.get('database_connected', False) else "❌ Offline"
        color = "#28a745" if metrics.get('database_connected', False) else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">Database</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        status = "✅ Connected" if metrics.get('search_connected', False) else "❌ Offline"
        color = "#28a745" if metrics.get('search_connected', False) else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.2rem;">{status}</div>
            <div class="metric-label">Search Engine</div>
        </div>
        """, unsafe_allow_html=True)

    # Query History Chart
    st.markdown("### 📊 Query Activity")

    query_chart_data = create_query_history_chart()
    if query_chart_data is not None and not query_chart_data.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.bar_chart(query_chart_data)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📭 No query history available yet")

    # Performance Chart
    st.markdown("### ⚡ Response Time Trends")

    perf_data = create_performance_chart()
    if perf_data is not None and not perf_data.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Create a line chart for response times
        chart_data = perf_data.set_index('timestamp')['processing_time']
        st.line_chart(chart_data)

        st.markdown('</div>', unsafe_allow_html=True)

        # Performance statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{perf_data['processing_time'].mean():.2f}s")
        with col2:
            st.metric("Fastest", f"{perf_data['processing_time'].min():.2f}s")
        with col3:
            st.metric("Slowest", f"{perf_data['processing_time'].max():.2f}s")
    else:
        st.info("📭 No performance data available yet")

    # Recent Queries Table
    st.markdown("### 📋 Recent Queries")

    history = st.session_state.get('query_history', [])
    if history:
        # Show last 10 queries
        recent_queries = history[-10:]

        # Create a table
        table_data = []
        for item in reversed(recent_queries):  # Most recent first
            table_data.append({
                'Time': item.get('timestamp', datetime.now()).strftime('%H:%M:%S'),
                'Mode': item.get('mode', 'unknown').title(),
                'Query': item.get('query', '')[:50] + ('...' if len(item.get('query', '')) > 50 else ''),
                'Response Time': f"{item.get('processing_time', 0):.2f}s"
            })

        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("📭 No queries recorded yet")

    # Export Data
    st.markdown("---")
    st.markdown("### 📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export Query History", use_container_width=True):
            history = st.session_state.get('query_history', [])
            if history:
                # Convert to DataFrame for export
                df_data = []
                for item in history:
                    df_data.append({
                        'timestamp': item.get('timestamp', datetime.now()),
                        'query': item.get('query', ''),
                        'mode': item.get('mode', ''),
                        'processing_time': item.get('processing_time', 0)
                    })
                df = pd.DataFrame(df_data)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rag_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No query history to export")

    with col2:
        if st.button("📈 Export Metrics", use_container_width=True):
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            st.download_button(
                label="Download JSON",
                data=str(metrics_data).replace("'", '"'),
                file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()