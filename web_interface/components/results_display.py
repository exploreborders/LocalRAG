"""
Results display components for the Local RAG Web Interface
"""

import streamlit as st

def render_results():
    """Render query results"""
    results = st.session_state.get('current_results')

    if not results:
        return

    st.markdown("### 📋 Results")

    # Display processing time
    processing_time = st.session_state.get('processing_time', 0)
    if processing_time > 0:
        st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")

    # Display results based on type
    if results.get('type') == 'retrieval':
        render_retrieval_results(results)
    elif results.get('type') == 'rag':
        render_rag_results(results)

def render_retrieval_results(results):
    """Render retrieval-only results"""
    st.markdown("**📄 Retrieved Documents:**")

    # Display formatted results
    formatted_results = results.get('formatted', '')
    if formatted_results:
        st.code(formatted_results, language=None)
    else:
        st.warning("No formatted results available")

    # Display individual results with expanders
    raw_results = results.get('results', [])
    if raw_results:
        st.markdown("**Document Details:**")
        for i, result in enumerate(raw_results, 1):
            doc = result.get('document', {})
            distance = result.get('distance', 0)

            with st.expander(f"📄 Document {i} (Distance: {distance:.4f})"):
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                # Content preview
                st.markdown("**Content:**")
                content_preview = page_content[:500] + "..." if len(page_content) > 500 else page_content
                st.write(content_preview)

                # Metadata
                if metadata:
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"- **{key}:** {value}")

def render_rag_results(results):
    """Render RAG (generation) results"""
    st.markdown("**🤖 AI-Generated Answer:**")

    # Display the generated answer
    formatted_answer = results.get('formatted', '')
    if formatted_answer:
        st.markdown(formatted_answer)
    else:
        st.warning("No formatted answer available")

    # Display source documents used
    result_data = results.get('result', {})
    if 'source_documents' in result_data:
        st.markdown("**📚 Source Documents:**")

        source_docs = result_data['source_documents']
        for i, doc in enumerate(source_docs, 1):
            with st.expander(f"📄 Source {i}"):
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                # Content preview
                content_preview = page_content[:300] + "..." if len(page_content) > 300 else page_content
                st.write(content_preview)

                # Metadata
                if metadata:
                    st.markdown("*Metadata:*")
                    for key, value in metadata.items():
                        st.caption(f"{key}: {value}")

def render_no_results():
    """Render message when no results are available"""
    st.info("🔍 Enter a query above to get started!")

def render_error_message(error_msg):
    """Render error message"""
    st.error(f"❌ Error: {error_msg}")

def render_performance_metrics():
    """Render performance metrics"""
    st.sidebar.markdown("### 📊 Performance")

    # System status
    status = st.session_state.get('system_status', {})

    col1, col2 = st.sidebar.columns(2)
    with col1:
        retriever_status = "✅ Active" if status.get('retriever_active') else "❌ Offline"
        st.metric("Retriever", retriever_status)

    with col2:
        rag_status = "✅ Active" if status.get('rag_available') else "❌ Offline"
        st.metric("RAG Pipeline", rag_status)

    # Query statistics
    total_queries = len(st.session_state.get('query_history', []))
    st.sidebar.metric("Total Queries", total_queries)

    # Average processing time
    history = st.session_state.get('query_history', [])
    if history:
        avg_time = sum(item.get('processing_time', 0) for item in history) / len(history)
        st.sidebar.metric("Avg Response Time", f"{avg_time:.2f}s")