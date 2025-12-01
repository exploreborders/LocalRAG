"""
Results display components for the Local RAG Web Interface
"""

import streamlit as st


def render_results():
    """Render query results"""
    results = st.session_state.get("current_results")

    if not results:
        return

    st.markdown("### 📋 Results")

    # Display processing time
    processing_time = st.session_state.get("processing_time", 0)
    if processing_time > 0:
        st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")

    # Display results based on type
    if results.get("type") == "topic-aware":
        render_topic_aware_results(results)
    elif results.get("type") == "rag":
        render_rag_results(results)
    elif results.get("type") == "advanced":
        render_advanced_search_results(results)
    else:
        # Default to RAG rendering for backward compatibility
        render_rag_results(results)

    # Render search suggestions for all result types
    render_search_suggestions()


def render_rag_results(results):
    """Render RAG (generation) results"""
    # Display query language if detected
    result_data = results.get("result", {})
    query_lang = result_data.get("query_language", "unknown")
    if query_lang != "unknown":
        lang_names = {
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
        lang_display = lang_names.get(query_lang, (query_lang or "unknown").upper())
        st.info(f"🌍 Detected query language: {lang_display}")

    st.markdown("**🤖 AI-Generated Answer:**")

    # Display the generated answer
    formatted_answer = results.get("formatted", "")
    if formatted_answer:
        st.markdown(formatted_answer)
    else:
        st.warning("No formatted answer available")

    # Show processing information
    processing_info = results.get("processing_info", {})
    if processing_info.get("ocr_used") or processing_info.get("is_scanned"):
        if processing_info.get("ocr_used"):
            st.info(
                "🤖 **OCR Processing Used**: This document was processed using Optical Character Recognition for scanned content."
            )
        elif processing_info.get("is_scanned"):
            st.info(
                "📄 **Scanned Document Detected**: This appears to be a scanned PDF with image-based content."
            )

    # Show quality information if available
    result_data = results.get("result", {})
    quality_issues = result_data.get("quality_issues", [])
    quality_score = result_data.get("quality_score")
    quality_recommendations = result_data.get("quality_recommendations", [])

    if quality_issues:
        st.warning("⚠️ **Content Quality Issues Detected**")
        for issue in quality_issues:
            st.write(f"• {issue}")

        if quality_score is not None:
            # Show quality score with color coding
            if quality_score >= 0.7:
                st.success(f"✅ Quality Score: {quality_score:.2f} (Good)")
            elif quality_score >= 0.4:
                st.warning(f"⚠️ Quality Score: {quality_score:.2f} (Needs Review)")
            else:
                st.error(f"❌ Quality Score: {quality_score:.2f} (Poor Quality)")

        if quality_recommendations:
            st.info("💡 **Recommended Actions:**")
            for rec in quality_recommendations:
                st.write(f"• {rec}")

            # Add reprocessing button if quality is poor
            if quality_score is not None and quality_score < 0.5:
                st.button(
                    "🔄 Reprocess Document",
                    key="reprocess_quality_issue",
                    help="Reprocess this document with improved extraction methods",
                )

    # Show summary statistics
    sources = result_data.get("sources", [])
    if sources:
        total_sources = len(set(source.get("document_id") for source in sources))
        total_chunks = len(sources)

        # Count unique chapters
        chapters_used = set()
        for source in sources:
            chapter_title = source.get("chapter_title")
            chapter_path = source.get("chapter_path")
            if chapter_title:
                chapters_used.add((chapter_title, chapter_path))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Sources", total_sources)
        with col2:
            st.metric("📖 Chapters", len(chapters_used))
        with col3:
            st.metric("📝 Chunks", total_chunks)

    # Display source documents and chapters used for generation
    result_data = results.get("result", {})
    sources = result_data.get("sources", [])

    if sources:
        st.markdown("**📚 Sources Used for Answer Generation:**")

        # Group by document to show unique sources
        doc_sources = {}
        for source in sources:
            doc_id = source.get("document_id")
            title = source.get("title", "Unknown")
            if doc_id not in doc_sources:
                doc_sources[doc_id] = {
                    "title": title,
                    "document_id": doc_id,
                    "chunks": [],
                }
            doc_sources[doc_id]["chunks"].append(source)

        # Display each unique document
        for i, (doc_id, doc_data) in enumerate(doc_sources.items(), 1):
            with st.expander(f"📄 Source {i}: {doc_data['title']}"):
                st.markdown(f"**Document:** {doc_data['title']}")
                st.markdown(f"**Document ID:** {doc_id}")
                st.markdown(f"**Chunks used:** {len(doc_data['chunks'])}")

                # Show relevance scores
                scores = [chunk.get("score", 0) for chunk in doc_data["chunks"]]
                avg_score = sum(scores) / len(scores) if scores else 0
                st.markdown(f"**Average relevance score:** {avg_score:.3f}")

                # Show chapters used
                chapters_used = set()
                for chunk in doc_data["chunks"]:
                    chapter_title = chunk.get("chapter_title")
                    chapter_path = chunk.get("chapter_path")
                    if chapter_title:
                        chapters_used.add((chapter_title, chapter_path))

                if chapters_used:
                    st.markdown("**📖 Chapters Referenced:**")
                    for chapter_title, chapter_path in sorted(chapters_used):
                        st.markdown(f"• **{chapter_title}** (Path: {chapter_path})")

                # Show tags and categories
                all_tags = set()
                all_categories = set()
                for chunk in doc_data["chunks"]:
                    all_tags.update(chunk.get("tags", []))
                    all_categories.update(chunk.get("categories", []))

                if all_tags:
                    st.markdown(f"**🏷️ Tags:** {', '.join(sorted(all_tags))}")

                if all_categories:
                    st.markdown(
                        f"**📂 Categories:** {', '.join(sorted(all_categories))}"
                    )

                # Show content preview from most relevant chunk
                if doc_data["chunks"]:
                    best_chunk = max(
                        doc_data["chunks"], key=lambda x: x.get("score", 0)
                    )
                    content_preview = best_chunk.get("content_preview", "")
                    if content_preview:
                        st.markdown("**📄 Most Relevant Content:**")
                        st.write(content_preview)


def render_topic_aware_results(results):
    """Render smart search results with topic relevance indicators"""
    st.markdown("**🎯 Smart Search Results:**")

    # Display formatted results
    formatted_results = results.get("formatted", "")
    if formatted_results:
        st.code(formatted_results, language=None)
    else:
        st.warning("No formatted results available")

    # Display individual results with topic boost information
    raw_results = results.get("results", [])
    if raw_results:
        st.markdown("**Document Details with Topic Relevance:**")
        for i, result in enumerate(raw_results, 1):
            doc = result.get("document", {})
            score = result.get("score", 0)
            topic_boost = result.get("topic_boost", 0)
            matching_topics = result.get("matching_topics", [])

            # Create expander title with topic boost indicator
            boost_indicator = (
                "🔥" if topic_boost > 0.5 else "⭐" if topic_boost > 0.2 else "📄"
            )
            boost_text = f" (Topic Boost: {topic_boost:.2f})" if topic_boost > 0 else ""

            with st.expander(
                f"{boost_indicator} Document {i} (Score: {score:.4f}){boost_text}"
            ):
                page_content = (
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                )
                metadata = doc.metadata if hasattr(doc, "metadata") else {}

                # Content preview
                st.markdown("**Content:**")
                content_preview = (
                    page_content[:500] + "..."
                    if len(page_content) > 500
                    else page_content
                )
                st.write(content_preview)

                # Topic relevance information
                if topic_boost > 0:
                    st.markdown("**🎯 Topic Relevance:**")
                    st.progress(min(topic_boost, 1.0))  # Cap at 1.0 for progress bar
                    st.write(f"**Boost Value:** {topic_boost:.2f}")

                    if matching_topics:
                        st.markdown("**Matching Topics:**")
                        for topic in matching_topics:
                            st.write(f"• {topic}")
                else:
                    st.info("No significant topic matches found for this document")

                # Metadata
                if metadata:
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"- **{key}:** {value}")


def render_advanced_search_results(results):
    """Render advanced search results with analytics and hybrid scoring"""
    result_data = results.get("result", {})

    # Check if this is an advanced search result
    if "analytics" in result_data:
        render_advanced_search_analytics(result_data)
        render_hybrid_search_results(result_data)
    else:
        # Fall back to regular rendering
        render_rag_results(results)


def render_advanced_search_analytics(result_data):
    """Render search analytics and performance metrics"""
    analytics = result_data.get("analytics", {})

    with st.expander("📊 Search Analytics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            search_time = analytics.get("search_time", 0)
            st.metric("Search Time", f"{search_time:.3f}s")

        with col2:
            total_results = analytics.get("total_results", 0)
            st.metric("Total Results", total_results)

        with col3:
            query_complexity = analytics.get("query_complexity", 0)
            complexity_label = (
                "Low"
                if query_complexity < 2
                else "Medium"
                if query_complexity < 4
                else "High"
            )
            st.metric("Query Complexity", complexity_label)

        with col4:
            avg_score = analytics.get("performance_metrics", {}).get("avg_score", 0)
            st.metric("Avg Relevance", f"{avg_score:.3f}")

        # Result distribution
        distribution = analytics.get("result_distribution", {})
        if distribution:
            st.markdown("**Result Distribution:**")
            dist_col1, dist_col2, dist_col3 = st.columns(3)

            with dist_col1:
                unique_docs = distribution.get("unique_documents", 0)
                st.metric("Unique Documents", unique_docs)

            with dist_col2:
                multi_chunk = distribution.get("documents_with_multiple_chunks", 0)
                st.metric("Multi-chunk Docs", multi_chunk)

            with dist_col3:
                chapters_used = len(set())  # Would need to calculate from results
                st.metric("Chapters Used", chapters_used)

        # Top tags and categories
        top_tags = distribution.get("top_tags", {})
        top_cats = distribution.get("top_categories", {})

        if top_tags or top_cats:
            st.markdown("**Popular Tags & Categories in Results:**")
            tag_cat_col1, tag_cat_col2 = st.columns(2)

            with tag_cat_col1:
                if top_tags:
                    st.markdown("**Top Tags:**")
                    for tag, count in list(top_tags.items())[:5]:
                        st.write(f"• {tag} ({count})")

            with tag_cat_col2:
                if top_cats:
                    st.markdown("**Top Categories:**")
                    for cat, count in list(top_cats.items())[:5]:
                        st.write(f"• {cat} ({count})")


def render_hybrid_search_results(result_data):
    """Render hybrid search results with vector/BM25 breakdown"""
    results = result_data.get("results", [])

    if not results:
        st.info("No results found for your search.")
        return

    st.markdown("### 🔍 Advanced Search Results")

    # Results summary
    total_results = len(results)
    st.info(f"Found {total_results} relevant results")

    # Display results
    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        vector_score = result.get("vector_score", 0)
        bm25_score = result.get("bm25_score", 0)
        document_title = result.get("document_title", "Unknown")
        content = result.get("content", "")
        tags = result.get("tags", [])
        categories = result.get("categories", [])

        # Score indicator
        score_indicator = "🔥" if score > 0.8 else "⭐" if score > 0.6 else "📄"

        with st.expander(
            f"{score_indicator} Result {i}: {document_title} (Score: {score:.3f})"
        ):
            # Score breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Score", f"{score:.3f}")
            with col2:
                st.metric("Vector Similarity", f"{vector_score:.3f}")
            with col3:
                st.metric("Keyword Match", f"{bm25_score:.3f}")

            # Content preview
            st.markdown("**Content Preview:**")
            content_preview = content[:300] + "..." if len(content) > 300 else content
            st.write(content_preview)

            # Tags and categories
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                if tags:
                    st.markdown(f"**🏷️ Tags:** {', '.join(tags)}")
            with meta_col2:
                if categories:
                    st.markdown(f"**📂 Categories:** {', '.join(categories)}")

            # Metadata
            metadata = result.get("metadata", {})
            if metadata:
                st.markdown("**Additional Info:**")
                for key, value in metadata.items():
                    if key in ["chapter_title", "chapter_path", "author"]:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")


def render_search_suggestions():
    """Render search suggestions and query expansion"""
    if "current_results" not in st.session_state:
        return

    results = st.session_state.current_results
    if not results or results.get("type") != "advanced":
        return

    result_data = results.get("result", {})
    expanded_query = result_data.get("expanded_query", {})

    if not expanded_query:
        return

    with st.expander("💡 Search Insights", expanded=False):
        # Query expansion suggestions
        expanded_tags = expanded_query.get("expanded_tags", [])
        expanded_categories = expanded_query.get("expanded_categories", [])

        if expanded_tags or expanded_categories:
            st.markdown("**🔗 Query Expansion:**")
            st.write("Your search was expanded using knowledge graph relationships:")

            if expanded_tags:
                st.write(f"• **Related Tags:** {', '.join(expanded_tags)}")
            if expanded_categories:
                st.write(f"• **Related Categories:** {', '.join(expanded_categories)}")

        # Parsed query information
        parsed_query = result_data.get("parsed_query", {})
        if parsed_query:
            st.markdown("**🔍 Query Analysis:**")

            if parsed_query.get("phrase_terms"):
                st.write(
                    f"• **Phrase searches:** {', '.join(f'"{p}"' for p in parsed_query['phrase_terms'])}"
                )

            if parsed_query.get("field_queries"):
                st.write("• **Field-specific searches:**")
                for field, values in parsed_query["field_queries"].items():
                    st.write(f"  - {field}: {', '.join(values)}")

            if parsed_query.get("excluded_terms"):
                st.write(
                    f"• **Excluded terms:** {', '.join(parsed_query['excluded_terms'])}"
                )

        # Performance suggestions
        analytics = result_data.get("analytics", {})
        search_time = analytics.get("search_time", 0)

        if search_time > 2.0:
            st.warning(
                "⚠️ Search took longer than usual. Consider simplifying your query or using more specific filters."
            )

        query_complexity = analytics.get("query_complexity", 0)
        if query_complexity > 3.0:
            st.info(
                "💡 Your query is quite complex. Try breaking it into simpler parts for faster results."
            )


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
    status = st.session_state.get("system_status", {})

    col1, col2 = st.sidebar.columns(2)
    with col1:
        retriever_status = (
            "✅ Active" if status.get("retriever_active") else "❌ Offline"
        )
        st.metric("Retriever", retriever_status)

    with col2:
        rag_status = "✅ Active" if status.get("rag_available") else "❌ Offline"
        st.metric("RAG Pipeline", rag_status)

    # Query statistics
    total_queries = len(st.session_state.get("query_history", []))
    st.sidebar.metric("Total Queries", total_queries)

    # Average processing time
    history = st.session_state.get("query_history", [])
    if history:
        avg_time = sum(item.get("processing_time", 0) for item in history) / len(
            history
        )
        st.sidebar.metric("Avg Response Time", f"{avg_time:.2f}s")
