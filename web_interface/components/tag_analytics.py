"""
Tag analytics and insights component for the web interface.

Provides comprehensive tag usage statistics, trends, and recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def render_tag_analytics():
    """
    Render comprehensive tag analytics dashboard.
    """
    try:
        from src.document_managers import TagManager
        from src.database.models import SessionLocal

        db = SessionLocal()
        tag_manager = TagManager(db)

        # Get tag statistics
        tag_stats = tag_manager.get_tag_usage_stats()

        if not tag_stats:
            st.info("No tags found. Start tagging documents to see analytics.")
            db.close()
            return

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(tag_stats)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tags", len(df))

        with col2:
            total_usage = df['document_count'].sum()
            st.metric("Total Tag Assignments", int(total_usage))

        with col3:
            avg_usage = df['document_count'].mean()
            st.metric("Avg Tags per Document", f"{avg_usage:.1f}")

        with col4:
            most_used = df.loc[df['document_count'].idxmax()]
            st.metric("Most Used Tag", most_used['name'])

        st.divider()

        # Tag usage visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Tag Usage Distribution")

            # Filter out tags with zero usage for cleaner chart
            active_tags = df[df['document_count'] > 0].copy()
            active_tags = active_tags.sort_values('document_count', ascending=True)

            if not active_tags.empty:
                fig = px.bar(
                    active_tags,
                    y='name',
                    x='document_count',
                    orientation='h',
                    title="Documents per Tag",
                    labels={'name': 'Tag', 'document_count': 'Document Count'},
                    color='document_count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active tags found.")

        with col2:
            st.subheader("🎨 Color Distribution")

            # Count tags by color
            color_counts = df['color'].value_counts().reset_index()
            color_counts.columns = ['color', 'count']

            if not color_counts.empty:
                # Create color swatches
                fig = go.Figure()

                for _, row in color_counts.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['count']],
                        y=[row['color']],
                        orientation='h',
                        marker_color=row['color'],
                        name=row['color'],
                        showlegend=False
                    ))

                fig.update_layout(
                    title="Tags by Color",
                    xaxis_title="Number of Tags",
                    yaxis_title="Color",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No color data available.")

        st.divider()

        # Tag insights and recommendations
        st.subheader("💡 Tag Insights & Recommendations")

        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown("### 📈 Popular Tags")
            popular_tags = tag_manager.get_popular_tags(limit=5)

            if popular_tags:
                for tag in popular_tags:
                    color = tag.get('color', '#6c757d')
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                        f"<div style='background-color: {color}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;'></div>"
                        f"<span><strong>{tag['name']}</strong> - {tag['document_count']} documents</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No popular tags data available.")

        with insights_col2:
            st.markdown("### 🔗 Tag Relationships")

            # Show co-occurring tags for the most popular tag
            if popular_tags:
                top_tag = popular_tags[0]
                related_tags = tag_manager.get_related_tags(top_tag['name'], limit=3)

                if related_tags:
                    st.markdown(f"**Tags often used with '{top_tag['name']}'**")
                    for related in related_tags:
                        color = related.get('color', '#6c757d')
                        st.markdown(
                            f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                            f"<div style='background-color: {color}; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;'></div>"
                            f"<span>{related['name']} ({related['co_occurrence']} co-occurrences)</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No relationship data available.")
            else:
                st.info("No relationship data available.")

        st.divider()

        # Tag management recommendations
        st.subheader("🛠️ Tag Management Recommendations")

        recommendations = []

        # Check for unused tags
        unused_tags = df[df['document_count'] == 0]
        if len(unused_tags) > 0:
            recommendations.append({
                'type': 'cleanup',
                'message': f"Found {len(unused_tags)} unused tags that could be removed",
                'action': 'Review and clean up unused tags'
            })

        # Check for overused tags
        avg_usage = df['document_count'].mean()
        overused_tags = df[df['document_count'] > avg_usage * 3]
        if len(overused_tags) > 0:
            recommendations.append({
                'type': 'refinement',
                'message': f"Found {len(overused_tags)} tags used much more than average",
                'action': 'Consider splitting these tags into more specific categories'
            })

        # Check for color conflicts
        color_groups = df.groupby('color').size()
        conflicting_colors = color_groups[color_groups > 3]
        if len(conflicting_colors) > 0:
            recommendations.append({
                'type': 'visual',
                'message': f"Some colors are used for {conflicting_colors.max()} or more tags",
                'action': 'Consider using different colors for better visual distinction'
            })

        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'cleanup':
                    st.warning(f"🧹 {rec['message']}\n\n*{rec['action']}*")
                elif rec['type'] == 'refinement':
                    st.info(f"🔄 {rec['message']}\n\n*{rec['action']}*")
                elif rec['type'] == 'visual':
                    st.info(f"🎨 {rec['message']}\n\n*{rec['action']}*")
        else:
            st.success("✅ Your tag system looks well-organized!")

        # Raw data table (collapsible)
        with st.expander("📋 Raw Tag Data", expanded=False):
            st.dataframe(
                df[['name', 'color', 'document_count', 'usage_count', 'created_at']],
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error loading tag analytics: {e}")
        logger.error(f"Tag analytics error: {e}")
    finally:
        if 'db' in locals():
            db.close()
    """
    Render comprehensive tag analytics dashboard.
    """
    st.header("🏷️ Tag Analytics & Insights")

    try:
        db = SessionLocal()
        tag_manager = TagManager(db)

        # Get tag statistics
        tag_stats = tag_manager.get_tag_usage_stats()

        if not tag_stats:
            st.info("No tags found. Start tagging documents to see analytics.")
            db.close()
            return

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(tag_stats)

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tags", len(df))

        with col2:
            total_usage = df['document_count'].sum()
            st.metric("Total Tag Assignments", int(total_usage))

        with col3:
            avg_usage = df['document_count'].mean()
            st.metric("Avg Tags per Document", f"{avg_usage:.1f}")

        with col4:
            most_used = df.loc[df['document_count'].idxmax()]
            st.metric("Most Used Tag", most_used['name'])

        st.divider()

        # Tag usage visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Tag Usage Distribution")

            # Filter out tags with zero usage for cleaner chart
            active_tags = df[df['document_count'] > 0].copy()
            active_tags = active_tags.sort_values('document_count', ascending=True)

            if not active_tags.empty:
                fig = px.bar(
                    active_tags,
                    y='name',
                    x='document_count',
                    orientation='h',
                    title="Documents per Tag",
                    labels={'name': 'Tag', 'document_count': 'Document Count'},
                    color='document_count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active tags found.")

        with col2:
            st.subheader("🎨 Color Distribution")

            # Count tags by color
            color_counts = df['color'].value_counts().reset_index()
            color_counts.columns = ['color', 'count']

            if not color_counts.empty:
                # Create color swatches
                fig = go.Figure()

                for _, row in color_counts.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['count']],
                        y=[row['color']],
                        orientation='h',
                        marker_color=row['color'],
                        name=row['color'],
                        showlegend=False
                    ))

                fig.update_layout(
                    title="Tags by Color",
                    xaxis_title="Number of Tags",
                    yaxis_title="Color",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No color data available.")

        st.divider()

        # Tag insights and recommendations
        st.subheader("💡 Tag Insights & Recommendations")

        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown("### 📈 Popular Tags")
            popular_tags = tag_manager.get_popular_tags(limit=5)

            if popular_tags:
                for tag in popular_tags:
                    color = tag.get('color', '#6c757d')
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                        f"<div style='background-color: {color}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;'></div>"
                        f"<span><strong>{tag['name']}</strong> - {tag['document_count']} documents</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No popular tags data available.")

        with insights_col2:
            st.markdown("### 🔗 Tag Relationships")

            # Show co-occurring tags for the most popular tag
            if popular_tags:
                top_tag = popular_tags[0]
                related_tags = tag_manager.get_related_tags(top_tag['name'], limit=3)

                if related_tags:
                    st.markdown(f"**Tags often used with '{top_tag['name']}'**")
                    for related in related_tags:
                        color = related.get('color', '#6c757d')
                        st.markdown(
                            f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                            f"<div style='background-color: {color}; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;'></div>"
                            f"<span>{related['name']} ({related['co_occurrence']} co-occurrences)</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No relationship data available.")
            else:
                st.info("No relationship data available.")

        st.divider()

        # Tag management recommendations
        st.subheader("🛠️ Tag Management Recommendations")

        recommendations = []

        # Check for unused tags
        unused_tags = df[df['document_count'] == 0]
        if len(unused_tags) > 0:
            recommendations.append({
                'type': 'cleanup',
                'message': f"Found {len(unused_tags)} unused tags that could be removed",
                'action': 'Review and clean up unused tags'
            })

        # Check for overused tags
        avg_usage = df['document_count'].mean()
        overused_tags = df[df['document_count'] > avg_usage * 3]
        if len(overused_tags) > 0:
            recommendations.append({
                'type': 'refinement',
                'message': f"Found {len(overused_tags)} tags used much more than average",
                'action': 'Consider splitting these tags into more specific categories'
            })

        # Check for color conflicts
        color_groups = df.groupby('color').size()
        conflicting_colors = color_groups[color_groups > 3]
        if len(conflicting_colors) > 0:
            recommendations.append({
                'type': 'visual',
                'message': f"Some colors are used for {conflicting_colors.max()} or more tags",
                'action': 'Consider using different colors for better visual distinction'
            })

        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'cleanup':
                    st.warning(f"🧹 {rec['message']}\n\n*{rec['action']}*")
                elif rec['type'] == 'refinement':
                    st.info(f"🔄 {rec['message']}\n\n*{rec['action']}*")
                elif rec['type'] == 'visual':
                    st.info(f"🎨 {rec['message']}\n\n*{rec['action']}*")
        else:
            st.success("✅ Your tag system looks well-organized!")

        # Raw data table (collapsible)
        with st.expander("📋 Raw Tag Data", expanded=False):
            st.dataframe(
                df[['name', 'color', 'document_count', 'usage_count', 'created_at']],
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error loading tag analytics: {e}")
        logger.error(f"Tag analytics error: {e}")
    finally:
        if 'db' in locals():
            db.close()


def render_tag_suggestions(document_id: int, document_content: str, document_title: str = ""):
    """
    Render AI-powered tag suggestions for a specific document.

    Args:
        document_id: Document ID
        document_content: Document content for analysis
        document_title: Document title/filename
    """
    st.subheader("🤖 AI Tag Suggestions")

    try:
        from src.document_managers import TagManager
        from src.database.models import SessionLocal, Document, DocumentTagAssignment

        db = SessionLocal()
        tag_manager = TagManager(db)

        # Get existing tags for this document
        existing_tags = tag_manager.get_document_tags(document_id)
        existing_tag_names = [tag.name for tag in existing_tags]

        # Generate suggestions
        with st.spinner("Analyzing document content..."):
            suggestions = tag_manager.suggest_tags_for_document(document_id, max_suggestions=6)

        if suggestions:
            st.success(f"Generated {len(suggestions)} tag suggestions")

            # Display suggestions in a grid
            cols = st.columns(min(3, len(suggestions)))

            for i, suggestion in enumerate(suggestions):
                with cols[i % 3]:
                    confidence = suggestion.get('confidence', 0.5)
                    tag_name = suggestion['tag']

                    # Color based on confidence
                    if confidence >= 0.8:
                        color = "#28a745"  # Green for high confidence
                        confidence_label = "High"
                    elif confidence >= 0.6:
                        color = "#ffc107"  # Yellow for medium confidence
                        confidence_label = "Medium"
                    else:
                        color = "#6c757d"  # Gray for low confidence
                        confidence_label = "Low"

                    # Check if tag already exists on document
                    already_assigned = tag_name.lower() in [t.lower() for t in existing_tag_names]

                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid {color};
                            border-radius: 8px;
                            padding: 10px;
                            margin: 5px 0;
                            background-color: {'#f8f9fa' if already_assigned else 'white'};
                        ">
                            <div style="font-weight: bold; color: {color};">{tag_name}</div>
                            <div style="font-size: 0.8em; color: #666;">
                                {confidence_label} confidence ({confidence:.1%})
                                {' ✓ Already assigned' if already_assigned else ''}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if not already_assigned:
                            if st.button(
                                "➕ Add",
                                key=f"add_suggestion_{document_id}_{i}",
                                use_container_width=True,
                                help=f"Add '{tag_name}' to document"
                            ):
                                # Create tag if it doesn't exist
                                tag = tag_manager.get_tag_by_name(tag_name)
                                if not tag:
                                    tag = tag_manager.create_tag_with_ai_color(tag_name)

                                if tag and tag_manager.add_tag_to_document(document_id, tag.id):
                                    st.success(f"✅ Added '{tag_name}'")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to add '{tag_name}'")

                    with col2:
                        if st.button(
                            "👁️ Preview",
                            key=f"preview_suggestion_{document_id}_{i}",
                            use_container_width=True,
                            help=f"See documents with '{tag_name}' tag"
                        ):
                            # Show existing documents with this tag
                            tag = tag_manager.get_tag_by_name(tag_name)
                            if tag:
                                tag_docs = db.query(Document).join(DocumentTagAssignment).filter(
                                    DocumentTagAssignment.tag_id == tag.id
                                ).limit(5).all()

                                if tag_docs:
                                    st.info(f"📄 Documents with '{tag_name}':")
                                    for doc in tag_docs:
                                        st.write(f"• {doc.filename}")
                                else:
                                    st.info(f"📄 No documents currently tagged with '{tag_name}'")
                            else:
                                st.info(f"📄 '{tag_name}' is a new tag")

            # Auto-assign high confidence tags
            high_confidence = [s for s in suggestions if s.get('confidence', 0) >= 0.8]
            if high_confidence and not existing_tags:  # Only suggest auto-assign for untagged documents
                st.divider()
                st.markdown("### ⚡ Quick Auto-Assign")
                st.markdown("Automatically assign high-confidence tags to this document:")

                if st.button("🚀 Auto-assign high confidence tags", use_container_width=True):
                    assigned = tag_manager.auto_assign_tags(document_id, min_confidence=0.8)
                    if assigned:
                        st.success(f"✅ Auto-assigned {len(assigned)} tags: {', '.join(assigned)}")
                        st.rerun()
                    else:
                        st.warning("No high-confidence tags were assigned")

        else:
            st.info("🤔 No tag suggestions generated. The document might be too short or contain limited distinctive content.")

    except Exception as e:
        st.error(f"❌ Error generating tag suggestions: {e}")
        logger.error(f"Tag suggestions error: {e}")
    finally:
        if 'db' in locals():
            db.close()