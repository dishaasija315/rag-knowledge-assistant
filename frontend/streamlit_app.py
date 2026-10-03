"""Streamlit Frontend Application for RAG Knowledge Assistant."""
import time
import streamlit as st

from frontend.api_client import APIClient

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .citation-meta {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 4px;
    }
    .refusal-card {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 4px;
        color: #991B1B;
        font-weight: 500;
    }
    .conflict-card {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> APIClient:
    return APIClient()


client = get_client()

# ------------------------------------------------------------------------------
# Sidebar: System Status & Document Management
# ------------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Grounded Document Intelligence")

    # Health Check Indicator
    health = client.get_health()
    if health.get("status") == "healthy":
        st.success("● Backend & Qdrant Online", icon="🟢")
    else:
        st.error("● Backend Disconnected", icon="🔴")
        st.info("Run `uvicorn app.main:app --port 8000` to start the backend.")

    st.divider()

    # Upload PDF
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("📥 Index Document", use_container_width=True, type="primary"):
            with st.spinner(f"Parsing & indexing '{uploaded_file.name}'..."):
                try:
                    file_bytes = uploaded_file.read()
                    res = client.upload_pdf(uploaded_file.name, file_bytes)
                    st.success(f"Indexed {res.get('chunks_indexed', 0)} chunks from {res.get('total_pages', 0)} pages!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Document Library
    st.subheader("📑 Document Library")
    docs = client.list_documents()

    if not docs:
        st.caption("No documents indexed yet.")
    else:
        for d in docs:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{d['filename']}**")
                    st.caption(f"{d.get('total_pages', 0)} pages | {d.get('chunk_count', 0)} chunks")
                with col2:
                    if st.button("🗑️", key=f"del_{d['doc_id']}", help="Delete document"):
                        if client.delete_document(d["doc_id"]):
                            st.toast("Document deleted successfully")
                            st.rerun()

    st.divider()
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=4)

# ------------------------------------------------------------------------------
# Main Application Content
# ------------------------------------------------------------------------------

st.markdown('<div class="main-header">RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strictly grounded multi-document Q&A, version comparison, and contradiction detection.</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "💬 Document Q&A",
    "⚖️ Document Comparison",
    "⚠️ Contradiction Detection",
    "📝 Targeted Summarization",
    "📊 System Evaluation",
])

# ------------------------------------------------------------------------------
# Tab 1: Document Q&A
# ------------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Ask Questions About Your Documents")
    
    # Optional Document Scope Selector
    doc_options = {d["filename"]: d["doc_id"] for d in docs}
    selected_doc_names = st.multiselect(
        "Filter by specific document(s) (optional):",
        options=list(doc_options.keys()),
        default=[],
        help="Leave empty to search across all uploaded documents.",
    )
    selected_doc_ids = [doc_options[name] for name in selected_doc_names] if selected_doc_names else None

    # Chat / Query input
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How many days of annual leave do employees receive?",
        key="qa_input",
    )

    if st.button("Ask Assistant", type="primary", key="btn_qa") or user_query:
        if user_query.strip():
            with st.spinner("Analyzing query and retrieving evidence from Qdrant..."):
                try:
                    res = client.query_rag(
                        query=user_query,
                        top_k=top_k,
                        doc_ids=selected_doc_ids,
                    )
                    answer = res.get("answer", "")
                    citations = res.get("citations", [])
                    is_refusal = "couldn't find this information" in answer.lower()

                    st.markdown("### 💡 Answer")
                    if is_refusal:
                        st.markdown(f'<div class="refusal-card">{answer}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(answer)
                        st.success("● Answer Grounded in Context", icon="✅")

                    # Display Source Citations
                    if citations:
                        st.markdown("### 🔍 Verified Source Citations")
                        for idx, cite in enumerate(citations, 1):
                            with st.expander(f"📍 Citation {idx}: {cite['filename']} (Page {cite['page_number']}) — Similarity: {cite.get('score', 0):.3f}", expanded=True):
                                st.markdown(f"**Chunk ID:** `{cite['chunk_id']}`")
                                st.markdown(f"**Supporting Excerpt:**\n> {cite['snippet']}")
                except Exception as e:
                    st.error(f"Error querying assistant: {e}")
        else:
            st.warning("Please enter a question.")

# ------------------------------------------------------------------------------
# Tab 2: Document Comparison
# ------------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Compare Two Documents")
    st.caption("Identify concrete changes, modified policies, additions, and removals.")

    if len(docs) < 2:
        st.info("Please upload at least two PDF documents to perform comparison.")
    else:
        doc_names = list(doc_options.keys())
        col1, col2 = st.columns(2)
        with col1:
            doc_a_name = st.selectbox("Document A (e.g. Baseline / 2025):", options=doc_names, index=0)
        with col2:
            doc_b_name = st.selectbox("Document B (e.g. Updated / 2026):", options=doc_names, index=1 if len(doc_names) > 1 else 0)

        compare_topic = st.text_input(
            "Comparison Topic / Focus (optional):",
            placeholder="e.g., Leave policy, Notice period, Remote work guidelines",
            key="comp_topic",
        )

        if st.button("⚖️ Run Comparison", type="primary", key="btn_compare"):
            if doc_a_name == doc_b_name:
                st.warning("Please select two different documents to compare.")
            else:
                with st.spinner("Retrieving corresponding sections and analyzing differences..."):
                    try:
                        res = client.compare_documents(
                            doc_id_a=doc_options[doc_a_name],
                            doc_id_b=doc_options[doc_b_name],
                            topic=compare_topic,
                            top_k=top_k,
                        )
                        st.markdown("### 📊 Comparative Analysis")
                        st.markdown(res.get("comparison_summary", ""))

                        st.divider()
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.markdown(f"**Evidence from {doc_a_name}:**")
                            for c in res.get("citations_doc_a", []):
                                st.markdown(f"- **Page {c['page_number']}**: {c['snippet'][:120]}...")
                        with col_c2:
                            st.markdown(f"**Evidence from {doc_b_name}:**")
                            for c in res.get("citations_doc_b", []):
                                st.markdown(f"- **Page {c['page_number']}**: {c['snippet'][:120]}...")
                    except Exception as e:
                        st.error(f"Comparison error: {e}")

# ------------------------------------------------------------------------------
# Tab 3: Contradiction Detection
# ------------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Detect Factual Contradictions Across Documents")
    st.caption("Spot conflicting policies and statements without guessing.")

    contra_topic = st.text_input(
        "Topic or Policy Area to Check:",
        placeholder="e.g., annual leave days, remote work allowance, notice period",
        key="contra_input",
    )

    if st.button("⚠️ Scan for Contradictions", type="primary", key="btn_contra"):
        if contra_topic.strip():
            with st.spinner("Retrieving cross-document context and analyzing conflicts..."):
                try:
                    res = client.detect_contradictions(
                        topic=contra_topic,
                        top_k=top_k * 2,
                    )
                    has_conflicts = res.get("has_contradictions", False)
                    contradictions = res.get("contradictions", [])

                    if has_conflicts and contradictions:
                        st.error(f"🚨 Found {len(contradictions)} Contradiction(s) for topic: '{contra_topic}'")
                        for idx, item in enumerate(contradictions, 1):
                            st.markdown(
                                f"""
                                <div class="conflict-card">
                                    <h4>Conflict #{idx}</h4>
                                    <p><strong>Claim A:</strong> "{item['claim_a']}"<br>
                                    <small><em>Source: {item['source_a']}, Page {item['page_a']}</em></small></p>
                                    <p><strong>Claim B:</strong> "{item['claim_b']}"<br>
                                    <small><em>Source: {item['source_b']}, Page {item['page_b']}</em></small></p>
                                    <p><strong>Analysis:</strong> {item['explanation']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.success(f"✅ No contradictions detected for '{contra_topic}'.")
                        st.info(res.get("explanation", "All statements are consistent."))

                    # Show Inspected Sources
                    inspected = res.get("inspected_citations", [])
                    if inspected:
                        with st.expander("Inspected Source Chunks", expanded=False):
                            for c in inspected:
                                st.markdown(f"- **{c['filename']} (p.{c['page_number']})**: {c['snippet']}")
                except Exception as e:
                    st.error(f"Error detecting contradictions: {e}")
        else:
            st.warning("Please enter a topic to check.")

# ------------------------------------------------------------------------------
# Tab 4: Targeted Summarization
# ------------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Targeted Topic Summarization")
    st.caption("Retrieve and synthesize only the sections relevant to your topic.")

    col1, col2 = st.columns([1, 2])
    with col1:
        sum_doc_name = st.selectbox(
            "Select Document (optional):",
            options=["All Documents"] + list(doc_options.keys()),
            key="sum_doc_select",
        )
    with col2:
        sum_topic = st.text_input(
            "Topic to Summarize:",
            placeholder="e.g., Leave & Attendance, Engineering & Deployment, Remote Work",
            key="sum_topic_input",
        )

    if st.button("📝 Generate Focused Summary", type="primary", key="btn_summarize"):
        if sum_topic.strip():
            target_doc_id = doc_options[sum_doc_name] if sum_doc_name != "All Documents" else None
            with st.spinner("Extracting topic-specific sections and synthesizing summary..."):
                try:
                    res = client.summarize_topic(
                        topic=sum_topic,
                        doc_id=target_doc_id,
                        top_k=top_k,
                    )
                    st.markdown("### 📋 Executive Summary")
                    st.markdown(res.get("summary", ""))

                    citations = res.get("citations", [])
                    if citations:
                        st.markdown("### 🔍 Supporting Evidence")
                        for idx, c in enumerate(citations, 1):
                            with st.expander(f"Source {idx}: {c['filename']} (Page {c['page_number']})"):
                                st.markdown(f"> {c['snippet']}")
                except Exception as e:
                    st.error(f"Summarization error: {e}")
        else:
            st.warning("Please enter a topic to summarize.")

# ------------------------------------------------------------------------------
# Tab 5: Benchmark Evaluation
# ------------------------------------------------------------------------------

with tabs[4]:
    st.subheader("📊 RAG Benchmark Evaluation")
    st.caption("Evaluate Retrieval Recall, Grounding Score, Citation Accuracy, and Zero-Hallucination Refusal.")

    if st.button("🚀 Run Evaluation Suite", type="primary", key="btn_eval"):
        with st.spinner("Running evaluation test cases against benchmark dataset..."):
            try:
                report = client.run_evaluation(top_k=top_k)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval Recall@K", f"{report['retrieval_accuracy']}%")
                col2.metric("Grounding Score", f"{report['grounding_accuracy']}%")
                col3.metric("Citation Accuracy", f"{report['citation_accuracy']}%")
                col4.metric("Refusal Accuracy", f"{report['refusal_accuracy']}%")

                st.divider()
                st.markdown(f"**Overall Benchmark Pass Rate:** `{report['overall_pass_rate']}%` across `{report['total_test_cases']}` test cases.")

                st.markdown("### 📋 Test Case Breakdown")
                for r in report.get("results", []):
                    badge = "🟢 PASSED" if r["status"] == "PASSED" else "🔴 FAILED"
                    with st.expander(f"{badge} | {r['id']}: {r['question']}"):
                        st.markdown(f"**Answer:** {r['answer']}")
                        st.markdown(f"**Citations:** {', '.join(r.get('citations', [])) if r.get('citations') else 'None'}")
                        if "retrieval_hit" in r:
                            st.caption(f"Retrieval Hit: {r['retrieval_hit']} | Grounding Hit: {r['grounding_hit']} | Citation Hit: {r['citation_hit']}")
            except Exception as e:
                st.error(f"Evaluation error: {e}")
