"""End-to-end verification script testing all required manual and programmatic scenarios."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pdf_loader import load_pdf_from_path
from app.core.vector_store import QdrantVectorStore
from app.workflows.rag_graph import get_rag_workflow
from app.comparison.comparator import get_comparator
from app.evaluation.evaluator import RAGEvaluator
from app.llm.prompts import STRICT_REFUSAL_MESSAGE


def run_verification():
    print("=== STARTING RAG KNOWLEDGE ASSISTANT VERIFICATION ===")

    # Initialize vector store
    vstore = QdrantVectorStore()
    vstore.delete_collection()
    vstore.ensure_collection_exists()

    pdf_realistic = Path("data/documents/RAG_Test_Document_Realistic.pdf")
    pdf_2025 = Path("data/documents/policy_2025.pdf")
    pdf_2026 = Path("data/documents/policy_2026.pdf")

    if not pdf_realistic.exists():
        print(f"Error: {pdf_realistic} does not exist!")
        sys.exit(1)

    # Index test documents
    doc_realistic = load_pdf_from_path(pdf_realistic)
    vstore.index_document(doc_realistic)

    doc_2025 = load_pdf_from_path(pdf_2025)
    doc_2026 = load_pdf_from_path(pdf_2026)
    vstore.index_document(doc_2025)
    vstore.index_document(doc_2026)

    workflow = get_rag_workflow()

    # --------------------------------------------------------------------------
    # TEST 1: Known valid question
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Valid Question: 'How many days of paid annual leave do employees receive?'")
    q1 = "How many days of paid annual leave do employees receive?"
    res1 = workflow.run(q1)
    print(f"Answer: {res1.answer}")
    print(f"Grounded: {res1.is_grounded}")
    print(f"Citations ({len(res1.citations)}):")
    for c in res1.citations:
        print(f"  - {c.filename}, Page {c.page_number} (Score: {c.score:.3f})")

    assert "24" in res1.answer, f"Expected 24 in answer, got: {res1.answer}"
    assert res1.is_grounded is True, "Expected is_grounded to be True"
    assert len(res1.citations) > 0, "Expected at least 1 citation"
    assert "[Chunk" not in res1.answer, "Answer must not contain chunk markers"
    assert "Based on the documents:" not in res1.answer, "Answer must not contain 'Based on the documents:'"
    print(">>> TEST 1 PASSED")

    # --------------------------------------------------------------------------
    # TEST 2: Known unsupported question (Strict Refusal)
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Unsupported Question: 'What is the company\'s maternity leave policy?'")
    q2 = "What is the company's maternity leave policy?"
    res2 = workflow.run(q2)
    print(f"Answer: {res2.answer}")
    print(f"Citations count: {len(res2.citations)}")

    assert res2.answer == STRICT_REFUSAL_MESSAGE, f"Expected exact refusal, got: '{res2.answer}'"
    assert len(res2.citations) == 0, "Citations must be empty on refusal"
    print(">>> TEST 2 PASSED")

    # --------------------------------------------------------------------------
    # TEST 3: Out-of-scope question (Mars rover launch schedule)
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Out-of-scope Question: 'What is the company\'s Mars rover launch schedule?'")
    q3 = "What is the company's Mars rover launch schedule?"
    res3 = workflow.run(q3)
    print(f"Answer: {res3.answer}")
    print(f"Citations count: {len(res3.citations)}")

    assert res3.answer == STRICT_REFUSAL_MESSAGE, f"Expected exact refusal, got: '{res3.answer}'"
    assert len(res3.citations) == 0, "Citations must be empty on refusal"
    print(">>> TEST 3 PASSED")

    # --------------------------------------------------------------------------
    # TEST 4: Scoped search to a specific document
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Scoped Search: Query specifically against policy_2025.pdf")
    q4 = "What is the standard employee notice period upon resignation in 2025?"
    res4 = workflow.run(q4, doc_id=doc_2025.doc_id)
    print(f"Answer: {res4.answer}")
    print(f"Citations:")
    for c in res4.citations:
        print(f"  - {c.filename} (Page {c.page_number})")

    assert any(c.filename == "policy_2025.pdf" for c in res4.citations), "Expected citation from policy_2025.pdf"
    assert not any(c.filename == "policy_2026.pdf" for c in res4.citations), "policy_2026.pdf should not be cited"
    print(">>> TEST 4 PASSED")

    # --------------------------------------------------------------------------
    # TEST 5: Comparison, Contradiction Detection, Summarization, and Evaluation
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Feature verification: Comparison, Contradictions, Summarization, Evaluation")
    comparator = get_comparator()

    # 5a. Comparison
    comp_res = comparator.compare_documents(
        doc_id_a=doc_2025.doc_id,
        doc_id_b=doc_2026.doc_id,
        topic="annual leave days",
    )
    print(f"Comparison Summary: {comp_res.comparison_summary[:120]}...")
    assert len(comp_res.citations_doc_a) > 0, "Expected citations from Doc A"
    assert len(comp_res.citations_doc_b) > 0, "Expected citations from Doc B"

    # 5b. Contradictions
    contra_res = comparator.detect_contradictions(topic="annual leave days conflict")
    print(f"Contradictions Detected: {contra_res.has_contradictions}")
    assert contra_res.has_contradictions is True, "Expected contradictions to be detected"
    assert len(contra_res.contradictions) > 0, "Expected at least 1 contradiction item"

    # 5c. Summarization
    sum_res = comparator.summarize_topic(topic="annual leave", doc_id=doc_2026.doc_id)
    print(f"Summary: {sum_res.summary[:120]}...")
    assert len(sum_res.citations) > 0, "Expected citations for summary"

    # 5d. Benchmark Evaluation
    evaluator = RAGEvaluator()
    eval_res = evaluator.run_evaluation(top_k=2)
    print(f"Evaluation Metrics:")
    print(f"  - Total Test Cases:   {eval_res['total_test_cases']}")
    print(f"  - Retrieval Accuracy: {eval_res['retrieval_accuracy']}%")
    print(f"  - Grounding Accuracy: {eval_res['grounding_accuracy']}%")
    print(f"  - Citation Accuracy:  {eval_res['citation_accuracy']}%")
    print(f"  - Refusal Accuracy:   {eval_res['refusal_accuracy']}%")
    print(f"  - Overall Pass Rate:  {eval_res['overall_pass_rate']}%")
    assert eval_res["overall_pass_rate"] >= 80.0, "Benchmark pass rate must be >= 80%"

    print("\n=========================================================")
    print("[SUCCESS] ALL 5 END-TO-END VERIFICATION SCENARIOS PASSED!")
    print("=========================================================")


if __name__ == "__main__":
    run_verification()
