"""RAG Evaluation framework measuring retrieval hit rate, grounding, citations, and refusal accuracy."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.workflows.rag_graph import get_rag_workflow
from app.retrieval.retriever import get_retriever
from app.ingestion.pdf_loader import load_pdf_from_path
from app.llm.prompts import STRICT_REFUSAL_MESSAGE

logger = logging.getLogger(__name__)


def load_evaluation_dataset(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load benchmark evaluation questions and expected labels."""
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation_dataset.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


class RAGEvaluator:
    """Evaluates the end-to-end RAG system against ground-truth benchmarks."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset = load_evaluation_dataset(dataset_path)
        self.workflow = get_rag_workflow()
        self.retriever = get_retriever()

        # Seed benchmark test document if collection is currently empty
        try:
            docs = self.retriever.vector_store.get_all_documents()
            if not docs:
                test_pdf_path = Path(__file__).resolve().parent.parent.parent / "data" / "documents" / "RAG_Test_Document_Realistic.pdf"
                if test_pdf_path.exists():
                    logger.info(f"Indexing evaluation benchmark document: {test_pdf_path.name}")
                    extracted = load_pdf_from_path(test_pdf_path)
                    self.retriever.vector_store.index_document(extracted)
        except Exception as e:
            logger.warning(f"Could not auto-seed evaluation document: {e}")

    def run_evaluation(self, top_k: int = 4) -> Dict[str, Any]:
        """
        Run all benchmark test cases and calculate quantitative metrics.
        
        Returns:
            Dict containing metrics summary and per-test breakdown.
        """
        total_cases = len(self.dataset)
        in_scope_cases = 0
        out_of_scope_cases = 0

        retrieval_hits = 0
        grounding_hits = 0
        citation_hits = 0
        refusal_hits = 0

        detailed_results = []

        for case in self.dataset:
            case_id = case["id"]
            question = case["question"]
            should_refuse = case.get("should_refuse", False)
            expected_keywords = [k.lower() for k in case.get("expected_answer_keywords", [])]
            expected_page = case.get("expected_page")
            expected_filename = case.get("expected_filename")

            # 1. Test Retrieval
            retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)

            # 2. Test End-to-End Workflow
            qa_response = self.workflow.run(question, top_k=top_k)
            answer = qa_response.answer.lower()
            citations = qa_response.citations

            test_result = {
                "id": case_id,
                "question": question,
                "should_refuse": should_refuse,
                "retrieved_count": len(retrieved_chunks),
                "answer": qa_response.answer,
                "citations": [f"{c.filename} (p.{c.page_number})" for c in citations],
            }

            if should_refuse:
                out_of_scope_cases += 1
                refusal_pass = (
                    STRICT_REFUSAL_MESSAGE.lower() in answer
                    or "couldn't find this information" in answer
                    or "could not find" in answer
                )
                if refusal_pass:
                    refusal_hits += 1
                test_result["status"] = "PASSED" if refusal_pass else "FAILED"
                test_result["metric"] = "Refusal Correctness"
            else:
                in_scope_cases += 1

                # Retrieval Recall Check
                ret_hit = False
                if expected_page is not None:
                    for chunk in retrieved_chunks:
                        if chunk.page_number == expected_page:
                            ret_hit = True
                            break
                else:
                    ret_hit = len(retrieved_chunks) > 0

                if ret_hit:
                    retrieval_hits += 1

                # Grounding / Keyword support Check
                grounded_pass = any(kw in answer for kw in expected_keywords)
                if grounded_pass:
                    grounding_hits += 1

                # Citation Accuracy Check
                cite_hit = False
                if expected_page is not None:
                    cite_hit = any(c.page_number == expected_page for c in citations)
                else:
                    cite_hit = len(citations) > 0

                if cite_hit:
                    citation_hits += 1

                case_passed = ret_hit and (grounded_pass or cite_hit)
                test_result["status"] = "PASSED" if case_passed else "FAILED"
                test_result["retrieval_hit"] = ret_hit
                test_result["grounding_hit"] = grounded_pass
                test_result["citation_hit"] = cite_hit

            detailed_results.append(test_result)

        retrieval_accuracy = (retrieval_hits / in_scope_cases) if in_scope_cases > 0 else 1.0
        grounding_accuracy = (grounding_hits / in_scope_cases) if in_scope_cases > 0 else 1.0
        citation_accuracy = (citation_hits / in_scope_cases) if in_scope_cases > 0 else 1.0
        refusal_accuracy = (refusal_hits / out_of_scope_cases) if out_of_scope_cases > 0 else 1.0

        summary = {
            "total_test_cases": total_cases,
            "in_scope_cases": in_scope_cases,
            "out_of_scope_cases": out_of_scope_cases,
            "retrieval_accuracy": round(retrieval_accuracy * 100, 2),
            "grounding_accuracy": round(grounding_accuracy * 100, 2),
            "citation_accuracy": round(citation_accuracy * 100, 2),
            "refusal_accuracy": round(refusal_accuracy * 100, 2),
            "overall_pass_rate": round(
                (sum(1 for r in detailed_results if r["status"] == "PASSED") / total_cases) * 100, 2
            ),
            "results": detailed_results,
        }

        return summary


def print_evaluation_report(summary: Dict[str, Any]):
    """Print formatted evaluation report to stdout."""
    print("\n" + "=" * 65)
    print("           RAG KNOWLEDGE ASSISTANT EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Test Cases Evaluated: {summary['total_test_cases']}")
    print(f"  - In-scope Questions:     {summary['in_scope_cases']}")
    print(f"  - Refusal (Out-of-scope): {summary['out_of_scope_cases']}")
    print("-" * 65)
    print(f"1. Retrieval Accuracy (Recall@4):   {summary['retrieval_accuracy']}%")
    print(f"2. Grounding / Fact Support:        {summary['grounding_accuracy']}%")
    print(f"3. Citation Correctness:            {summary['citation_accuracy']}%")
    print(f"4. Refusal Accuracy (Zero-halluc.): {summary['refusal_accuracy']}%")
    print("-" * 65)
    print(f"OVERALL BENCHMARK PASS RATE:        {summary['overall_pass_rate']}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    report = evaluator.run_evaluation()
    print_evaluation_report(report)
