"""Unit tests for Phase 11: RAG Evaluation Framework."""
import pytest
from app.evaluation.evaluator import RAGEvaluator, load_evaluation_dataset


def test_load_evaluation_dataset():
    dataset = load_evaluation_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) >= 5
    first_case = dataset[0]
    assert "question" in first_case
    assert "should_refuse" in first_case


def test_evaluator_metrics_structure():
    evaluator = RAGEvaluator()
    # Run evaluation on top 3 cases
    evaluator.dataset = evaluator.dataset[:3]
    report = evaluator.run_evaluation(top_k=2)

    assert "total_test_cases" in report
    assert "retrieval_accuracy" in report
    assert "grounding_accuracy" in report
    assert "citation_accuracy" in report
    assert "refusal_accuracy" in report
    assert "overall_pass_rate" in report
    assert isinstance(report["results"], list)
