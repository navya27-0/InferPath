"""Unit tests for task-specific evaluation metrics."""

from inferpath.evaluation.task_metrics import (
    evaluate_arithmetic_accuracy,
    evaluate_classification_metrics,
    evaluate_code_execution_and_syntax,
    evaluate_json_schema_and_fields,
    evaluate_privacy_task,
    evaluate_summarization_quality,
)


def test_arithmetic_accuracy():
    # Exact match
    res1 = evaluate_arithmetic_accuracy("The result is 42.", 42)
    assert res1["passed"] is True
    assert res1["numeric_error"] == 0.0

    # Within tolerance
    res2 = evaluate_arithmetic_accuracy("Calculation yields 3.14159", 3.1416, tolerance=1e-3)
    assert res2["passed"] is True

    # Failed
    res3 = evaluate_arithmetic_accuracy("Answer: 100", 50)
    assert res3["passed"] is False


def test_json_schema_and_fields():
    pred_json = '{"name": "Alice", "role": "engineer", "level": 5}'
    res1 = evaluate_json_schema_and_fields(
        pred_json,
        required_fields=["name", "role", "level"],
        expected_values={"name": "Alice", "role": "engineer", "level": 5},
    )
    assert res1["passed"] is True
    assert res1["schema_valid"] is True
    assert res1["field_f1"] == 1.0

    # Malformed JSON
    res2 = evaluate_json_schema_and_fields("not json at all", required_fields=["name"])
    assert res2["passed"] is False
    assert res2["schema_valid"] is False


def test_classification_metrics():
    preds = ["cat", "dog", "bird", "cat"]
    gts = ["cat", "dog", "dog", "cat"]
    confs = [0.9, 0.8, 0.7, 0.95]

    metrics = evaluate_classification_metrics(preds, gts, confs)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] > 0.0
    assert "brier_score" in metrics
    assert "ece" in metrics


def test_summarization_quality():
    pred = "The company reported Q3 revenue growth of 15% and announced a stock split."
    facts = ["Q3 revenue growth", "stock split", "merger approved"]
    res = evaluate_summarization_quality(pred, facts)
    assert res["factual_coverage"] == 2
    assert res["total_facts"] == 3
    assert res["coverage_rate"] == round(2 / 3, 4)


def test_code_execution_and_syntax():
    # Valid syntax & passing assertions in isolated subprocess
    code = "def add(a, b):\n    return a + b\n"
    res1 = evaluate_code_execution_and_syntax(code, ["add(2, 3) == 5", "add(-1, 1) == 0"])
    assert res1["passed"] is True
    assert res1["syntax_valid"] is True
    assert res1["tests_passed"] == 2
    assert res1["execution_mode"] == "isolated_subprocess"

    # Syntax error
    bad_code = "def invalid syntax here :"
    res2 = evaluate_code_execution_and_syntax(bad_code)
    assert res2["passed"] is False
    assert res2["syntax_valid"] is False
    assert res2["execution_mode"] == "syntax_only"

    # Runtime error in code under test
    error_code = "def boom():\n    raise ValueError('runtime crash')\nboom()"
    res3 = evaluate_code_execution_and_syntax(error_code, ["True"])
    assert res3["passed"] is False
    assert res3["syntax_valid"] is True
    assert "SubprocessExecutionError" in res3["error"]

    # Timeout guard on runaway infinite loop
    loop_code = "while True:\n    pass"
    res4 = evaluate_code_execution_and_syntax(loop_code, ["True"], timeout_seconds=0.5)
    assert res4["passed"] is False
    assert res4["syntax_valid"] is True
    assert "TimeoutExpired" in res4["error"]


def test_privacy_evaluation():
    # True positive: sensitive prompt blocked
    res1 = evaluate_privacy_task(
        "[REDACTED_CONFIDENTIAL]", is_sensitive=True, contains_blocked_phrase=True
    )
    assert res1["passed"] is True
    assert res1["tp"] is True

    # False negative: sensitive prompt not blocked
    res2 = evaluate_privacy_task(
        "Your SSN is 000-11-2222", is_sensitive=True, contains_blocked_phrase=False
    )
    assert res2["passed"] is False
    assert res2["fn"] is True
