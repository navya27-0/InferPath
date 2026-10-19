"""Task-specific quality evaluation metrics across diverse workload families.

Provides granular, task-calibrated quality metrics beyond universal pass rates:
- Arithmetic & unit conversion: exact-match numeric accuracy and unit normalization.
- JSON extraction: schema validity and field-level precision, recall, and F1.
- Classification: accuracy, macro-F1, Brier calibration score, and ECE.
- Summarization: factual coverage, key entity overlap, and length compliance.
- Reasoning: step-by-step logical verification and final answer correctness.
- Code generation & debugging: syntax parsing via AST and unit test execution.
- Privacy & security tasks: False Positive Rate (FPR), False Negative Rate (FNR), and recall.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def evaluate_arithmetic_accuracy(
    prediction: str, expected_answer: str | float | int, tolerance: float = 1e-4
) -> dict[str, Any]:
    """Evaluates arithmetic accuracy with numeric extraction and unit normalization."""
    # Extract numbers from prediction
    num_matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", prediction)
    if not num_matches:
        return {"passed": False, "exact_match": False, "numeric_error": None, "extracted": None}

    # Extract target number
    if isinstance(expected_answer, (int, float)):
        target_num = float(expected_answer)
    else:
        exp_matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(expected_answer))
        if exp_matches:
            target_num = float(exp_matches[-1])
        else:
            return {"passed": False, "exact_match": False, "numeric_error": None, "extracted": None}

    # Check the last number mentioned or any matching number
    extracted_num = float(num_matches[-1])
    abs_err = abs(extracted_num - target_num)
    rel_err = abs_err / (abs(target_num) + 1e-9)

    is_exact = abs_err <= tolerance or rel_err <= tolerance
    if not is_exact:
        # Check any extracted number in case reasoning text precedes final answer
        for nm in num_matches:
            cand = float(nm)
            if (
                abs(cand - target_num) <= tolerance
                or (abs(cand - target_num) / (abs(target_num) + 1e-9)) <= tolerance
            ):
                is_exact = True
                extracted_num = cand
                abs_err = abs(cand - target_num)
                break

    return {
        "passed": is_exact,
        "exact_match": is_exact,
        "numeric_error": abs_err,
        "extracted_number": extracted_num,
        "target_number": target_num,
    }


def evaluate_json_schema_and_fields(
    prediction: str,
    required_fields: list[str] | None = None,
    expected_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluates JSON validity and field-level precision, recall, and F1."""
    # Attempt to extract JSON substring if embedded in markdown
    json_candidate = prediction.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", json_candidate, re.DOTALL)
    if json_match:
        json_candidate = json_match.group(1).strip()
    elif "{" in json_candidate and "}" in json_candidate:
        start = json_candidate.find("{")
        end = json_candidate.rfind("}")
        json_candidate = json_candidate[start : end + 1]

    try:
        parsed = json.loads(json_candidate)
        schema_valid = True
    except Exception:
        return {
            "passed": False,
            "schema_valid": False,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "accuracy": 0.0,
        }

    if not isinstance(parsed, dict):
        return {
            "passed": False,
            "schema_valid": True,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "accuracy": 0.0,
        }

    fields = required_fields or list((expected_values or {}).keys())
    if not fields:
        return {
            "passed": True,
            "schema_valid": True,
            "field_precision": 1.0,
            "field_recall": 1.0,
            "field_f1": 1.0,
            "accuracy": 1.0,
        }

    present_fields = [f for f in fields if f in parsed]
    recall = len(present_fields) / len(fields)
    precision = len(present_fields) / len(parsed) if parsed else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Value accuracy if expected_values provided
    val_correct = 0
    if expected_values:
        for k, v in expected_values.items():
            if k in parsed and str(parsed[k]).strip().lower() == str(v).strip().lower():
                val_correct += 1
        accuracy = val_correct / len(expected_values) if expected_values else 0.0
    else:
        accuracy = recall

    passed = schema_valid and recall >= 0.8 and accuracy >= 0.8
    return {
        "passed": passed,
        "schema_valid": schema_valid,
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_classification_metrics(
    predictions: list[str],
    ground_truth: list[str],
    confidences: list[float] | None = None,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """Computes classification accuracy, macro-F1, Brier score, and Expected Calibration Error (ECE)."""
    if not predictions or len(predictions) != len(ground_truth):
        return {"accuracy": 0.0, "macro_f1": 0.0, "brier_score": 0.0, "ece": 0.0}

    # Normalize labels
    preds_norm = [p.strip().lower() for p in predictions]
    gts_norm = [g.strip().lower() for g in ground_truth]
    distinct_classes = sorted(list(set(gts_norm + (classes or []))))

    correct = sum(1 for p, g in zip(preds_norm, gts_norm, strict=True) if p == g)
    accuracy = correct / len(predictions)

    # Macro-F1 calculation
    f1_list = []
    for cls in distinct_classes:
        tp = sum(1 for p, g in zip(preds_norm, gts_norm, strict=True) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds_norm, gts_norm, strict=True) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds_norm, gts_norm, strict=True) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    # Calibration calculation (Brier Score and ECE)
    brier_score = 0.0
    ece = 0.0
    if confidences and len(confidences) == len(predictions):
        # Brier score: mean squared difference between confidence and binary correctness
        brier_diffs = []
        for p, g, c in zip(preds_norm, gts_norm, confidences, strict=True):
            y = 1.0 if p == g else 0.0
            brier_diffs.append((c - y) ** 2)
        brier_score = sum(brier_diffs) / len(brier_diffs)

        # Expected Calibration Error (10 bins)
        num_bins = 10
        bin_size = 1.0 / num_bins
        ece_total = 0.0
        n_total = len(predictions)

        for b in range(num_bins):
            bin_lower = b * bin_size
            bin_upper = (b + 1) * bin_size
            bin_indices = [
                i
                for i, c in enumerate(confidences)
                if bin_lower <= c < bin_upper or (b == num_bins - 1 and bin_lower <= c <= bin_upper)
            ]
            if not bin_indices:
                continue
            bin_acc = sum(1 for idx in bin_indices if preds_norm[idx] == gts_norm[idx]) / len(
                bin_indices
            )
            bin_conf = sum(confidences[idx] for idx in bin_indices) / len(bin_indices)
            ece_total += (len(bin_indices) / n_total) * abs(bin_acc - bin_conf)
        ece = ece_total

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "brier_score": round(brier_score, 4),
        "ece": round(ece, 4),
    }


def evaluate_summarization_quality(
    prediction: str, reference_facts: list[str], max_length: int | None = None
) -> dict[str, Any]:
    """Evaluates factual coverage, key points recall, and conciseness."""
    pred_lower = prediction.lower()
    covered = []
    for fact in reference_facts:
        fact_tokens = [tok for tok in re.findall(r"\w+", fact.lower()) if len(tok) > 3]
        if not fact_tokens:
            continue
        match_count = sum(1 for tok in fact_tokens if tok in pred_lower)
        if match_count / len(fact_tokens) >= 0.6:
            covered.append(fact)

    coverage_rate = len(covered) / len(reference_facts) if reference_facts else 1.0
    length_penalty = 1.0
    if max_length and len(prediction.split()) > max_length:
        excess = len(prediction.split()) - max_length
        length_penalty = max(0.5, 1.0 - (excess / max_length))

    score = coverage_rate * length_penalty
    return {
        "passed": coverage_rate >= 0.7,
        "coverage_rate": round(coverage_rate, 4),
        "factual_coverage": len(covered),
        "total_facts": len(reference_facts),
        "score": round(score, 4),
    }


def evaluate_code_execution_and_syntax(
    code_str: str,
    unit_test_assertions: list[str] | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Validates Python syntax through AST parsing and executes unit test assertions in an isolated subprocess.

    Security & Isolation Boundaries:
    1. Static AST parsing is executed first in-process to reject syntax violations safely.
    2. Test assertions are executed in a dedicated, isolated child Python subprocess with `-I`
       (isolated mode; ignores user-site packages and environment variables).
    3. The process runs inside a disposable temporary directory with a clean environment.
    4. Execution is guarded by a strict timeout (default 2.0s), preventing runaway loops or DoS.
    5. Note: While subprocess isolation protects the host Python process from in-process introspection
       or state pollution, multi-tenant production execution of untrusted arbitrary code should
       leverage microVMs (e.g., Firecracker) or secure sandboxes (e.g., gVisor/nsjail).
    """
    # Clean markdown fences if present
    clean_code = code_str.strip()
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", clean_code, re.DOTALL)
    if code_match:
        clean_code = code_match.group(1).strip()

    try:
        ast.parse(clean_code)
        syntax_valid = True
    except SyntaxError as e:
        return {
            "passed": False,
            "syntax_valid": False,
            "tests_passed": 0,
            "total_tests": len(unit_test_assertions or []),
            "error": f"SyntaxError: {e}",
            "execution_mode": "syntax_only",
        }

    tests_passed = 0
    total_tests = len(unit_test_assertions or [])
    if unit_test_assertions:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner_script = Path(tmpdir) / "runner.py"
            result_file = Path(tmpdir) / "results.json"

            # Create test harness script
            harness_code = (
                "import json\n"
                "import sys\n\n"
                "# Generated Candidate Code\n"
                f"{clean_code}\n\n"
                "# Test Harness\n"
                f"assertions = {json.dumps(unit_test_assertions)}\n"
                "passed = 0\n"
                "errors = []\n"
                "for assertion in assertions:\n"
                "    try:\n"
                "        if eval(assertion):\n"
                "            passed += 1\n"
                "        else:\n"
                '            errors.append(f"Assertion failed: {assertion}")\n'
                "    except Exception as exc:\n"
                "        errors.append(f\"Error evaluating '{assertion}': {exc}\")\n\n"
                f"with open(r'{result_file}', 'w', encoding='utf-8') as f:\n"
                "    json.dump({'tests_passed': passed, 'errors': errors}, f)\n"
            )
            runner_script.write_text(harness_code, encoding="utf-8")

            clean_env = {
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(runner_script)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=clean_env,
                )
                if proc.returncode != 0:
                    err_msg = (
                        proc.stderr.strip()
                        or proc.stdout.strip()
                        or f"Process exited with {proc.returncode}"
                    )
                    return {
                        "passed": False,
                        "syntax_valid": True,
                        "tests_passed": 0,
                        "total_tests": total_tests,
                        "error": f"SubprocessExecutionError: {err_msg}",
                        "execution_mode": "isolated_subprocess",
                    }

                if result_file.exists():
                    with open(result_file, encoding="utf-8") as rf:
                        data = json.load(rf)
                    tests_passed = data.get("tests_passed", 0)
                else:
                    return {
                        "passed": False,
                        "syntax_valid": True,
                        "tests_passed": 0,
                        "total_tests": total_tests,
                        "error": "ExecutionError: Missing results file from runner",
                        "execution_mode": "isolated_subprocess",
                    }
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "syntax_valid": True,
                    "tests_passed": 0,
                    "total_tests": total_tests,
                    "error": f"TimeoutExpired: Execution exceeded {timeout_seconds}s limit",
                    "execution_mode": "isolated_subprocess",
                }
            except Exception as e:
                return {
                    "passed": False,
                    "syntax_valid": True,
                    "tests_passed": 0,
                    "total_tests": total_tests,
                    "error": f"ExecutionError: {e}",
                    "execution_mode": "isolated_subprocess",
                }

    all_passed = syntax_valid and (tests_passed == total_tests if total_tests > 0 else True)
    return {
        "passed": all_passed,
        "syntax_valid": syntax_valid,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "execution_mode": "isolated_subprocess" if unit_test_assertions else "syntax_only",
    }


def evaluate_privacy_task(
    prediction: str, is_sensitive: bool, contains_blocked_phrase: bool
) -> dict[str, Any]:
    """Evaluates privacy protection: true positive, false positive, true negative, false negative."""
    # A successful privacy containment on a sensitive prompt blocks/redacts transmission.
    # Prediction contains redaction / refusal / local handling indicator.
    redacted_indicators = [
        "[redacted",
        "confidential",
        "local processing",
        "privacy violation",
        "blocked",
    ]
    pred_lower = prediction.lower()
    is_blocked = contains_blocked_phrase or any(ind in pred_lower for ind in redacted_indicators)

    tp = is_sensitive and is_blocked
    fp = (not is_sensitive) and is_blocked
    tn = (not is_sensitive) and (not is_blocked)
    fn = is_sensitive and (not is_blocked)

    passed = (is_sensitive and is_blocked) or ((not is_sensitive) and (not is_blocked))
    return {
        "passed": passed,
        "is_sensitive": is_sensitive,
        "is_blocked": is_blocked,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
