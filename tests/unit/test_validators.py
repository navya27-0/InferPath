"""Unit tests for deterministic response validators and composite validation pipeline."""

import pytest

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import RouteTier
from inferpath.validators.base import CompositeValidator
from inferpath.validators.format import FormatValidator
from inferpath.validators.length import LengthValidator
from inferpath.validators.non_empty import NonEmptyValidator
from inferpath.validators.refusal import RefusalDetector
from inferpath.validators.summary_consistency import SummaryConsistencyValidator


@pytest.fixture
def base_request():
    return UserRequest(request_id="v_req_1", prompt="Sample prompt")


def test_non_empty_validator(base_request):
    val = NonEmptyValidator()

    # Valid
    assert val.validate("Valid content", base_request).is_valid is True

    # Empty
    res_empty = val.validate("", base_request)
    assert res_empty.is_valid is False
    assert res_empty.failure_code == "EMPTY_OUTPUT"

    # Whitespace only
    res_ws = val.validate("   \n\t  ", base_request)
    assert res_ws.is_valid is False
    assert res_ws.failure_code == "EMPTY_OUTPUT"


def test_length_validator(base_request):
    val = LengthValidator()

    # Default bounds pass
    assert val.validate("Standard text", base_request).is_valid is True

    # Min bound violation
    req_min = UserRequest(request_id="v2", prompt="Test", metadata={"min_length": 20})
    res_short = val.validate("Too short", req_min)
    assert res_short.is_valid is False
    assert res_short.failure_code == "LENGTH_TOO_SHORT"

    # Max bound violation
    req_max = UserRequest(request_id="v3", prompt="Test", metadata={"max_length": 10})
    res_long = val.validate("This sentence is longer than ten characters.", req_max)
    assert res_long.is_valid is False
    assert res_long.failure_code == "LENGTH_TOO_LONG"


def test_format_validator_json():
    val = FormatValidator()
    req_json = UserRequest(
        request_id="v4",
        prompt="Output JSON",
        task_parameters=TaskParameters(expected_format="json"),
    )

    # Valid JSON
    assert val.validate('{"status": "ok", "code": 200}', req_json).is_valid is True
    # Valid JSON in code block
    assert val.validate('```json\n{"status": "ok"}\n```', req_json).is_valid is True

    # Invalid JSON
    res_bad = val.validate("Not a JSON object", req_json)
    assert res_bad.is_valid is False
    assert res_bad.failure_code == "FORMAT_INVALID_JSON"


def test_format_validator_required_headings():
    val = FormatValidator()
    req_headings = UserRequest(
        request_id="v5",
        prompt="Write article",
        metadata={"required_headings": ["Introduction", "Methodology", "Conclusion"]},
    )

    # Present
    text_ok = "# Introduction\nText here\n## Methodology\nMore text\n### Conclusion\nDone."
    assert val.validate(text_ok, req_headings).is_valid is True

    # Missing heading
    text_missing = "# Introduction\nText here\n### Conclusion\nDone."
    res_missing = val.validate(text_missing, req_headings)
    assert res_missing.is_valid is False
    assert res_missing.failure_code == "FORMAT_MISSING_HEADINGS"
    assert "Methodology" in res_missing.rejection_reason


def test_refusal_detector(base_request):
    detector = RefusalDetector()

    # Valid text containing negative sentiment or normal words
    assert (
        detector.validate("I cannot believe how good this movie was!", base_request).is_valid
        is True
    )
    assert detector.validate("We cannot decline the client offer.", base_request).is_valid is True

    # Explicit refusal phrases
    refusal_samples = [
        "I cannot fulfill this request because it violates safety guidelines.",
        "As an AI language model, I cannot provide instructions on that.",
        "I apologize, but I cannot assist with this task.",
        "I must decline to generate that content.",
    ]
    for sample in refusal_samples:
        res = detector.validate(sample, base_request)
        assert res.is_valid is False
        assert res.failure_code == "MODEL_REFUSAL"


def test_summary_consistency_validator():
    val = SummaryConsistencyValidator()
    context = (
        "In 1969, Apollo 11 astronauts Neil Armstrong and Buzz Aldrin landed the Eagle lunar module "
        "on the Moon while Michael Collins remained in orbit in the command module Columbia."
    )
    req_consistency = UserRequest(
        request_id="v6",
        prompt="Summarize the landing",
        context=context,
        metadata={"enable_summary_consistency": True},
    )

    # Consistent summary
    good_summary = "Armstrong and Aldrin landed Apollo 11 on the Moon in 1969."
    assert val.validate(good_summary, req_consistency).is_valid is True

    # Inconsistent summary (completely unrelated text omitting all entities)
    bad_summary = "The weather was sunny and birds were singing happily in the garden."
    res_bad = val.validate(bad_summary, req_consistency)
    assert res_bad.is_valid is False
    assert res_bad.failure_code == "SUMMARY_INCONSISTENCY"


def test_composite_validator_escalation():
    """Verify composite validator recommends fallback for Tier C failure."""
    comp = CompositeValidator(
        validators=[NonEmptyValidator(), FormatValidator(), RefusalDetector()]
    )
    req = UserRequest(
        request_id="v7",
        prompt="Test",
        task_parameters=TaskParameters(expected_format="json"),
    )

    # Empty failure on Tier C recommends fallback to Tier D
    res_c = comp.validate("", req, current_tier=RouteTier.SMALL_MODEL_C)
    assert res_c.is_valid is False
    assert res_c.fallback_recommended is True
    assert res_c.next_recommended_tier == RouteTier.LARGE_MODEL_D

    # Failure on Tier D cannot escalate further
    res_d = comp.validate("", req, current_tier=RouteTier.LARGE_MODEL_D)
    assert res_d.is_valid is False
    assert res_d.fallback_recommended is False
    assert res_d.next_recommended_tier is None
