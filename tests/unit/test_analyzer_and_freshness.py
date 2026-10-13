"""Unit tests for RequestAnalyzer and FreshnessClassifier safeguards."""

from inferpath.cache.freshness import FreshnessClassifier
from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier, TaskType
from inferpath.router.analyzer import RequestAnalyzer


def test_freshness_explicit_current_time():
    prompts = [
        "What time is it right now?",
        "Tell me the current time in London",
        "What is today's date?",
        "What day is today?",
        "What date is it?",
    ]
    for prompt in prompts:
        is_fresh, reason = FreshnessClassifier.is_time_sensitive(prompt)
        assert is_fresh is True, f"Expected '{prompt}' to be time-sensitive"
        assert reason is not None


def test_freshness_live_changing_data():
    prompts = [
        "What is the weather in Paris today?",
        "Current stock price of Microsoft",
        "Latest bitcoin price",
        "Exchange rate USD to EUR",
        "Breaking news in politics",
    ]
    for prompt in prompts:
        is_fresh, reason = FreshnessClassifier.is_time_sensitive(prompt)
        assert is_fresh is True, f"Expected '{prompt}' to be time-sensitive"
        assert reason is not None


def test_freshness_historical_and_rhetorical_idioms_not_time_sensitive():
    # Historical questions or discourse idioms where 'today' or 'now' does not mean live data
    prompts = [
        "Today we know that dinosaurs went extinct 66 million years ago. Explain the impact hypothesis.",
        "Scientists now understand that the universe is expanding. Who discovered this?",
        "In 1945, World War II ended. What were the key treaties signed?",
        "Who was the first president of the United States?",
    ]
    for prompt in prompts:
        is_fresh, _ = FreshnessClassifier.is_time_sensitive(prompt)
        assert is_fresh is False, f"Expected '{prompt}' NOT to be flagged as time-sensitive"


def test_freshness_timeless_questions():
    prompts = [
        "Calculate 25 * 4",
        "Convert 25C to Fahrenheit",
        "How many kilometers are in a mile?",
        "Explain photosynthesis",
        "Write a Python function to reverse a linked list",
    ]
    for prompt in prompts:
        is_fresh, _ = FreshnessClassifier.is_time_sensitive(prompt)
        assert is_fresh is False, f"Expected '{prompt}' NOT to be time-sensitive"


def test_analyzer_task_intent_extraction():
    analyzer = RequestAnalyzer()

    # Math
    req_math = UserRequest(request_id="m1", prompt="Calculate 125 + 375")
    res_math = analyzer.analyze(req_math)
    assert res_math.task_intent == "arithmetic"
    assert res_math.suitable_tier == RouteTier.DETERMINISTIC_A
    assert res_math.task_family == TaskType.ARITHMETIC_MATH

    # Unit
    req_unit = UserRequest(request_id="u1", prompt="Convert 100 meters to feet")
    res_unit = analyzer.analyze(req_unit)
    assert res_unit.task_intent == "unit_conversion"
    assert res_unit.suitable_tier == RouteTier.DETERMINISTIC_A

    # Template
    req_tpl = UserRequest(
        request_id="t1", prompt="Draft a formal email to John regarding project kickoff"
    )
    res_tpl = analyzer.analyze(req_tpl)
    assert res_tpl.task_intent == "template_formal_email"
    assert res_tpl.suitable_tier == RouteTier.TEMPLATE_B
    assert res_tpl.template_id == "formal_email"


def test_analyzer_privacy_not_downgraded():
    analyzer = RequestAnalyzer()

    # Explicit internal stays internal
    req_internal = UserRequest(
        request_id="p1",
        prompt="Explain our internal sprint backlog",
        privacy_class=PrivacyClass.INTERNAL,
    )
    res_internal = analyzer.analyze(req_internal)
    assert res_internal.privacy_class == PrivacyClass.INTERNAL

    # Sensitive token upgrades public to confidential_private conservatively
    req_sensitive = UserRequest(
        request_id="p2",
        prompt="My API_KEY is sk-openrouter-secret-12345. How do I authenticate?",
        privacy_class=PrivacyClass.PUBLIC,
    )
    res_sensitive = analyzer.analyze(req_sensitive)
    assert res_sensitive.privacy_class == PrivacyClass.CONFIDENTIAL_PRIVATE


def test_analyzer_prompt_length_is_not_difficulty_signal():
    analyzer = RequestAnalyzer()

    # Short prompt that is HARD (has complexity keyword 'synthesize' or 'proof')
    short_hard = UserRequest(request_id="sh1", prompt="Synthesize evidence for cosmic inflation.")
    res_sh = analyzer.analyze(short_hard)
    assert res_sh.suitable_tier == RouteTier.LARGE_MODEL_D
    assert "synthesize" in res_sh.complexity_indicators

    # Long prompt that is SIMPLE (arithmetic calculation with verbose preamble)
    long_simple = UserRequest(
        request_id="ls1",
        prompt=(
            "Hello, I am working on an engineering project and I have gathered several measurements "
            "over the course of three weeks across various experimental setups. We need to verify "
            "one particular calculation for our preliminary dataset: please compute 500 * 20."
        ),
    )
    res_ls = analyzer.analyze(long_simple)
    assert res_ls.suitable_tier == RouteTier.DETERMINISTIC_A
    assert res_ls.task_intent == "arithmetic"
