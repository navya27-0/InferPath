"""Route-decision test table verifying 45+ labeled prompts across all priority rules.

Verifies:
- Explicit rule IDs (R01..R10)
- Priority precedence: Privacy -> Freshness -> Deterministic -> Templates -> Complexity -> Small -> Large
- Edge cases: short hard, long simple, ambiguous, private local vs external, missing template slots.
"""

import pytest

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.router import RuleBasedRouter
from inferpath.security.privacy import PrivacyViolationError

# Labeled test table cases: (id, prompt, privacy_class, is_time_sensitive, task_params, expected_tier, expected_rule, expected_clarification, rationale)
ROUTE_TEST_CASES = [
    # --- Group 1: Deterministic Arithmetic (R03) ---
    (
        "TC01",
        "Calculate 42 * 17",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Basic arithmetic multiplication",
    ),
    (
        "TC02",
        "What is (100 - 25) / 5?",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Arithmetic with parentheses",
    ),
    (
        "TC03",
        "Compute 2 ** 10",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Arithmetic exponentiation",
    ),
    (
        "TC04",
        "Calculate 50 + 25 * 2",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Arithmetic operator precedence",
    ),
    # --- Group 2: Deterministic Unit Conversion (R04) ---
    (
        "TC05",
        "Convert 25°C to Fahrenheit",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Acceptance criteria: 25C to Fahrenheit",
    ),
    (
        "TC06",
        "Convert 100 F into Celsius",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Temperature conversion F to C",
    ),
    (
        "TC07",
        "Convert 5 kilometers to miles",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Length conversion km to miles",
    ),
    (
        "TC08",
        "Convert 10 meters into feet",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Length conversion meters to feet",
    ),
    (
        "TC09",
        "Convert 500 grams to ounces",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Mass conversion grams to ounces",
    ),
    (
        "TC10",
        "Convert 2 kilograms to pounds",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Mass conversion kg to lbs",
    ),
    # --- Group 3: Deterministic Date Formatting (R05) ---
    (
        "TC11",
        "Format 2026-10-13 as human_long",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R05",
        False,
        "Date formatting ISO to human long",
    ),
    (
        "TC12",
        "Format date 13/10/2026 in rfc2822",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R05",
        False,
        "Date formatting to RFC2822",
    ),
    (
        "TC13",
        "Convert date '2026-12-25' to %B %d, %Y",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R05",
        False,
        "Date formatting with custom pattern",
    ),
    # --- Group 4: Deterministic JSON & Schema Validation (R06) ---
    (
        "TC14",
        'Validate json: {"name": "Alice", "valid": true}',
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R06",
        False,
        "JSON syntax validation",
    ),
    (
        "TC15",
        'Validate schema for {"id": 10}',
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R06",
        False,
        "JSON schema validation intent",
    ),
    # --- Group 5: Jinja2 Templates (R07) Complete vs Missing ---
    (
        "TC16",
        "Draft a leave application for Alice for annual leave from 2026-11-01 to 2026-11-05 due to vacation",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        False,
        "Complete leave application template",
    ),
    (
        "TC17",
        "Draft a leave application for Bob for medical leave due to dental surgery",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        True,
        "Acceptance criteria: leave application missing dates requires clarification",
    ),
    (
        "TC18",
        "Write a formal email to John regarding project kickoff from Alice",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        False,
        "Complete formal email template",
    ),
    (
        "TC19",
        "Draft a formal email to Dr. Jones",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        True,
        "Formal email missing subject, purpose, sender",
    ),
    (
        "TC20",
        "Meeting minutes for Architecture Review on 2026-10-13 regarding caching decisions",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        True,
        "Meeting minutes missing attendees",
    ),
    (
        "TC21",
        "Draft cover letter for Carol for Senior Software Engineer at Google with 8 years of Python experience",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        False,
        "Complete cover letter template",
    ),
    (
        "TC22",
        "Write a cover letter for David",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.TEMPLATE_B,
        "R07",
        True,
        "Cover letter missing job_title, company_name, experience",
    ),
    # --- Group 6: Standard Tasks to Small Model (R09) ---
    (
        "TC23",
        "Summarize the key events of the Apollo 11 moon landing.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Standard summarization query",
    ),
    (
        "TC24",
        "Draft a polite thank-you message to a colleague for lunch.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Standard short message drafting",
    ),
    (
        "TC25",
        "What is the capital of France?",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Factoid question answering",
    ),
    (
        "TC26",
        "Classify the sentiment of: 'I love this product, it works flawlessly!'",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Sentiment classification",
    ),
    (
        "TC27",
        "Explain how binary search works in simple terms.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Standard educational explanation",
    ),
    # --- Group 7: Complex Tasks to Large Model (R08) ---
    (
        "TC28",
        "Compare and contrast the economic theories of Keynes and Hayek.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Complex comparison task",
    ),
    (
        "TC29",
        "Synthesize findings across quantum computing and cryptography literature.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Complex synthesis task",
    ),
    (
        "TC30",
        "Evaluate evidence for gravitational wave detection from binary neutron stars.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Complex evidence evaluation",
    ),
    (
        "TC31",
        "Provide a multi-step proof that the square root of 2 is irrational.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Multi-step mathematical proof",
    ),
    (
        "TC32",
        "Analyze the trade-offs between Paxos and Raft consensus algorithms.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Complex trade-offs analysis",
    ),
    # --- Group 8: Privacy Precedence (R01) vs Local Tier ---
    (
        "TC33",
        "Summarize internal salary and payroll records for employee Alice",
        PrivacyClass.CONFIDENTIAL_PRIVATE,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R01",
        False,
        "Confidential query requiring external model blocked by privacy policy",
    ),
    (
        "TC34",
        "Analyze customer credit card transactions and SSN records",
        PrivacyClass.INTERNAL,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R01",
        False,
        "Internal data requiring external model blocked by privacy policy",
    ),
    (
        "TC35",
        "Calculate 25 * 4 in a strictly confidential context",
        PrivacyClass.CONFIDENTIAL_PRIVATE,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Confidential arithmetic safely executed locally by Tier A without leakage",
    ),
    # --- Group 9: Freshness / Dynamic Information Precedence (R02) ---
    (
        "TC36",
        "What is the current stock price of Apple right now?",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R02",
        False,
        "Live stock price query bypasses cache",
    ),
    (
        "TC37",
        "What is the weather in Seattle today?",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R02",
        False,
        "Live weather forecast query bypasses cache",
    ),
    (
        "TC38",
        "What time is it in Tokyo right now?",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R02",
        False,
        "Explicit current time request bypasses cache",
    ),
    (
        "TC39",
        "Latest breaking news in artificial intelligence",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R02",
        False,
        "Breaking news query bypasses cache",
    ),
    # --- Group 10: Counterexamples: Prompt Length != Difficulty ---
    (
        "TC40",
        "Prove P != NP.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Short but hard prompt (4 words) routed to large model",
    ),
    (
        "TC41",
        "Synthesize dark matter research.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R08",
        False,
        "Short but hard prompt (4 words) routed to large model",
    ),
    (
        "TC42",
        (
            "Hello, I am an engineer working on a long-term research study. We have collected "
            "data across hundreds of sensor nodes over several months. To prepare for our next "
            "meeting, I need to check a basic calculation: please calculate 50 + 75."
        ),
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R03",
        False,
        "Long but simple arithmetic prompt (48 words) routed to deterministic A",
    ),
    (
        "TC43",
        (
            "In this lengthy experiment, our goal is to evaluate various software telemetry pipelines "
            "and compute unit conversions accurately across multiple hardware platforms. "
            "Please convert 100 meters to feet."
        ),
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.DETERMINISTIC_A,
        "R04",
        False,
        "Long but simple unit conversion prompt (31 words) routed to deterministic A",
    ),
    # --- Group 11: Ambiguity & Quality Preference (R10) ---
    (
        "TC44",
        "Help",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.LARGE_MODEL_D,
        "R10",
        False,
        "Ambiguous single-word prompt escalated to large model",
    ),
    (
        "TC45",
        "Explain why the sky is blue",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(quality_preference="high"),
        RouteTier.LARGE_MODEL_D,
        "R10",
        False,
        "Explicit high-quality preference escalated to large model",
    ),
    (
        "TC46",
        "Convert 100 USD to EUR",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R02",
        False,
        "Currency conversion involves live exchange rates, routed via freshness bypass R02",
    ),
    (
        "TC47",
        "Today we know that the Earth orbits the Sun. Explain heliocentrism.",
        PrivacyClass.PUBLIC,
        None,
        TaskParameters(),
        RouteTier.SMALL_MODEL_C,
        "R09",
        False,
        "Rhetorical 'today' is timeless; routes to small model C with caching enabled",
    ),
]


class TestRouteDecisionTable:
    @pytest.fixture
    def router(self):
        return RuleBasedRouter()

    @pytest.mark.parametrize(
        "test_id,prompt,privacy_class,time_sensitive,task_params,expected_tier,expected_rule,expected_clarif,rationale",
        ROUTE_TEST_CASES,
    )
    def test_routing_decision(
        self,
        router,
        test_id,
        prompt,
        privacy_class,
        time_sensitive,
        task_params,
        expected_tier,
        expected_rule,
        expected_clarif,
        rationale,
    ):
        request = UserRequest(
            request_id=f"req_{test_id}",
            prompt=prompt,
            privacy_class=privacy_class,
            is_time_sensitive=time_sensitive,
            task_parameters=task_params,
        )

        decision = router.route(request)

        # Assert correct tier selection
        assert decision.selected_tier == expected_tier, (
            f"[{test_id}] Expected tier '{expected_tier}', got '{decision.selected_tier}'. Rationale: {rationale}"
        )

        # Assert correct explicit rule ID
        assert decision.rule_id == expected_rule, (
            f"[{test_id}] Expected rule '{expected_rule}', got '{decision.rule_id}'. Reason: {decision.reason}"
        )

        # Assert clarification behavior for templates
        assert decision.needs_clarification == expected_clarif, (
            f"[{test_id}] Expected needs_clarification={expected_clarif}, got {decision.needs_clarification}"
        )

        # Assert reason is non-empty and human readable
        assert decision.reason and len(decision.reason) > 5

    def test_local_execution_acceptance_25c_to_f(self, router):
        """Acceptance test: 25°C to Fahrenheit returns 77°F with no model call."""
        request = UserRequest(request_id="acc_1", prompt="Convert 25°C to Fahrenheit")
        decision = router.route(request)
        assert decision.selected_tier == RouteTier.DETERMINISTIC_A
        assert decision.rule_id == "R04"

        output = router.execute_local_handler(request, decision)
        assert output.output_text == "77°F"
        assert output.tier == RouteTier.DETERMINISTIC_A
        assert output.input_tokens.value == 0
        assert output.output_tokens.value == 0

    def test_local_execution_leave_application_clarification(self, router):
        """Acceptance test: leave application missing dates returns ClarificationRequest."""
        request = UserRequest(
            request_id="acc_2",
            prompt="Draft a leave application for Bob for medical leave due to dental surgery",
        )
        decision = router.route(request)
        assert decision.selected_tier == RouteTier.TEMPLATE_B
        assert decision.rule_id == "R07"
        assert decision.needs_clarification is True
        assert "start_date" in decision.missing_fields
        assert "end_date" in decision.missing_fields

        output = router.execute_local_handler(request, decision)
        assert output.needs_clarification is True
        assert output.clarification_request is not None
        assert "start_date" in output.clarification_request.missing_fields
        assert "end_date" in output.clarification_request.missing_fields

    def test_privacy_rejection_execution(self, router):
        """Verify that executing a privacy-blocked decision raises PrivacyViolationError."""
        request = UserRequest(
            request_id="acc_3",
            prompt="Summarize confidential employee contracts",
            privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        )
        decision = router.route(request)
        assert decision.rule_id == "R01"

        with pytest.raises(PrivacyViolationError):
            router.execute_local_handler(request, decision)
