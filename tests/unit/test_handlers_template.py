"""Unit tests for Jinja2 template handlers, slot validation, and clarification requests."""

import pytest

from inferpath.models.enums import RouteTier
from inferpath.templates.engine import TemplateEngine


class TestTemplateHandlers:
    @pytest.fixture
    def engine(self):
        return TemplateEngine()

    def test_acceptance_criteria_leave_application_missing_dates(self, engine):
        """Acceptance test: a leave application without start and end dates asks for clarification explicitly."""
        fields = {
            "employee_name": "Bob",
            "leave_type": "sick leave",
            "reason": "dental surgery",
        }
        output = engine.execute_handler("leave_application", fields)

        assert output.needs_clarification is True
        assert output.validated is False
        assert output.clarification_request is not None
        missing = output.clarification_request.missing_fields
        assert "start_date" in missing
        assert "end_date" in missing
        assert "start_date" in output.output_text
        assert "end_date" in output.output_text

    def test_complete_leave_application(self, engine):
        fields = {
            "employee_name": "Alice",
            "leave_type": "annual leave",
            "start_date": "2026-11-01",
            "end_date": "2026-11-05",
            "reason": "family vacation",
            "supervisor_name": "Dr. Smith",
        }
        output = engine.execute_handler("leave_application", fields)

        assert output.needs_clarification is False
        assert output.validated is True
        assert "LEAVE APPLICATION" in output.output_text
        assert "Alice" in output.output_text
        assert "2026-11-01" in output.output_text
        assert "2026-11-05" in output.output_text
        assert "Dr. Smith" in output.output_text
        assert output.tier == RouteTier.TEMPLATE_B

    def test_complete_formal_email(self, engine):
        fields = {
            "recipient_name": "Mr. Davis",
            "subject": "Q3 Budget Review",
            "purpose": "discussing the allocation of engineering resources",
            "sender_name": "Carol",
            "call_to_action": "Please let me know if Thursday works for you.",
        }
        output = engine.execute_handler("formal_email", fields)

        assert output.needs_clarification is False
        assert "Subject: Q3 Budget Review" in output.output_text
        assert "Mr. Davis" in output.output_text
        assert "Carol" in output.output_text

    def test_missing_formal_email_fields(self, engine):
        fields = {
            "recipient_name": "Mr. Davis",
            # subject, purpose, sender_name missing
        }
        output = engine.execute_handler("formal_email", fields)
        assert output.needs_clarification is True
        assert "subject" in output.clarification_request.missing_fields
        assert "purpose" in output.clarification_request.missing_fields
        assert "sender_name" in output.clarification_request.missing_fields

    def test_meeting_minutes_complete_and_missing(self, engine):
        # Missing fields
        out_missing = engine.execute_handler(
            "meeting_minutes", {"meeting_title": "Sprint Planning"}
        )
        assert out_missing.needs_clarification is True
        assert "date" in out_missing.clarification_request.missing_fields
        assert "attendees" in out_missing.clarification_request.missing_fields

        # Complete
        fields = {
            "meeting_title": "Sprint Planning",
            "date": "2026-10-13",
            "attendees": ["Alice", "Bob"],
            "agenda": ["Review backlog", "Estimate points"],
            "key_decisions": ["Prioritize cache engine"],
        }
        out_complete = engine.execute_handler("meeting_minutes", fields)
        assert out_complete.needs_clarification is False
        assert "Sprint Planning" in out_complete.output_text
        assert "Alice" in out_complete.output_text

    def test_cover_letter_complete_and_missing(self, engine):
        # Missing
        out_missing = engine.execute_handler("cover_letter", {"applicant_name": "David"})
        assert out_missing.needs_clarification is True
        assert "job_title" in out_missing.clarification_request.missing_fields
        assert "company_name" in out_missing.clarification_request.missing_fields

        # Complete
        fields = {
            "applicant_name": "David",
            "job_title": "Senior AI Engineer",
            "company_name": "DeepMind",
            "relevant_experience": "10 years of experience building autonomous systems.",
        }
        out_complete = engine.execute_handler("cover_letter", fields)
        assert out_complete.needs_clarification is False
        assert "Senior AI Engineer" in out_complete.output_text
        assert "DeepMind" in out_complete.output_text

    def test_template_injection_sandbox_safety(self, engine):
        """Verify that attempts to escape the Jinja2 sandbox are safely mitigated."""
        fields = {
            "employee_name": "{{ ''.__class__.__mro__[1].__subclasses__() }}",
            "leave_type": "sick leave",
            "start_date": "2026-11-01",
            "end_date": "2026-11-02",
            "reason": "medical",
        }
        output = engine.execute_handler("leave_application", fields)
        # In sandboxed environment with autoescaping, literal text is safely rendered and never evaluated as code
        assert output.needs_clarification is False
        assert "__subclasses__" in output.output_text
