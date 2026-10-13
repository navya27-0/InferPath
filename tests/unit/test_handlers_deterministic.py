"""Unit tests for deterministic handlers: calculator, unit converter, date formatter, json validator."""

import pytest

from inferpath.handlers.calculator import CalculatorError, SafeCalculator
from inferpath.handlers.date_formatter import DateFormatter, DateFormattingError
from inferpath.handlers.json_validator import JSONSchemaValidator
from inferpath.handlers.unit_converter import UnitConversionError, UnitConverter
from inferpath.models.enums import ProvenanceLabel, RouteTier


class TestSafeCalculator:
    def test_basic_arithmetic(self):
        assert SafeCalculator.evaluate("2 + 2") == 4
        assert SafeCalculator.evaluate("10 - 3") == 7
        assert SafeCalculator.evaluate("6 * 7") == 42
        assert SafeCalculator.evaluate("100 / 4") == 25
        assert SafeCalculator.evaluate("7 // 2") == 3
        assert SafeCalculator.evaluate("10 % 3") == 1
        assert SafeCalculator.evaluate("2 ** 5") == 32
        assert SafeCalculator.evaluate("2 ^ 5") == 32

    def test_precedence_and_parentheses(self):
        assert SafeCalculator.evaluate("2 + 3 * 4") == 14
        assert SafeCalculator.evaluate("(2 + 3) * 4") == 20
        assert SafeCalculator.evaluate("((10 - 2) * 3) / 4") == 6

    def test_rejection_code_injection(self):
        unsafe_payloads = [
            "__import__('os').system('ls')",
            "eval('2 + 2')",
            "exec('x = 1')",
            "open('/etc/passwd').read()",
            "''.__class__.__mro__[1].__subclasses__()",
            "print('hello')",
            "len([1, 2, 3])",
            "x + 5",
            "[x for x in range(10)]",
            "lambda x: x + 1",
        ]
        for payload in unsafe_payloads:
            with pytest.raises(CalculatorError):
                SafeCalculator.evaluate(payload)

    def test_division_by_zero(self):
        with pytest.raises(CalculatorError, match="Division or modulo by zero"):
            SafeCalculator.evaluate("10 / 0")
        with pytest.raises(CalculatorError, match="Division or modulo by zero"):
            SafeCalculator.evaluate("10 // 0")
        with pytest.raises(CalculatorError, match="Division or modulo by zero"):
            SafeCalculator.evaluate("10 % 0")

    def test_computational_limits(self):
        # Giant exponent rejected
        with pytest.raises(CalculatorError, match="Exponent .* exceeds maximum"):
            SafeCalculator.evaluate("2 ** 1000000")

        # Huge string length rejected
        long_expr = "1 + " * 300 + "1"
        with pytest.raises(CalculatorError, match="exceeds limit"):
            SafeCalculator.evaluate(long_expr)

    def test_execute_handler_output_provenance(self):
        output = SafeCalculator.execute_handler("What is 15 * 4 + 10?")
        assert output.output_text == "70"
        assert output.tier == RouteTier.DETERMINISTIC_A
        assert output.input_tokens.value == 0
        assert output.input_tokens.provenance == ProvenanceLabel.MEASURED
        assert output.output_tokens.value == 0
        assert output.output_tokens.provenance == ProvenanceLabel.MEASURED
        assert output.latency_ms.provenance == ProvenanceLabel.MEASURED


class TestUnitConverter:
    def test_acceptance_criteria_temperature(self):
        """Acceptance test: 25°C to Fahrenheit returns 77°F without invoking either mock model."""
        res_num, res_str = UnitConverter.convert(25, "°C", "Fahrenheit")
        assert res_num == 77.0
        assert res_str == "77°F"

        output = UnitConverter.execute_handler("Convert 25°C to Fahrenheit")
        assert output.output_text == "77°F"
        assert output.tier == RouteTier.DETERMINISTIC_A
        assert output.input_tokens.value == 0

    def test_temperature_conversions(self):
        assert UnitConverter.convert(0, "C", "F")[0] == 32.0
        assert UnitConverter.convert(100, "C", "F")[0] == 212.0
        assert UnitConverter.convert(32, "F", "C")[0] == 0.0
        assert UnitConverter.convert(0, "C", "K")[0] == 273.15
        assert UnitConverter.convert(273.15, "K", "C")[0] == 0.0

    def test_length_conversions(self):
        res, _ = UnitConverter.convert(1, "km", "m")
        assert res == 1000.0

        res_mi, _ = UnitConverter.convert(1, "mi", "km")
        assert pytest.approx(res_mi, 0.001) == 1.609344

        res_ft, _ = UnitConverter.convert(1, "m", "feet")
        assert pytest.approx(res_ft, 0.01) == 3.28084

    def test_mass_conversions(self):
        res, _ = UnitConverter.convert(1, "kg", "g")
        assert res == 1000.0

        res_lb, _ = UnitConverter.convert(1, "kg", "lb")
        assert pytest.approx(res_lb, 0.01) == 2.20462

    def test_incompatible_dimensions(self):
        with pytest.raises(UnitConversionError, match="Incompatible dimensions"):
            UnitConverter.convert(10, "meters", "kilograms")

        with pytest.raises(UnitConversionError, match="Incompatible dimensions"):
            UnitConverter.convert(25, "celsius", "feet")

    def test_currency_excluded(self):
        with pytest.raises(UnitConversionError, match="Currency conversion .* explicitly excluded"):
            UnitConverter.convert(100, "USD", "EUR")


class TestDateFormatter:
    def test_parse_and_format_iso(self):
        parsed = DateFormatter.parse_date("2026-10-13")
        assert parsed.year == 2026
        assert parsed.month == 10
        assert parsed.day == 13

        res_rfc = DateFormatter.format_date(parsed, "rfc2822")
        assert "13 Oct 2026" in res_rfc

        res_human = DateFormatter.format_date(parsed, "human_long")
        assert res_human == "October 13, 2026"

    def test_execute_handler(self):
        output = DateFormatter.execute_handler("Format 2026-10-13 as human_long")
        assert output.output_text == "October 13, 2026"
        assert output.tier == RouteTier.DETERMINISTIC_A
        assert output.input_tokens.value == 0

    def test_invalid_calendar_date(self):
        with pytest.raises(DateFormattingError):
            DateFormatter.parse_date("2026-02-30")

    def test_dynamic_date_rejected(self):
        with pytest.raises(DateFormattingError, match="require dynamic evaluation"):
            DateFormatter.parse_date("today")


class TestJSONSchemaValidator:
    def test_valid_json_without_schema(self):
        outcome = JSONSchemaValidator.validate_json_string('{"name": "Alice", "age": 30}')
        assert outcome["valid"] is True
        assert outcome["schema_validated"] is False

    def test_valid_json_with_schema(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }
        outcome = JSONSchemaValidator.validate_json_string(
            '{"name": "Alice", "age": 30}', schema=schema
        )
        assert outcome["valid"] is True
        assert outcome["schema_validated"] is True

    def test_invalid_json_schema_mismatch(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        outcome = JSONSchemaValidator.validate_json_string('{"other": 123}', schema=schema)
        assert outcome["valid"] is False
        assert outcome["schema_validated"] is True
        assert "required" in outcome["error"]

    def test_malformed_json_syntax(self):
        outcome = JSONSchemaValidator.validate_json_string('{"name": "Alice",')
        assert outcome["valid"] is False
        assert "Invalid JSON syntax" in outcome["error"]
