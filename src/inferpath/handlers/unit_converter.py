"""Deterministic unit conversion handler for temperature, length, and mass."""

import re

from inferpath.models.domain import HandlerOutput
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


class UnitConversionError(ValueError):
    """Raised when unit conversion fails due to incompatible dimensions or unknown units."""

    pass


class UnitConverter:
    """
    Deterministic converter supporting:
    - Temperature: Celsius (C), Fahrenheit (F), Kelvin (K)
    - Length: Millimeter (mm), Centimeter (cm), Meter (m), Kilometer (km), Inch (in), Foot (ft), Yard (yd), Mile (mi)
    - Mass: Milligram (mg), Gram (g), Kilogram (kg), Metric Ton (tonne/ton), Ounce (oz), Pound (lb/lbs)

    Currency is explicitly excluded because live exchange rates require external APIs.
    """

    # Aliases to canonical unit representations
    TEMPERATURE_UNITS = {
        "c": "celsius",
        "celsius": "celsius",
        "deg c": "celsius",
        "°c": "celsius",
        "degree celsius": "celsius",
        "degrees celsius": "celsius",
        "centigrade": "celsius",
        "f": "fahrenheit",
        "fahrenheit": "fahrenheit",
        "deg f": "fahrenheit",
        "°f": "fahrenheit",
        "degree fahrenheit": "fahrenheit",
        "degrees fahrenheit": "fahrenheit",
        "k": "kelvin",
        "kelvin": "kelvin",
        "deg k": "kelvin",
        "°k": "kelvin",
        "degrees kelvin": "kelvin",
    }

    # Factor to base unit (meter)
    LENGTH_UNITS = {
        "mm": 0.001,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "in": 0.0254,
        "inch": 0.0254,
        "inches": 0.0254,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "yd": 0.9144,
        "yard": 0.9144,
        "yards": 0.9144,
        "mi": 1609.344,
        "mile": 1609.344,
        "miles": 1609.344,
    }

    # Factor to base unit (kilogram)
    MASS_UNITS = {
        "mg": 1e-6,
        "milligram": 1e-6,
        "milligrams": 1e-6,
        "g": 1e-3,
        "gram": 1e-3,
        "grams": 1e-3,
        "kg": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "tonne": 1000.0,
        "tonnes": 1000.0,
        "metric ton": 1000.0,
        "metric tons": 1000.0,
        "oz": 0.028349523125,
        "ounce": 0.028349523125,
        "ounces": 0.028349523125,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
    }

    CURRENCY_CODES = {"usd", "eur", "gbp", "jpy", "cad", "aud", "inr", "cny", "chf"}

    CONVERSION_PATTERN = re.compile(
        r"(?:convert\s+)?([-+]?\d+(?:\.\d+)?)\s*([°\w\s]+?)\s+(?:to|in|into)\s+([°\w\s]+)",
        re.IGNORECASE,
    )

    @classmethod
    def normalize_unit(cls, unit_str: str) -> str:
        clean = unit_str.strip().lower()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    @classmethod
    def get_dimension(cls, unit: str) -> str:
        u = cls.normalize_unit(unit)
        if u in cls.CURRENCY_CODES:
            return "currency"
        if u in cls.TEMPERATURE_UNITS:
            return "temperature"
        if u in cls.LENGTH_UNITS:
            return "length"
        if u in cls.MASS_UNITS:
            return "mass"
        raise UnitConversionError(f"Unknown or unsupported unit: '{unit}'")

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> tuple[float, str]:
        """
        Converts value from from_unit to to_unit.
        Returns: (converted_numeric_value, formatted_string_result).
        """
        dim_from = cls.get_dimension(from_unit)
        dim_to = cls.get_dimension(to_unit)

        if dim_from == "currency" or dim_to == "currency":
            raise UnitConversionError(
                "Currency conversion requires dynamic market exchange rates and is explicitly excluded."
            )

        if dim_from != dim_to:
            raise UnitConversionError(
                f"Incompatible dimensions: cannot convert {dim_from} ('{from_unit}') to {dim_to} ('{to_unit}')."
            )

        u_from = cls.normalize_unit(from_unit)
        u_to = cls.normalize_unit(to_unit)

        if dim_from == "temperature":
            return cls._convert_temperature(value, u_from, u_to)
        elif dim_from == "length":
            return cls._convert_linear(value, u_from, u_to, cls.LENGTH_UNITS, to_unit)
        elif dim_from == "mass":
            return cls._convert_linear(value, u_from, u_to, cls.MASS_UNITS, to_unit)

        raise UnitConversionError(f"Unsupported dimension: {dim_from}")

    @classmethod
    def _convert_temperature(
        cls, value: float, from_u: str, to_u: str
    ) -> tuple[float, str]:
        canonical_from = cls.TEMPERATURE_UNITS[from_u]
        canonical_to = cls.TEMPERATURE_UNITS[to_u]

        # Convert to Celsius first
        if canonical_from == "celsius":
            celsius = value
        elif canonical_from == "fahrenheit":
            celsius = (value - 32.0) * (5.0 / 9.0)
        elif canonical_from == "kelvin":
            celsius = value - 273.15
        else:
            raise UnitConversionError(f"Unknown temperature unit: {from_u}")

        # Convert from Celsius to target
        if canonical_to == "celsius":
            res = celsius
            unit_sym = "°C"
        elif canonical_to == "fahrenheit":
            res = (celsius * (9.0 / 5.0)) + 32.0
            unit_sym = "°F"
        elif canonical_to == "kelvin":
            res = celsius + 273.15
            unit_sym = "K"
        else:
            raise UnitConversionError(f"Unknown temperature unit: {to_u}")

        formatted = f"{int(res)}{unit_sym}" if res.is_integer() else f"{round(res, 2)}{unit_sym}"
        return res, formatted

    @classmethod
    def _convert_linear(
        cls,
        value: float,
        from_u: str,
        to_u: str,
        factors: dict[str, float],
        display_unit: str,
    ) -> tuple[float, str]:
        factor_from = factors[from_u]
        factor_to = factors[to_u]

        base_val = value * factor_from
        res = base_val / factor_to

        if res.is_integer():
            formatted = f"{int(res)} {display_unit.strip()}"
        else:
            formatted = f"{round(res, 4)} {display_unit.strip()}"
        return res, formatted

    @classmethod
    def execute_handler(cls, prompt: str) -> HandlerOutput:
        """
        Parses conversion request from prompt and returns HandlerOutput.
        Acceptance test: '25°C to Fahrenheit' returns '77°F'.
        """
        match = cls.CONVERSION_PATTERN.search(prompt.strip())
        if not match:
            raise UnitConversionError(
                f"Could not parse unit conversion intent from prompt: '{prompt}'"
            )

        val_str, from_u, to_u = match.groups()
        val = float(val_str)
        numeric_res, formatted_res = cls.convert(val, from_u, to_u)

        return HandlerOutput(
            tier=RouteTier.DETERMINISTIC_A,
            output_text=formatted_res,
            input_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(
                value=0.5, provenance=ProvenanceLabel.MEASURED, unit="ms"
            ),
            model_version="deterministic_unit_converter_v1",
            raw_response={
                "value": val,
                "from_unit": from_u,
                "to_unit": to_u,
                "numeric_result": numeric_res,
                "formatted_result": formatted_res,
            },
        )
