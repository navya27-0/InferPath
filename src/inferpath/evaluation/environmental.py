"""Environmental and resource estimation module for InferPath research evaluation.

Calculates Scope 2 computing energy and water footprint estimates with:
- Transparent low / central / high uncertainty ranges.
- Explicit system boundaries for datacenter operational energy and facility cooling water.
- Separate accounting for direct facility cooling water (WUE) vs indirect power generation water footprint.
- Strict ProvenanceLabel.ESTIMATED labeling; never presented as measured.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from inferpath.config.settings import CarbonBoundConfig, load_carbon_coefficients
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import BoundaryDefinition, ProvenanceValue, RangeBounds


class EnvironmentalEstimate(BaseModel):
    """Complete environmental resource profile for a single request or aggregated workload."""

    # Energy estimate in Watt-hours (Wh)
    energy_wh: ProvenanceValue[float] = Field(
        description="Operational compute electricity consumed in Watt-hours"
    )
    energy_range_wh: RangeBounds[float]

    # Water breakdown in milliliters (mL)
    direct_cooling_water_ml: ProvenanceValue[float] = Field(
        description="Direct facility evaporative cooling water consumed on-site (WUE)"
    )
    indirect_power_water_ml: ProvenanceValue[float] = Field(
        description="Indirect water footprint consumed during regional electricity generation"
    )
    total_water_ml: ProvenanceValue[float] = Field(
        description="Total combined operational water footprint in mL"
    )
    water_range_ml: RangeBounds[float]

    # Boundaries and assumptions
    pue_used: float
    carbon_config_date: str
    verification_status: str
    limitations: str


class EnvironmentalEstimator:
    """Estimates energy and water consumption from execution telemetry and carbon coefficients."""

    def __init__(self, config: CarbonBoundConfig | None = None, config_path: Path | None = None):
        self.config = config or load_carbon_coefficients(config_path)

        self.energy_boundary = BoundaryDefinition(
            included=self.config.system_boundary_energy.included,
            excluded=self.config.system_boundary_energy.excluded,
            notes=self.config.system_boundary_energy.notes,
        )

        self.direct_water_boundary = BoundaryDefinition(
            included="Direct on-site evaporative cooling water consumed at the datacenter facility (WUE).",
            excluded="Upstream power plant cooling water, fuel extraction, hardware manufacturing embodied water.",
            notes="Separately accounted from indirect power generation water.",
        )

        self.indirect_water_boundary = BoundaryDefinition(
            included="Indirect freshwater consumed during thermal/hydro electricity generation powering compute.",
            excluded="On-site cooling towers, supply chain manufacturing water.",
            notes="Derived from regional grid water factor heuristics.",
        )

        self.total_water_boundary = BoundaryDefinition(
            included=self.config.system_boundary_water.included,
            excluded=self.config.system_boundary_water.excluded,
            notes=self.config.system_boundary_water.notes,
        )

    def estimate(
        self,
        tier: RouteTier | str,
        model_id: str | None,
        total_tokens: int,
        was_cached: bool = False,
    ) -> EnvironmentalEstimate:
        """Computes low, central, and high environmental estimates for a query execution."""
        tier_str = tier.value if isinstance(tier, RouteTier) else str(tier)

        if was_cached:
            # Cache lookup energy is minimal (~0.000001 Wh for SQLite seek)
            energy_central = 0.000002
            energy_low = 0.000001
            energy_high = 0.000005
        elif tier_str == RouteTier.DETERMINISTIC_A.value:
            coeff = self.config.coefficients_per_1000_tokens.get("deterministic_a")
            rate_low = coeff.low if coeff else 0.000001
            rate_cent = coeff.central if coeff else 0.000005
            rate_high = coeff.high if coeff else 0.00001
            tokens_k = max(1, total_tokens) / 1000.0
            pue = self.config.datacenter_pue.central
            energy_low = tokens_k * rate_low * pue
            energy_central = tokens_k * rate_cent * pue
            energy_high = tokens_k * rate_high * self.config.datacenter_pue.high
        elif tier_str == RouteTier.TEMPLATE_B.value:
            coeff = self.config.coefficients_per_1000_tokens.get("template_b")
            rate_low = coeff.low if coeff else 0.000005
            rate_cent = coeff.central if coeff else 0.00001
            rate_high = coeff.high if coeff else 0.00002
            tokens_k = max(1, total_tokens) / 1000.0
            pue = self.config.datacenter_pue.central
            energy_low = tokens_k * rate_low * pue
            energy_central = tokens_k * rate_cent * pue
            energy_high = tokens_k * rate_high * self.config.datacenter_pue.high
        else:
            # Model tier (Small C or Large D)
            lookup_key = model_id or (
                "mock-small" if tier_str == RouteTier.SMALL_MODEL_C.value else "mock-large"
            )
            coeff = self.config.coefficients_per_1000_tokens.get(lookup_key)
            if not coeff:
                # Default heuristic based on tier
                if tier_str == RouteTier.SMALL_MODEL_C.value:
                    coeff = self.config.coefficients_per_1000_tokens.get("mock-small")
                else:
                    coeff = self.config.coefficients_per_1000_tokens.get("mock-large")

            rate_low = coeff.low if coeff else 0.10
            rate_cent = coeff.central if coeff else 0.30
            rate_high = coeff.high if coeff else 0.80

            tokens_k = max(1, total_tokens) / 1000.0
            pue_low = self.config.datacenter_pue.low
            pue_cent = self.config.datacenter_pue.central
            pue_high = self.config.datacenter_pue.high

            energy_low = tokens_k * rate_low * pue_low
            energy_central = tokens_k * rate_cent * pue_cent
            energy_high = tokens_k * rate_high * pue_high

        # Water calculation:
        # Separate direct on-site cooling water (~58% of total water intensity)
        # from indirect power generation water (~42% of total water intensity)
        water_int_cent = self.config.water_intensity.central
        water_int_low = self.config.water_intensity.low
        water_int_high = self.config.water_intensity.high

        # 58% direct facility cooling
        direct_water_central = energy_central * (water_int_cent * 0.58)
        direct_water_low = energy_low * (water_int_low * 0.58)
        direct_water_high = energy_high * (water_int_high * 0.58)

        # 42% indirect power generation
        indirect_water_central = energy_central * (water_int_cent * 0.42)
        indirect_water_low = energy_low * (water_int_low * 0.42)
        indirect_water_high = energy_high * (water_int_high * 0.42)

        total_water_central = direct_water_central + indirect_water_central
        total_water_low = direct_water_low + indirect_water_low
        total_water_high = direct_water_high + indirect_water_high

        return EnvironmentalEstimate(
            energy_wh=ProvenanceValue(
                value=round(energy_central, 6),
                provenance=ProvenanceLabel.ESTIMATED,
                unit="Wh",
                system_boundary=self.energy_boundary,
                range_bounds=RangeBounds(
                    low=round(energy_low, 6),
                    central=round(energy_central, 6),
                    high=round(energy_high, 6),
                ),
            ),
            energy_range_wh=RangeBounds(
                low=round(energy_low, 6),
                central=round(energy_central, 6),
                high=round(energy_high, 6),
            ),
            direct_cooling_water_ml=ProvenanceValue(
                value=round(direct_water_central, 6),
                provenance=ProvenanceLabel.ESTIMATED,
                unit="mL",
                system_boundary=self.direct_water_boundary,
                range_bounds=RangeBounds(
                    low=round(direct_water_low, 6),
                    central=round(direct_water_central, 6),
                    high=round(direct_water_high, 6),
                ),
            ),
            indirect_power_water_ml=ProvenanceValue(
                value=round(indirect_water_central, 6),
                provenance=ProvenanceLabel.ESTIMATED,
                unit="mL",
                system_boundary=self.indirect_water_boundary,
                range_bounds=RangeBounds(
                    low=round(indirect_water_low, 6),
                    central=round(indirect_water_central, 6),
                    high=round(indirect_water_high, 6),
                ),
            ),
            total_water_ml=ProvenanceValue(
                value=round(total_water_central, 6),
                provenance=ProvenanceLabel.ESTIMATED,
                unit="mL",
                system_boundary=self.total_water_boundary,
                range_bounds=RangeBounds(
                    low=round(total_water_low, 6),
                    central=round(total_water_central, 6),
                    high=round(total_water_high, 6),
                ),
            ),
            water_range_ml=RangeBounds(
                low=round(total_water_low, 6),
                central=round(total_water_central, 6),
                high=round(total_water_high, 6),
            ),
            pue_used=self.config.datacenter_pue.central,
            carbon_config_date=self.config.effective_date,
            verification_status=self.config.verification_status,
            limitations=self.config.limitations,
        )
