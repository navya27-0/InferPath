"""Deterministic mock model adapters for offline testing and simulated execution."""

import hashlib
from typing import Any, Protocol, runtime_checkable

from inferpath.models.domain import ModelResult
from inferpath.models.enums import ProvenanceLabel


@runtime_checkable
class ModelAdapter(Protocol):
    """Abstract protocol for model execution adapters (Prompt 1 specification)."""

    model_id: str
    price_table_date: str

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Generates a model completion synchronously."""
        ...

    async def generate_async(self, prompt: str, **params: Any) -> ModelResult:
        """Generates a model completion asynchronously."""
        ...


class BaseMockModel:
    """Base class providing deterministic canned generation and simulated provenance."""

    def __init__(
        self,
        model_id: str,
        prompt_price_per_1m: float,
        completion_price_per_1m: float,
        simulated_base_latency_ms: float,
        price_table_date: str = "2026-10-01",
    ):
        self.model_id = model_id
        self.prompt_price_per_1m = prompt_price_per_1m
        self.completion_price_per_1m = completion_price_per_1m
        self.simulated_base_latency_ms = simulated_base_latency_ms
        self.price_table_date = price_table_date

    def _determine_output_text(self, prompt: str) -> str:
        """Produces deterministic domain synthesis matching benchmark expectations."""
        clean = prompt.strip().lower()

        # 1. Arithmetic & Unit Conversions
        if "125 * 8 + 450" in clean or ("125" in clean and "8" in clean and "450" in clean):
            return "The calculated result is 1450."
        if "15%" in clean and ("84" in clean or "84.00" in clean):
            return "A 15% tip on a bill of $84.00 is $12.60."
        if "100 km/h" in clean or "km/h to mph" in clean or "kilometers per hour" in clean:
            return "100 km/h converts to 62.14 mph."
        if "25" in clean and "fahrenheit" in clean:
            return "25°C is equal to 77°F."
        if "yen" in clean:
            return "The live exchange rate is approximately 152 Japanese Yen per 1 USD."
        if "usd to eur" in clean or ("exchange rate" in clean and "eur" in clean):
            return "The current market exchange rate is approximately 0.92 EUR per 1 USD."
        if "layover buffer" in clean or ("14:15" in clean and "16:45" in clean):
            return "The remaining layover buffer is 75.0 minutes after 45 minutes for customs and terminal transfer."
        if "100,000" in clean and "budget" in clean:
            return "Budget allocation: A: $45,000, B: $22,500, C: $32,500."

        # 2. Classification
        if "intent" in clean:
            if "json" in clean:
                return '{"intent": "BILLING"}'
            if "billing" in clean or "invoice" in clean or "charge" in clean:
                return "BILLING"
            if "technical" in clean or "error" in clean or "crash" in clean:
                return "TECHNICAL"
            return "GENERAL"
        if "spam" in clean:
            if "json" in clean:
                return '{"label": "SPAM", "confidence": 0.95}'
            return "SPAM"
        if "sentiment" in clean:
            if "json" in clean:
                return '{"sentiment": "POSITIVE"}'
            if any(
                w in clean for w in ("breathtaking", "flawlessly", "love", "excellent", "great")
            ):
                return "POSITIVE"
            if any(w in clean for w in ("terrible", "awful", "broken", "worst", "hate")):
                return "NEGATIVE"
            return "NEUTRAL"
        if "urgency" in clean or "ticket urgency" in clean or "priority" in clean:
            return "P1_CRITICAL"

        # 3. Short Explanations
        if "404" in clean and ("http" in clean or "error" in clean or "web" in clean):
            return "HTTP status code 404 indicates that the requested web page or resource was Not Found on the server."
        if "dns" in clean and ("domain" in clean or "server" in clean):
            return "A Domain Name System (DNS) server translates human-readable domain names into machine-routable IP addresses."
        if "nitrogen" in clean and ("boiling" in clean or "temperature" in clean):
            return "The boiling point of liquid nitrogen at standard atmospheric pressure is -195.79°C (77.36 K)."
        if "photosynthesis" in clean or ("plants" in clean and "sunlight" in clean):
            return "Photosynthesis is the process by which green plants convert sunlight and water into chemical energy in the form of glucose while releasing oxygen."
        if "binary search" in clean:
            return "Binary search is an efficient search algorithm on sorted arrays with O(log n) time complexity, repeatedly dividing the search space in half."
        if "heliocentrism" in clean:
            return "Heliocentrism is the astronomical model where the Earth and planets revolve around the Sun at the center of the solar system."
        if "sky is blue" in clean:
            return "The sky appears blue because short blue wavelengths of sunlight undergo Rayleigh scattering by gas molecules in Earth's atmosphere."
        if "france" in clean and "capital" in clean:
            return "The capital of France is Paris."

        # 4. Rewriting
        if (
            "passive to active" in clean
            or "financial statements" in clean
            or "audit committee" in clean
        ):
            return "The audit committee reviewed the quarterly financial statements in detail."
        if "3 concise bullet points" in clean or "bullet points" in clean:
            return (
                "- Postgres connection pooling\n- API rate limiting\n- Database index optimization"
            )
        if (
            "formal tone" in clean
            or "make this professional" in clean
            or "push back the deadline" in clean
        ):
            return "Dear Team, we must postpone the project deadline because the recent build encountered unexpected delay."
        if "carlos smith" in clean:
            if "yaml" in clean:
                return "name: Carlos Smith\nage: 34\noccupation: Software Engineer"
            return '{"name": "Carlos Smith", "age": 34, "occupation": "Software Engineer"}'

        # 5. Summarization & Reasoning
        if "quarterly earnings" in clean or "initech" in clean:
            if "json" in clean:
                return '{"revenue": "$14.2M", "margin": "21%", "churn": "1.2%"}'
            return "Initech Corporation recorded $14.2M in Q3 revenue with a 21% operating margin driven by enterprise cloud services. Operating cash flow grew steadily."
        if "climate" in clean or "net-zero" in clean or "environmental policy" in clean:
            return "The environmental policy mandates commercial buildings to achieve net-zero emissions by 2030 through renewable energy investments."
        if "outage" in clean or "post-mortem" in clean:
            return "The primary root cause of the outage was BGP route misconfiguration causing blackholing of gateway traffic."
        if (
            "user data rights" in clean
            or "terms of service" in clean
            or "content and deletion" in clean
        ):
            return "Users retain ownership rights over their submitted content and hold the right to delete content at any time."
        if "three boxes" in clean or ("apples" in clean and "oranges" in clean):
            return "The boxes are labeled Apples, Oranges, and Apples and Oranges. By selecting from the mislabeled Apples and Oranges box, we can deduce the exact fruit in each box."
        if "concurrency" in clean and "thread" in clean:
            return "Analysis: Threads 1, 2, and 3 create a race condition on shared variables without proper synchronization."

        # 6. Synthesis & Reasoning
        if "postgresql" in clean and "mongodb" in clean:
            return "# Data Model\nPostgreSQL is a relational database using structured SQL schemas, whereas MongoDB is a flexible document store.\n# Transactions\nPostgreSQL enforces strict ACID transactions, while MongoDB offers distributed transactional support.\n# Scalability\nPostgreSQL scales vertically with read-replicas, while MongoDB provides native horizontal sharding."
        if "keynes" in clean and "hayek" in clean:
            return "# Economic Theory Comparison\nKeynes advocates government demand-side intervention and fiscal stimulus during downturns, whereas Hayek emphasizes decentralized market mechanisms and the Austrian school price system."
        if "paxos" in clean and "raft" in clean:
            return "# Distributed Consensus Comparison\nPaxos employs symmetric peer consensus with high algorithmic complexity, whereas Raft implements decomposed leader-based consensus for understandable state machine replication."
        if "high-availability" in clean or "cloud architecture proposal" in clean:
            return "# High-Availability Cloud Architecture\nProposal: Deploy a multi-region infrastructure with automated load balancer distribution, database replication, and an in-memory cache cluster."
        if "user research findings" in clean or ("onboarding" in clean and "pricing" in clean):
            return "Synthesis: The research highlights friction in user onboarding, confusion over pricing tiers, and provides a clear recommendation for streamlined workflows."
        if "remote work" in clean:
            return "# Remote Work Synthesis\nEmpirical research highlights enhanced productivity and employee retention alongside communication friction and social isolation risks."
        if "square root of 2" in clean or "sqrt(2)" in clean:
            return "# Mathematical Proof: Irrationality of Square Root of 2\nStep 1: Assume sqrt(2) = a/b in lowest terms where a, b are integers.\nStep 2: 2 = a^2/b^2, which implies a^2 = 2b^2, so a is an even integer (a = 2k).\nStep 3: Substituting yields 4k^2 = 2b^2, hence b^2 = 2k^2, so b is also even.\nStep 4: This contradicts our assumption that a and b share no common factors. Therefore, sqrt(2) is irrational."

        # 7. Templated Documents
        if "acme corp" in clean or "kickoff" in clean:
            return "Subject: Project Kickoff Meeting\nDear Acme Corp Team,\nWe look forward to initiating our project kickoff on 2026-11-01.\nBest regards,\nInferPath Team"
        if "ticket #" in clean or "4082" in clean:
            return "Subject: Ticket Acknowledgement #4082\nDear Customer,\nWe have received your support request regarding billing for ticket #4082 and are actively investigating.\nSincerely,\nSupport Team"
        if "leave application" in clean or "leave request" in clean:
            return "LEAVE APPLICATION\nEmployee: John Doe\nDates: 2026-11-01 to 2026-11-05\nReason: family emergency\nLeave Type: emergency leave"
        if "meeting minutes" in clean or "sprint retro" in clean:
            return "MEETING MINUTES\nTitle: Sprint Retro\nDate: 2026-10-14\nAttendees: Alice, Bob\nDecision: adopt ruff linter\nAgenda: Discussion on Sprint Retro"

        # Fallback deterministic digest
        digest_suffix = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"[{self.model_id}] Deterministic simulated completion for request {digest_suffix}."

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Synchronously produces deterministic canned output tagged strictly as simulated."""
        output_text = self._determine_output_text(prompt)

        # Deterministic simulated token counts (approx 1 token per word + overhead)
        prompt_words = len(prompt.split())
        output_words = len(output_text.split())

        simulated_input_tokens = max(1, prompt_words + 4)
        simulated_output_tokens = max(1, output_words + 2)

        # Calculated cost from dated rates
        cost_usd = (simulated_input_tokens / 1_000_000.0) * self.prompt_price_per_1m + (
            simulated_output_tokens / 1_000_000.0
        ) * self.completion_price_per_1m

        # Deterministic simulated latency
        simulated_latency_ms = self.simulated_base_latency_ms + (simulated_output_tokens * 1.5)

        return ModelResult(
            text=output_text,
            input_tokens=simulated_input_tokens,
            output_tokens=simulated_output_tokens,
            latency_ms=round(simulated_latency_ms, 2),
            cost_usd_calculated=round(cost_usd, 6),
            provider_reported_cost_usd=None,
            price_table_date=self.price_table_date,
            source=ProvenanceLabel.SIMULATED,  # Strictly SIMULATED
            model_id=self.model_id,
        )

    async def generate_async(self, prompt: str, **params: Any) -> ModelResult:
        """Asynchronous execution wrapper returning deterministic output."""
        return self.generate(prompt, **params)


class MockSmallModel(BaseMockModel):
    """
    Mock adapter representing Tier C (Small Model).
    Low simulated latency, low token price, deterministic canned responses.
    Simulates realistic small model behavior: handles normal requests, but
    on complex multi-step proofs without headings, may omit headers to test fallback.
    Operates strictly offline without network access or API keys.
    """

    def __init__(
        self,
        model_id: str = "mock-small",
        prompt_price_per_1m: float = 0.05,
        completion_price_per_1m: float = 0.20,
        simulated_base_latency_ms: float = 20.0,
        price_table_date: str = "2026-10-01",
    ):
        super().__init__(
            model_id=model_id,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            simulated_base_latency_ms=simulated_base_latency_ms,
            price_table_date=price_table_date,
        )

    def _determine_output_text(self, prompt: str) -> str:
        clean = prompt.strip().lower()
        # On very complex synthesis tasks requiring headings, small model occasionally omits header
        # to trigger FormatValidator failure and test the bounded fallback escalation
        if "square root of 2" in clean:
            # Missing markdown heading to trigger format validator fallback
            return "Proof that square root of 2 is irrational: Assume a/b in lowest terms, then a^2=2b^2, which leads to contradiction that both are even."

        return super()._determine_output_text(prompt)


class MockLargeModel(BaseMockModel):
    """
    Mock adapter representing Tier D (Large Model).
    Higher simulated latency, higher token price, richer deterministic responses
    with complete headings and detailed mathematical proofs.
    Operates strictly offline without network access or API keys.
    """

    def __init__(
        self,
        model_id: str = "mock-large",
        prompt_price_per_1m: float = 2.50,
        completion_price_per_1m: float = 10.00,
        simulated_base_latency_ms: float = 85.0,
        price_table_date: str = "2026-10-01",
    ):
        super().__init__(
            model_id=model_id,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            simulated_base_latency_ms=simulated_base_latency_ms,
            price_table_date=price_table_date,
        )

    def _determine_output_text(self, prompt: str) -> str:
        clean = prompt.strip().lower()
        if "square root of 2" in clean:
            return "# Mathematical Proof: Irrationality of Square Root of 2\nStep 1: Assume sqrt(2) = a/b in lowest terms where a, b are integers.\nStep 2: 2 = a^2/b^2, which implies a^2 = 2b^2, so a is an even integer (a = 2k).\nStep 3: Substituting yields 4k^2 = 2b^2, hence b^2 = 2k^2, so b is also even.\nStep 4: This contradicts our assumption that a and b share no common factors. Therefore, sqrt(2) is irrational."

        return super()._determine_output_text(prompt)
