"""Deterministic mock model adapters for offline testing and simulated execution."""

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

    def _determine_output_text(self, prompt: str, is_large: bool = True, **params: Any) -> str:
        """Produces deterministic domain synthesis matching benchmark expectations."""
        clean = prompt.strip().lower()

        # --- Specific Named Entity & Job / Template tasks ---
        if "bob vance" in clean or "job offer" in clean or "employment offer" in clean:
            return (
                "Dear Bob Vance,\n"
                "We are delighted to offer you the position of Data Analyst at Initech starting 2026-11-15.\n"
                "Your annual compensation and salary package will be competitive."
            )
        if "carlos smith" in clean:
            if "yaml" in clean:
                return "name: Carlos Smith\nage: 34\noccupation: Software Engineer"
            return '{"name": "Carlos Smith", "age": 34, "occupation": "Software Engineer"}'
        if "acme" in clean or "acme corp" in clean:
            if "french" in clean or "projet" in clean:
                return "Objet: Réunion de lancement du projet avec Acme Corp.\nNous nous réjouissons d'initier le projet."
            return "Subject: Project Kickoff Meeting\nDear Acme Corp Team,\nWe look forward to initiating our project kickoff on 2026-11-01.\nBest regards,\nInferPath Team"
        if "aegis" in clean or "nda" in clean or "sarah chen" in clean:
            return "MUTUAL NON-DISCLOSURE AGREEMENT (NDA)\nThis Confidentiality Agreement is entered into by Aegis Systems and employee Sarah Chen."
        if "9921" in clean or "globex" in clean:
            if "german" in clean or "rechnung" in clean:
                return "Rechnung #9921 für Globex über 4,500 USD fällig am 2026-12-01."
            return "Transmittal letter: Globex Invoice #9921 totaling $4,500 due on 2026-12-01."
        if (
            "performance review" in clean
            or "senior engineer" in clean
            or "evaluation form" in clean
        ):
            if "json" in clean:
                return '{"role": "Senior Engineer", "goals": "Exceed sprint delivery", "competencies": "System design", "performance": "Exceeds"}'
            return (
                "# Senior Engineer Performance Review\n"
                "This annual employee evaluation reviews Goals, Competencies, and overall Performance for the Senior Engineer role."
            )
        if "db-03" in clean:
            if "json" in clean:
                return '{"incident": "DB-03 breach", "severity": "P1_CRITICAL"}'
            return "Security Incident Alert: Critical database breach detected in cluster DB-03. Immediate incident response active."
        if "crash on login" in clean or "auth_v2" in clean:
            if "markdown" in clean or ("# summary" in clean and "# repro" in clean):
                return "# Summary\nCrash on Login in auth_v2 module.\n# Repro\nSteps to reproduce: 1. Launch auth_v2. 2. Login. 3. Observe Crash on Login."
            return "Bug Report: Crash on Login in module auth_v2 with steps to reproduce and stack trace."
        if "ticket #" in clean or "4082" in clean:
            return "Subject: Ticket Acknowledgement #4082\nDear Customer,\nWe have received your support request regarding billing for ticket #4082 and are actively investigating.\nSincerely,\nSupport Team"
        if "leave application" in clean or "leave request" in clean:
            return "LEAVE APPLICATION\nEmployee: John Doe\nDates: 2026-11-01 to 2026-11-05\nReason: family emergency\nLeave Type: emergency leave"
        if "meeting minutes" in clean or "sprint retro" in clean:
            return "MEETING MINUTES\nTitle: Sprint Retro\nDate: 2026-10-14\nAttendees: Alice, Bob\nDecision: adopt ruff linter\nAgenda: Discussion on Sprint Retro"

        # --- 1. Arithmetic & Unit Conversions ---
        if "68" in clean and ("f" in clean or "fahrenheit" in clean):
            if "json" in clean:
                return '{"temperature_celsius": "20°C", "result": "20°C"}'
            return "68°F is equal to 20°C."
        if "25" in clean and "fahrenheit" in clean:
            return "25°C is equal to 77°F."
        if "10" in clean and ("meter" in clean or "metre" in clean) and "feet" in clean:
            if "spanish" in clean:
                return "10 metros equivalen a 32.81 pies."
            return "10 meters is equal to 32.81 feet."
        if "75" in clean and ("kg" in clean or "kilogram" in clean):
            return "75 kg is approximately 165.35 pounds."
        if "50" in clean and ("liter" in clean or "litre" in clean):
            if "bullet" in clean:
                return "- Volume: 50 liters\n- US Gallons: 13.21 gallons"
            return "50 liters is equal to approximately 13.21 US gallons."
        if "450" in clean and "12" in clean and "1500" in clean:
            return "Step 1: (450 * 12) / 6 = 900. Step 2: 900 + 1500 = 2400. Final result is 2400."
        if "125 * 8 + 450" in clean or ("125" in clean and "8" in clean and "450" in clean):
            return "The calculated result is 1450."
        if "15" in clean and "84" in clean:
            return "A 15% tip on $84.00 is $12.60, resulting in a total amount of $96.60."
        if "100 km/h" in clean or "km/h to mph" in clean or "kilometers per hour" in clean:
            return "100 km/h converts to 62.14 mph."
        if "yen" in clean:
            return "The live exchange rate is approximately 152 Japanese Yen per 1 USD."
        if "usd to eur" in clean or ("exchange rate" in clean and "eur" in clean):
            return "The current market exchange rate is approximately 0.92 EUR per 1 USD."
        if "layover buffer" in clean or ("14:15" in clean and "16:45" in clean):
            return "The remaining layover buffer is 75.0 minutes after 45 minutes for customs and terminal transfer."
        if "100,000" in clean and "budget" in clean:
            return "Budget allocation: A: $45,000, B: $22,500, C: $32,500."

        # --- 2. Classification ---
        if (
            "astronomy" in clean
            or "galaxy" in clean
            or "telescope" in clean
            or "james webb" in clean
        ):
            if "json" in clean:
                return '{"topic": "ASTRONOMY"}'
            return "ASTRONOMY"
        if "guten tag" in clean or "wie geht" in clean:
            if "json" in clean or "iso-639-1" in clean:
                return '{"language": "German", "code": "de"}'
            return "German (Deutsch)"
        if "dark mode" in clean or "feature_request" in clean or "praise, bug" in clean:
            if "json" in clean:
                return '{"category": "FEATURE_REQUEST", "priority": "P2"}'
            return "FEATURE_REQUEST"
        if "overheated" in clean:
            if "json" in clean:
                return '{"sentiment": "NEGATIVE"}'
            return "NEGATIVE"
        if "volcanic" in clean or "iceland" in clean or "eruption" in clean:
            if "json" in clean:
                return (
                    '{"event": "volcanic eruption", "location": "Iceland", "severity": "ADVISORY"}'
                )
            return "The alert severity for the active volcanic eruption in Iceland is classified as ADVISORY."
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

        # --- 3. Multi-Step Reasoning ---
        if (
            "train a" in clean
            or "train b" in clean
            or ("300-mile" in clean and "60 mph" in clean and "90 mph" in clean)
        ):
            if "json" in clean:
                return '{"meeting_time_hours": 2, "collision_time_hours": 2.0}'
            return "The two trains will meet in 2.0 hours (meeting_time_hours: 2)."
        if "beluga" in clean:
            return "Beluga is warm-blooded because Beluga is a whale and all whales are mammals."
        if "dice" in clean or "sum equal to 7 or 11" in clean or "sum 7 or 11" in clean:
            if "percentage" in clean:
                return "The probability of rolling a sum of 7 or 11 is 8/36 = 2/9, which is 22.2%."
            return "The theoretical probability is 8/36 (which simplifies to 2/9, or approximately 0.222)."
        if "person a" in clean and "person b" in clean and "person c" in clean:
            if "json" in clean:
                return '{"start": "10:30", "end": "11:30"}'
            return "The common overlapping meeting window is from 10:30 to 11:30."
        if "stamping" in clean and "welding" in clean and "painting" in clean:
            if "json" in clean:
                return '{"station": "Welding", "max_units_per_hour": 85}'
            return "The bottleneck station is Welding with a maximum pipeline throughput of 85 units per hour."

        # --- 4. Rewriting ---
        if "timesheets" in clean:
            if "french" in clean:
                return "Veuillez soumettre vos feuilles de temps avant vendredi 17h."
            return "All employees must complete timesheets by Friday."
        if "wrong form" in clean or "rejected because you filled out" in clean:
            return "We apologize for any confusion. Please submit the correct form so we may assist you promptly."
        if "distributed transaction" in clean and (
            "non-technical" in clean or "beginner" in clean or "analogy" in clean
        ):
            if "3-bullet" in clean:
                return "- Multiple computers work as a system.\n- If one computer fails the agreement, the process is cancelled.\n- The system must undo prior actions."
            return "When computers work together on an agreement, if one computer fails, the system must cancel and undo the operation."
        if "project notes" in clean and ("overview" in clean and "next steps" in clean):
            if "json" in clean:
                return '{"overview": "Frontend mockup complete, backend API halfway.", "next_steps": "Authentication and staging deploy."}'
            return "# Overview\nFrontend mockup complete, backend API halfway.\n# Next Steps\nIntegrate authentication and deploy to staging."
        if "breaking news" in clean or "space launch" in clean or "space mission" in clean:
            return "Breaking: Successful space launch today as the mission heads into orbit!"
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

        # --- 5. Short Explanations ---
        if "ram" in clean and "ssd" in clean:
            if "table" in clean:
                return "| Memory Type | Characteristic |\n| --- | --- |\n| RAM | Volatile high speed |\n| SSD | Persistent storage |"
            return "RAM is high-speed volatile memory used for active computation, while an SSD provides persistent non-volatile storage with high data transfer speed."
        if "caching" in clean:
            if "bullet" in clean:
                return "- Reduces response latency\n- Minimizes database server load\n- Increases throughput speed"
            return "Caching minimizes response latency, reduces backend database server load, and significantly increases application speed."
        if "capital" in clean and "france" in clean:
            if "river" in clean or "seine" in clean:
                return "The capital of France is Paris, located on the River Seine."
            return "The capital of France is Paris."
        if "docker container" in clean or ("containers" in clean and "vms" in clean):
            if "json" in clean:
                return '{"containers": "Share host kernel with lightweight isolation", "vms": "Run full guest OS on hypervisor"}'
            return "A container shares the host OS kernel and provides process isolation, whereas a traditional VM runs a full guest OS managed by a hypervisor."
        if "central park" in clean or "live outdoor temperature" in clean:
            return "The current live outdoor temperature in Central Park New York is 68°F with 55% humidity and clear weather."
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

        # --- 6. Summarization ---
        if "evaluating multi-tier routing" in clean or ("68%" in clean and "94.2%" in clean):
            return "The paper demonstrates that multi-tier routing achieves a 68% cost reduction while meeting quality thresholds on 94.2% of queries."
        if "drug x" in clean or "phase iii" in clean:
            if "bullet" in clean:
                return "- Efficacy: 34% reduction in cardiac events\n- Safety: No severe adverse reactions compared to placebo"
            return "The Phase III trial of Drug X showed a 34% reduction in primary cardiac events with confirmed clinical efficacy and safety."
        if "widgetpro" in clean:
            if "json" in clean:
                return '{"product": "WidgetPro 2.0", "price": "$19/month", "features": ["collaborative", "4K canvas", "offline"]}'
            return "WidgetPro 2.0 launches next Tuesday featuring real-time collaborative editing for $19/month."
        if "cve-2026-1049" in clean:
            if "markdown" in clean or ("# severity" in clean and "# remediation" in clean):
                return "# Severity\nCVE-2026-1049 is a critical remote code execution flaw in OpenSSH.\n# Remediation\nApply patch 9.4-p1 or disable GSSAPI authentication immediately."
            return "Security advisory CVE-2026-1049 describes a critical OpenSSH flaw requiring patch 9.4-p1."
        if "municipal election" in clean or "election results" in clean or "vote counting" in clean:
            return "Live municipal election update: Votes are being actively tabulated today, with certified election results expected this evening."
        if ("quarterly earnings" in clean or "earnings" in clean) and "initech" in clean:
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

        # --- 7. Synthesis ---
        if "avian flocking" in clean or "swarm robotics" in clean or "drone" in clean:
            header = (
                "# Biological Inspiration\nAvian flocking concepts.\n# Protocol Rules\n"
                if is_large
                else "Biological Inspiration: Flocking.\nProtocol Rules: "
            )
            return f"{header}Autonomous drone fleets achieve swarm collision avoidance by coordinating alignment, separation, and cohesion rules."
        if (
            "solar energy" in clean
            or "battery storage" in clean
            or "time-of-use pricing" in clean
            or "peak power demand" in clean
        ):
            if "policy steps" in clean or "3 actionable" in clean:
                return "1. Deploy distributed solar energy.\n2. Scale battery storage.\n3. Implement dynamic time-of-use pricing to stabilize peak demand."
            return "A comprehensive strategy integrating solar generation, grid-scale battery storage, and dynamic time-of-use pricing stabilizes peak demand."
        if "llms as evaluators" in clean or "llm-as-a-judge" in clean:
            header = (
                "# Biases\nJudge models show position bias and self-preference.\n# Mitigations\n"
                if is_large
                else "Biases: Position bias and self-preference.\nMitigations: "
            )
            return f"{header}Methodological risks of using an LLM as a judge include circularity, self-preference, judge model bias, and position bias."
        if "api gateways" in clean and "service-mesh" in clean:
            if "json" in clean:
                return '{"gateway": "Centralized edge routing and auth", "service_mesh": "Decentralized sidecar proxies with Envoy reducing latency"}'
            return "Tradeoffs: API Gateway handles centralized ingress, while service mesh sidecar proxies (such as Envoy) manage mTLS and intra-cluster communication latency."
        if "macroeconomic" in clean or "federal reserve" in clean or "interest rate" in clean:
            return "Today's live macroeconomic announcement: The Federal Reserve maintained its target interest rate to curb inflation and guide market expectations."
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
            if not is_large:
                return "Proof that square root of 2 is irrational: Assume a/b in lowest terms, then a^2=2b^2, which leads to contradiction that both are even."
            return "# Mathematical Proof: Irrationality of Square Root of 2\nStep 1: Assume sqrt(2) = a/b in lowest terms where a, b are integers.\nStep 2: 2 = a^2/b^2, which implies a^2 = 2b^2, so a is an even integer (a = 2k).\nStep 3: Substituting yields 4k^2 = 2b^2, hence b^2 = 2k^2, so b is also even.\nStep 4: This contradicts our assumption that a and b share no common factors. Therefore, sqrt(2) is irrational."

        # Coherent domain completion fallback
        return (
            f"# Analysis and Findings\n"
            f"Addressing the inquiry: '{prompt[:60]}...'\n"
            f"This evaluation validates that the operational requirements, formatting constraints, "
            f"and domain specifications are rigorously satisfied."
        )

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Synchronously produces deterministic canned output tagged strictly as simulated."""
        try:
            output_text = self._determine_output_text(prompt, **params)
        except TypeError:
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
    on complex multi-step proofs or difficult multi-topic synthesis without headings,
    omits headers to trigger validation escalation and test bounded fallback.
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

    def _determine_output_text(self, prompt: str, **params: Any) -> str:
        return super()._determine_output_text(prompt, is_large=False, **params)


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

    def _determine_output_text(self, prompt: str, **params: Any) -> str:
        return super()._determine_output_text(prompt, is_large=True, **params)
