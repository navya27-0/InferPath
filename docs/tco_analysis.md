# Total Cost of Ownership (TCO) & Router Overhead Analysis

## 1. Motivation: Accounting for Operational & Compute Overhead

Discussions of LLM routing economics often suffer from **pure-inference myopia**: cost reductions are computed solely based on token price differences, ignoring the operational costs of the router itself:
- CPU compute time to analyze, route, and validate requests.
- Async queueing and event-loop processing overhead.
- Cache infrastructure costs (disk I/O and Redis memory footprint).
- Observability ingestion fees (structured JSON logging and distributed tracing).
- Network egress and serialization costs.

For very short prompts, these auxiliary operational expenses can represent a non-trivial percentage of total expenditure.

---

## 2. Quantitative TCO Breakdown across Prompt Lengths

Artifact: [`data/results/tco_analysis.json`](../data/results/tco_analysis.json)

Using standardized 2026 enterprise cloud pricing:
- Serverless vCPU compute: $\$0.00001667 / \text{vCPU-second}$ (AWS Lambda / Google Cloud Run).
- Telemetry ingestion: $\$0.50 / \text{GB}$ (Datadog / Amazon CloudWatch).
- Network egress: $\$0.09 / \text{GB}$.
- Amortized distributed cache: $\$0.0000005 / \text{request}$ (Redis Enterprise cluster).

| Prompt Size | Inference Cost | Router Compute | Validation + Cache | Observability | Total TCO | Router Overhead % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 tokens** | $0.0000090 | $0.0000001 | $0.0000006 | $0.0000005 | **$0.0000100** | **10.11%** |
| **50 tokens** | $0.0000150 | $0.0000001 | $0.0000006 | $0.0000005 | **$0.0000160** | **6.41%** |
| **150 tokens**| $0.0000450 | $0.0000001 | $0.0000006 | $0.0000005 | **$0.0000461** | **2.34%** |
| **500 tokens**| $0.0001500 | $0.0000001 | $0.0000006 | $0.0000006 | **$0.0001513** | **0.83%** |
| **2000 tokens**| $0.0006000| $0.0000001 | $0.0000006 | $0.0000013 | **$0.0006020** | **0.34%** |
| **4000 tokens**| $0.0012000| $0.0000001 | $0.0000006 | $0.0000024 | **$0.0012031** | **0.25%** |

---

## 3. Findings & Economic Breakeven Boundary

1. **Short Prompt Tax**:
   On ultra-short inputs ($<25$ tokens, such as single-word classifications or greetings), router compute and logging constitute **6.4% to 10.1%** of total query cost.
2. **Rapid Amortization**:
   As prompts exceed **150 tokens**, router overhead rapidly collapses below **2.3%**, and drops under **0.35%** for typical enterprise context lengths ($>1,000$ tokens).
3. **Net ROI**:
   Because routing a 500-token prompt from Frontier ($0.0035) to Small ($0.00015) saves approximately $\$0.00335$ per call, the $\$0.0000013$ operational overhead represents only **0.038%** of gross savings, proving that InferPath yields overwhelming net-positive financial ROI.
