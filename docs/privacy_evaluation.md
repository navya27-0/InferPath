# Privacy Classification & Adversarial DLP Evaluation

## 1. Threat Model: Defense-in-Depth vs Complete Data-Loss Prevention

A crucial security boundary must be emphasized: **InferPath's privacy classifier is an automated defense-in-depth safeguard, not an infallible Data Loss Prevention (DLP) system**.

Callers remain primarily responsible for declaring appropriate data classifications (`public`, `internal`, `confidential_private`) at request submission time. InferPath's pattern detector serves as an autonomous secondary safety net to intercept unclassified credentials, PII, and sensitive records before external API transmission occurs.

---

## 2. Adversarial Privacy Probe Results

Artifact: [`data/results/privacy_evaluation.json`](../data/results/privacy_evaluation.json)

The privacy subsystem was evaluated against a suite of 20 positive and negative adversarial probes covering:
- Plaintext API keys and bearer tokens
- Social Security Numbers and Credit Card numbers
- Base64 and Hex encoded secrets
- Multilingual sensitive terms (Spanish, French, German)
- Sensitive context and metadata payload leaks
- Prompt-injection exfiltration attacks
- Negative controls (public code, benign hashes, UUIDs, documentation)

### 2.1 Confusion Matrix

| Actual \\ Predicted | Predicted Sensitive (Blocked) | Predicted Non-Sensitive (Allowed) |
| :--- | :---: | :---: |
| **Actual Sensitive (13 probes)** | **8 (True Positive)** | **5 (False Negative)** |
| **Actual Benign (7 probes)** | **0 (False Positive)** | **7 (True Negative)** |

### 2.2 Performance Metrics

| Metric | Measured Value | Security Interpretation |
| :--- | :---: | :--- |
| **Precision** | **1.000 (100.0%)** | Zero false alarms: benign code, documentation, and UUIDs are never blocked. |
| **Recall / Sensitivity** | **0.615 (61.5%)** | Intercepts standard credentials, PII, and direct sensitive keywords. |
| **Specificity** | **1.000 (100.0%)** | Clean rejection of benign queries. |
| **False Positive Rate (FPR)** | **0.000 (0.0%)** | Ideal operational friction-free rate. |
| **False Negative Rate (FNR)** | **0.385 (38.5%)** | Highlights vulnerability to indirect paraphrasing and non-English phrasing. |

---

## 3. Analysis & Key Security Recommendations

1. **Defense-in-Depth Boundary**:
   The measured **38.5% False Negative Rate** under adversarial obfuscation demonstrates why keyword and regex heuristics cannot substitute for enterprise DLP or caller-declared classification.
2. **Encrypted / Obfuscated Secrets**:
   While base64 patterns with known prefixes (`sk-`, `ghp_`) are successfully blocked, novel high-entropy strings without semantic anchors can bypass simple regex filters.
3. **Mandatory Policy Recommendation**:
   In enterprise production deployments (Mode 3), caller identity management and API gateway DLP filters must enforce data classification at the perimeter, with InferPath acting as the second-layer guardrail.
