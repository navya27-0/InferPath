# Privacy Classification & Adversarial DLP Evaluation

## 1. Threat Model: Defense-in-Depth vs. Complete Data Loss Prevention

A fundamental architectural boundary must be made explicit:
> **InferPath's pattern-based privacy detector is an automated defense-in-depth safeguard, not an infallible or standalone Data Loss Prevention (DLP) system.**

In production AI architectures, callers and ingress API gateways remain primarily responsible for declaring appropriate data classifications (`public`, `internal`, `confidential_private`) at request submission time. InferPath's pattern detector operates as an autonomous, zero-overhead second-layer safeguard designed to intercept unclassified credentials, PII, and sensitive records before external API transmission occurs.

---

## 2. Adversarial Privacy Probe Results

- **Committed Result Artifact**: [`data/results/privacy_evaluation.json`](../data/results/privacy_evaluation.json)
- **Validation Script**: [`scripts/evaluate_privacy.py`](../scripts/evaluate_privacy.py)
- **Automated Test**: [`tests/unit/test_privacy_eval.py`](../tests/unit/test_privacy_eval.py)
- **Evidence Label**: `[MEASURED OFFLINE]`

### 2.1 Probe Suite Composition

The evaluation suite comprises **20 carefully curated probes** across 8 distinct categories (13 sensitive positive targets, 7 benign negative controls):

| Category | Probe Count | Target Type | Example / Intent |
| :--- | :---: | :--- | :--- |
| **Direct Credentials** | 2 | API keys, bearer tokens | Standard bearer tokens and project keys (`sk-proj-...`). |
| **Plaintext PII** | 2 | SSN, payment cards | Social security numbers, formatted credit cards with CVV. |
| **Encoded Credentials** | 2 | Base64, Hex | Raw Base64 token strings, Hex-encoded firmware secrets. |
| **Multilingual PII / Secrets** | 3 | Spanish, French, German | Non-English sensitive banking and trade-secret disclosures. |
| **Context Leaks** | 1 | Unstructured prompt context | RSA private key embedded within prompt background context. |
| **Metadata Leaks** | 1 | Request dictionary fields | API credentials passed within request metadata. |
| **Prompt Injection Exfiltration** | 1 | Adversarial system override | Prompts attempting to force disclosure of proprietary IP. |
| **Indirect Confidential Disclosures**| 1 | Unlabeled confidential memos | Indirect references to proprietary M&A project documents. |
| **Benign Controls (Negative)** | 7 | Public code, docs, math, UUIDs | Benign hashing functions, OAuth docs, UUIDs, git commands. |

---

### 2.2 Empirical Confusion Matrix

Across the 20 evaluated probes:

| Actual \\ Predicted | Predicted Sensitive (Blocked / Intercepted) | Predicted Non-Sensitive (Allowed / Forwarded) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Sensitive (13 probes)** | **8 (True Positives, TP)** | **5 (False Negatives, FN)** | 13 |
| **Actual Benign (7 probes)** | **0 (False Positives, FP)** | **7 (True Negatives, TN)** | 7 |
| **Total** | 8 | 12 | 20 |

---

### 2.3 Statistical Performance Metrics & 95% Confidence Intervals

Metrics computed with 95% Wilson score confidence intervals:

| Metric | Measured Value | 95% Wilson Score CI | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Precision** | **1.000 (100.0%)** | `[67.6%, 100.0%]` | Zero false positives on benign developer inputs, code snippets, and UUIDs. |
| **Recall / Sensitivity** | **0.615 (61.5%)** | `[35.5%, 82.3%]` | Catches plaintext credentials, PII, context/metadata leaks, and hex strings. |
| **Specificity** | **1.000 (100.0%)** | `[64.6%, 100.0%]` | 100% of benign queries pass without friction or spurious blocking. |
| **F1 Score** | **0.762 (76.2%)** | — | Balanced harmonic mean under zero-false-alarm constraint. |
| **False Positive Rate (FPR)** | **0.000 (0.0%)** | `[0.0%, 35.4%]` | Zero developer disruption on public queries. |
| **False Negative Rate (FNR)** | **0.385 (38.5%)** | `[17.7%, 64.5%]` | **Substantial vulnerability under adversarial obfuscation.** |

---

## 3. Deep-Dive Forensic Failure Analysis

### 3.1 Which Obfuscation Techniques Bypassed Detection?

Five probes successfully bypassed the pattern-based detector (False Negatives):

1. **`priv_01` (Unanchored Project Keys)**:
   - *Probe*: `sk-proj-94819481948194819481948194819481`
   - *Cause*: Regex was tuned to legacy `sk-[A-Za-z0-9]{32}` patterns and did not match hyphenated project-scoped prefixes without surrounding keyword context.
2. **`priv_05` (Raw Base64 Token Without Plaintext Anchor)**:
   - *Probe*: `Token: c2stYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY==`
   - *Cause*: The payload represents an encoded `sk-...` string, but raw Base64 without decoded evaluation or high-entropy anchoring was bypassed to prevent catastrophic false positives on benign base64 data.
3. **`priv_07` (Multilingual Spanish PII)**:
   - *Probe*: `Documento confidencial: número de seguro social 987-65-4321`
   - *Cause*: Standard regex rules target English SSN / Social Security terminology; localized variations require multilingual dictionaries.
4. **`priv_08` (Multilingual French Banking Terms)**:
   - *Probe*: `Fichier confidentiel contenant carte bancaire 4970 1234 5678 9012`
   - *Cause*: Space-delimited card numbers paired with French identifiers were not normalized prior to rule matching.
5. **`priv_09` (Multilingual German Trade Secret)**:
   - *Probe*: `Streng vertraulich: Geschäftsgeheimnis der Fusionsvereinbarung`
   - *Cause*: Compound German nouns without English translation anchors bypassed simple keyword matching.

### 3.2 Operational Consequence: Does a False Negative Trigger External Transmission?

**YES.** If an incoming request satisfies all of the following conditions:
1. The client leaves `data_classification` set to `public` (or unclassified);
2. The prompt contains sensitive data disguised using one of the obfuscated methods above;
3. No pre-gateway DLP filter is active;

Then InferPath's pattern detector will classify the request as non-sensitive and route it to an external provider adapter (Tier C or Tier D). **This represents a genuine data-leakage risk if the router is improperly relied upon as an enterprise security perimeter.**

### 3.3 Protection Offered by Metadata-Based Classification

In contrast to content-based pattern matching, **caller-declared metadata classification is deterministic and 100% effective**:
- When a caller sets `data_classification=confidential_private` or `internal`, InferPath's Stage 1 guardrail unconditionally intercepts the request:
  - If the requested tier or operation requires external transmission, InferPath returns an immediate **HTTP 403 Privacy Violation**.
  - If local deterministic execution (Tier A) or on-premise templates (Tier B) can fulfill the request, it is serviced strictly locally without external network egress.
- Metadata enforcement has a **0.0% False Negative Rate** because it evaluates declared protocol fields rather than parsing natural language text.

---

## 4. Multi-Layer Defense-in-Depth Mitigation Architecture

To address the 38.5% FNR under adversarial obfuscation in production enterprise environments (Mode 3), the following mitigations should be layered:

```
[Incoming Request]
       │
       ▼
┌────────────────────────────────────────┐
│ 1. Mandatory Ingress Classification    │  --> Declared confidential? -> Immediate Local Route / 403
└──────────────────┬─────────────────────┘
                   │ If declared Public
                   ▼
┌────────────────────────────────────────┐
│ 2. Pre-Flight Normalization & Decoding │  --> Base64 / Hex / URL decode to canonical plaintext
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 3. Multi-Signal Pattern & Entropy DLP  │  --> Regex + Shannon Entropy check for high-entropy keys
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 4. Compact Local Embedding Classifier  │  --> 5ms local ONNX model for multilingual semantic intent
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ 5. Outbound Adapter Egress Firewall    │  --> Inspect model output tokens before caller delivery
└────────────────────────────────────────┘
```

1. **Pre-Flight Decoding**: Automatically decode Base64, Hex, and URL-encoded chunks prior to pattern scanning.
2. **Shannon Entropy Heuristics**: Flag unanchored high-entropy token strings ($H > 4.5\text{ bits/char}$) for inspection.
3. **Small Offline Embedding Classifier**: Run a lightweight (<5ms) quantized local ONNX model (e.g. DeBERTa/BERT privacy classifier) to catch multilingual and paraphrased disclosures.
4. **Outbound Egress Filtering**: Inspect outgoing responses from model adapters to catch reflected credentials.
5. **Enforce Perimeter Tagging**: API gateways must inject authenticated user and tenant clearance tags before requests reach InferPath.

---

## 5. Methodological Conclusion

> **Core Research Finding**:  
> Pattern-based privacy detection provides an effective, zero-overhead defense-in-depth signal (achieving **100.0% precision** with zero false alarms on benign developer traffic), but exhibits a measured **38.5% false-negative rate under adversarial obfuscation**.  
> **It is therefore insufficient as a standalone privacy classifier and must not be used as the sole perimeter defense in untrusted multi-tenant environments.**
