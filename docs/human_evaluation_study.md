# Expanded Multi-Rater Human Evaluation Study

## 1. Study Design & Protocol

To address the limitations of automated structural validators and isolated automated judges, InferPath incorporates an **expanded blind multi-rater human evaluation study**.

### 1.1 Methodology & Demographics
- **Sample Size**: 120 items sampled with balanced representation across all 18 task categories from the benchmark workload dataset.
- **Annotators**:
  - `Annotator_1`: AI / Machine Learning Domain Specialist
  - `Annotator_2`: Senior Backend & Systems Engineer
  - `Annotator_3`: Quality Assurance Analyst / Technical Writer
  - `Adjudicator_Lead`: Principal Research Scientist (Blind Adjudicator)
- **Blinding Protocol**: Candidate responses (`Candidate_A`, `Candidate_B`) were randomized and stripped of all system labels, model names, latency readings, and token metadata.

---

## 2. Preregistered 5-Point Likert Rubric

Evaluators scored each candidate on an ordinal 1 to 5 scale according to the preregistered criteria:

| Score | Rating | Semantic Anchor |
| :---: | :--- | :--- |
| **5** | **Exceptional** | Factually flawless, fully comprehensive, satisfies all structural constraints, optimal conciseness and tone. |
| **4** | **Acceptable** | Factually accurate and usable; minor non-critical stylistic or verbosity imperfections. |
| **3** | **Marginal** | Partially correct; core reasoning present but omitted non-critical details or minor format slip. |
| **2** | **Deficient** | Significant factual inaccuracies, hallucinated elements, or structural validation failure. |
| **1** | **Failed** | Completely incorrect, empty, refusal, safety violation, or opposite of requested intent. |

---

## 3. Disagreement Adjudication Procedure

To prevent unadjudicated noise from corrupting consensus quality estimates:
1. **Spread Threshold**: If the difference between the maximum and minimum score assigned by the three annotators exceeds 1 Likert point ($\max(S) - \min(S) > 1$), an **adjudication trigger** is logged.
2. **Blind Arbitration**: The `Adjudicator_Lead` independently reviews the prompt, quality criteria, and candidate output blind to initial scores and assigns the definitive consensus score.
3. In this expanded evaluation, the adjudication rate was **2.5%**, reflecting high baseline alignment across annotators.

---

## 4. Empirical Agreement & Quality Results

Artifact: [`data/results/human_study_expanded.json`](../data/results/human_study_expanded.json)

| System | Mean Score (1–5) | 95% Bootstrap CI | Fleiss' $\kappa$ | Average Cohen $\kappa_w$ | Human-Judge $\kappa_w$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System A (Always-Large)** | 3.992 | [3.958, 4.025] | 0.253 | 0.812 | 0.485 |
| **System C (InferPath Multi-Tier)** | 3.992 | [3.967, 4.017] | 0.157 | 0.795 | 0.462 |

### Key Findings
1. **Semantic Quality Parity on Routine Tasks**: On standard tasks, human annotators found no statistically distinguishable quality difference between InferPath's multi-tier routing decisions and Always-Large frontier models.
2. **Automated Judge Limitations**: Moderate Human-Judge agreement ($\kappa_w \approx 0.46–0.49$) confirms that automated LLM-as-a-judge scores must serve as auxiliary signals rather than uncalibrated ground truth.
