# A Hallucination Detection and Mitigation Framework for Faithful Text Summarization Using LLMs

**Authors:** Liu et al. (2025)  
**Venue:** Scientific Reports  
**Link:** [Nature](https://www.nature.com/articles/s41598-025-31075-1)

---

## 1. Methodology

### Hallucination Problem
Defined as the inclusion of any span in a generated summary that is not supported by the input document. The paper identifies multifaceted causes:
- Imperfect representation learning in the encoder
- Erroneous decoding in the decoder
- Exposure bias during inference

### Detection Framework (Q-S-E Methodology)

**Q - Question-Answer Generation:**
Uses an "answer-first" strategy where the LLM first identifies key factual statements (answers) from the source document (entities, figures, events) and then formulates corresponding questions. This reduces the risk of hallucination within the QA pairs themselves.

**S - Sorting:**
Prioritizes QA pairs based on their ROUGE-1 F1 scores relative to the generated summary to identify the most semantically relevant content for evaluation.

**E - Evaluation:**
Generates answers for the selected questions using only the summary. It then compares these to the original source-derived answers using **FactCC**, which determines if claims are logically and factually entailed.

### Mitigation Approach
An **iterative hallucination resolution mechanism** using prompt engineering:
1. When a discrepancy is detected, the system provides the LLM with a "hint" that localizes the error and provides the correct source information
2. The process repeats (evaluation → prompt refinement → revision) until the hallucination level falls below a predefined threshold δ

---

## 2. Strengths

- **Framework Achievements:**
  - Enables quantitative detection of hallucinations
  - Provides transparency in the modification process (users can see exactly which parts of a summary were corrected)
  - Bridges the gap between generation accuracy and user satisfaction by improving interpretability

- **Medical/PubMed Testing:**
  - Demonstrated strong suitability for scientific and technical domains
  - On the **PubMed** dataset, achieved:
    - Highest factual consistency (FactCC score of 37.78)
    - Best informativeness (BartScore of -1.76)
  - Outperformed baselines like ChatGPT and GPT-3
  - Additional baselines included BART, PEGASUS, Ele-aware, and SummIt (zero-shot)

---

## 3. Limitations

- **Optimization Constraints:** There is a "sweet spot" for iterative refinement; the paper recommends limiting to approximately **6 iterations**. Exceeding this can lead to loss of critical information and decreased ROUGE/BertScore metrics
- **Recommendation Scope:** The 6-iteration recommendation is based on experiments over 50 PubMed samples and may not generalize across datasets
- **Challenging Hallucinations:** The framework focuses primarily on factual consistency. Time-sensitive domains and temporal accuracy remain challenges for future work

---

## 4. Key Findings

### Factual Consistency Results
The framework markedly improved factual consistency across three benchmarks:

| Dataset | FactCC Improvement over ChatGPT |
|---------|--------------------------------|
| CNN/Daily Mail | +1.1 points |
| PubMed | +2.09 points |
| ArXiv | Best performance (38.14) |

### Metrics Used
- ROUGE (1, 2, L)
- BertScore
- FactCC
- BartScore

### Practical Mitigation Strategies
1. **Error Localization:** Pinpointing specific segments of inconsistency to focus the LLM's attention
2. **Correction Guidance:** Providing the correct factual answer from the source text within the refinement prompt to ensure the revision is grounded
3. **Human Alignment:** Human evaluations (using a five-point Likert scale) showed a significant preference for iteratively modified summaries over those from standard pre-trained models
4. **Ablation Insight:** Multiple iterations with QA sorting outperform single-iteration and no-sorting baselines (tested on 50 samples each from PubMed and CNN/Daily Mail)

---

## Design Takeaway

This paper informs the ethical considerations section of the project. Medical summarization carries significant hallucination risk — the model may generate plausible-sounding but factually incorrect medical claims. The paper tested their framework on PubMed data (the same domain as this project) and provides practical mitigation strategies: error localization, iterative refinement, and fact-checking against source documents. For this project, we acknowledge hallucination as a key risk and discuss mitigations including disclaimers ("not medical advice"), extractive constraints, and the importance of human review for any clinical application.
