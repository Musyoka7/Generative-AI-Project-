# Generative and Collaborative AI - Project Plan

## Project Overview

| Aspect | Decision |
|--------|----------|
| **Modality** | Text Generation |
| **Task** | Summarization |
| **Domain** | Medical Research (PubMed) |
| **Angle** | Summarizing clinical papers for easier understanding |
| **Dataset** | PubMed/PMC (paper → abstract pairs) |
| **Evaluation** | ROUGE, BERTScore (compare to real abstracts) |
| **Ethics Hook** | Hallucination risks + mitigations |
| **Fine-Tuning** | LoRA on BART using paper-abstract pairs |
| **Dataset Source** | SUMPUBMED (Hugging Face: Blaise-g/SumPubmed) — 32,684 papers |

---

## Marks Distribution

```
Task 9: Compare & Contrast    ███████████████ 15
Task 7: Fine-Tune             ███████████████ 15
Task 5: Prompt Engineering    ███████████████ 15
Task 2: Literature Review     ██████████ 10
Task 8: Evaluate Fine-Tune    ██████████ 10
Task 11: Code Submission      ██████████ 10
Task 1: Goal                  █████ 5
Task 3: Baseline              █████ 5
Task 4: Evaluate Baseline     █████ 5
Task 6: Evaluate Prompts      █████ 5
Task 10: Ethics               █████ 5
                              ─────────────────
                              TOTAL: 100
```

---

## Execution Order

| Phase | Tasks | Why This Order |
|-------|-------|----------------|
| **1. Setup** | Task 1 (Goal) | Define before building |
| **2. Research** | Task 2 (Literature) | Informs your approach |
| **3. Baseline** | Task 3 → Task 4 | Build, then evaluate |
| **4. Prompts** | Task 5 → Task 6 | Engineer, then evaluate |
| **5. Fine-Tune** | Task 7 → Task 8 | Train, then evaluate |
| **6. Synthesize** | Task 9 (Compare) | Needs all previous results |
| **7. Reflect** | Task 10 (Ethics) | Can write anytime |
| **8. Polish** | Task 11 (Code) | Clean up at end |

---

## Task 1: Coursework Goal (5 marks)

- [ ] **Complete**

### Requirements

| Requirement | What You'll Write |
|-------------|-------------------|
| **Objective** | Build a medical research summarization system that condenses PubMed papers into patient-accessible summaries |
| **Problem Statement** | Patients and non-experts struggle to understand clinical research. This creates a knowledge gap between published findings and public understanding |
| **Intended Outcome** | A fine-tuned model that produces accurate, readable summaries of medical papers |
| **Success Criteria** | ROUGE-L > 0.40, BERTScore > 0.80 on held-out test set |
| **Risk Assessment** | Hallucination risk, over-simplification losing critical nuance |

### Deliverable
~200 words in report introduction

---

## Task 2: Literature Review (10 marks)

- [ ] **Complete**

### Papers to Include

| # | Paper | Link | Why Include It |
|---|-------|------|----------------|
| **1** | Lewis et al. (2020) — "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" | [arXiv](https://arxiv.org/abs/1910.13461) | Foundational model — justifies BART as your base architecture for summarization |
| **2** | Lu et al. (2025) — "Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings" | [arXiv](https://arxiv.org/abs/2505.21242) | Domain-specific challenges — explains medical terminology issues |
| **3** | Hu et al. (2022) — "LoRA: Low-Rank Adaptation of Large Language Models" | [arXiv](https://arxiv.org/abs/2106.09685) | Fine-tuning methodology — justifies LoRA for efficient training |
| **4** | "A hallucination detection and mitigation framework for faithful text summarization using LLMs" (2025) | [Nature](https://www.nature.com/articles/s41598-025-31075-1) | Ethics support — tested on PubMed, provides mitigation strategies |

### For Each Paper You Need:
- 1-2 paragraph summary of methodology
- Strengths and limitations
- **Design takeaway** — how it influenced YOUR pipeline

### Deliverable
~600-800 words

---

## Task 3: Generate Baseline (5 marks)

- [ ] **Complete**

### Requirements

| Component | What You'll Do |
|-----------|----------------|
| **Model** | BART (facebook/bart-base or bart-large-cnn for summarization) |
| **Prompts** | 3-5 simple prompts like: `"Summarize this medical paper:"` |
| **Outputs** | 6-8 representative summaries from different paper types |
| **Documentation** | Record inference parameters (temperature, max_length, etc.) |

### Example Baseline Prompt
```
Summarize the following medical research paper:

{paper_text}

Summary:
```

### Deliverable
Code cells + 6-8 output examples in report

---

## Task 4: Evaluate Baseline (5 marks)

- [ ] **Complete**

### Metrics

| Metric | Why Use It |
|--------|------------|
| **ROUGE-1, ROUGE-2, ROUGE-L** | Standard for summarization, measures word overlap |
| **BERTScore** | Captures semantic similarity (not just exact words) |

### What You'll Show
```
┌─────────────────────────────────────────┐
│  BASELINE RESULTS                       │
├─────────────────────────────────────────┤
│  ROUGE-1:  0.28                         │
│  ROUGE-2:  0.11                         │
│  ROUGE-L:  0.24                         │
│  BERTScore F1:  0.71                    │
├─────────────────────────────────────────┤
│  Interpretation: Baseline captures      │
│  general topic but misses key details   │
│  and medical terminology...             │
└─────────────────────────────────────────┘
```

### Deliverable
Metrics table + 3-5 sentence interpretation

---

## Task 5: Prompt Engineering (15 marks)

- [ ] **Complete**

### Techniques to Apply

| Technique | Example |
|-----------|---------|
| **Role framing** | `"You are a medical researcher. Summarize this paper for a patient with no medical background:"` |
| **Audience variation** | Doctor vs patient vs general public |
| **Structure prompts** | `"Summarize in 3 bullet points: 1) What was studied 2) Key findings 3) Implications"` |
| **Few-shot examples** | Provide 1-2 example summaries before asking for output |
| **Length control** | `"In exactly 50 words, summarize..."` |

### What Top Marks Require
- Systematic testing (change ONE variable at a time)
- Before/after comparison grids
- Hypothesis → Test → Result format

### Example Grid

| Prompt Version | Change Made | ROUGE-L | Observation |
|----------------|-------------|---------|-------------|
| v1 (baseline) | None | 0.24 | Generic output |
| v2 | Added role framing | 0.31 | More structured |
| v3 | Added audience (patient) | 0.35 | Simpler language |
| v4 | Added few-shot example | 0.38 | Better format |

### Deliverable
4-6 prompt variations with systematic comparison

---

## Task 6: Evaluate Prompts (5 marks)

- [ ] **Complete**

### What You'll Show
- Same metrics as baseline (ROUGE, BERTScore)
- Side-by-side comparison table
- Short analysis: "Prompt v4 improved ROUGE-L by 58% over baseline because..."

### Deliverable
Comparison table + ~150 word analysis

---

## Task 7: Fine-Tune Generator (15 marks)

- [ ] **Complete**

### Requirements

| Component | What You'll Do |
|-----------|----------------|
| **Dataset** | SUMPUBMED — 26,147 train / 3,268 dev / 3,269 test |
| **Train/Val Split** | Pre-split in dataset (or use subset for faster training) |
| **Technique** | LoRA (Low-Rank Adaptation) — works on laptop |
| **Base Model** | BART (facebook/bart-base or bart-large-cnn) |
| **Parameters to Document** | Batch size, learning rate, epochs, LoRA rank |

### Configuration to Document
```
┌─────────────────────────────────────────┐
│  FINE-TUNING CONFIGURATION              │
├─────────────────────────────────────────┤
│  Base Model:     BART-base (139M params)│
│  Technique:      LoRA (rank=8)          │
│  Dataset:        800 train / 200 val    │
│  Batch Size:     8                      │
│  Learning Rate:  2e-4                   │
│  Epochs:         3                      │
│  Training Time:  ~45 minutes            │
└─────────────────────────────────────────┘
```

### Deliverable
Full fine-tuning code + config documentation

---

## Task 8: Evaluate Fine-Tuning (10 marks)

- [ ] **Complete**

### What You'll Discuss
- Same metrics on test set
- **Generalization:** Does it work on unseen papers?
- **Overfitting:** Does it just memorize training data?
- **Cost/Latency:** How long does inference take?

### Key Comparison

| Model | ROUGE-L | BERTScore | Inference Time |
|-------|---------|-----------|----------------|
| Baseline | 0.24 | 0.71 | 1.2s |
| Prompt Engineered | 0.38 | 0.79 | 1.3s |
| Fine-Tuned | 0.52 | 0.87 | 1.4s |

### Deliverable
Metrics + generalization discussion

---

## Task 9: Compare & Contrast (15 marks)

- [ ] **Complete**

### What You'll Produce
- **One consolidated table** showing all three stages
- **Side-by-side example outputs** (same paper, three different summaries)
- **"What worked and why"** reflection

### Example Side-by-Side

| Stage | Summary of Same Paper | ROUGE-L |
|-------|----------------------|---------|
| **Baseline** | "The study examines zinc. Results were found. More research needed." | 0.24 |
| **Prompt Eng.** | "This clinical trial studied zinc supplementation in children. Key finding: 23% improvement." | 0.38 |
| **Fine-Tuned** | "A 12-week RCT (n=200) found zinc supplementation reduced cold duration by 23% in children aged 5-10 (p<0.05)." | 0.52 |

### Reflection Points
- Fine-tuning captured medical terminology better
- Prompt engineering improved structure but not domain knowledge
- Baseline was too generic

### Deliverable
Comparison table + example outputs + ~300 word reflection

---

## Task 10: Ethical Considerations (5 marks)

- [ ] **Complete**

### Risks and Mitigations

| Risk | Specific to Your Project | Mitigation |
|------|-------------------------|------------|
| **Hallucination** | Model generates "zinc cures cancer" when paper said "showed minor improvement" | Extractive constraints, confidence thresholds, disclaimers |
| **Oversimplification** | Critical dosage info lost in summary | Multi-level summaries (technical + layman) |
| **Misuse** | People use AI summary as medical advice | Clear "not medical advice" disclaimers |
| **Bias** | Training data skewed toward certain populations | Document dataset demographics, acknowledge limitations |

### Deliverable
~300 words discussing risks + mitigations

---

## Task 11: Code Submission (10 marks)

- [ ] **Complete**

### Requirements

| Requirement | How You'll Meet It |
|-------------|-------------------|
| **Clean** | Consistent formatting, no dead code |
| **Well-commented** | Docstrings for all functions |
| **Modular** | Separate functions for data loading, training, evaluation |
| **Reproducible** | Fixed random seeds, config cell at top |
| **Single notebook** | One `.ipynb` that regenerates all report figures |

### Notebook Structure
```
1. Config & Setup (seeds, parameters)
2. Data Loading (SUMPUBMED from Hugging Face)
3. Baseline Generation
4. Baseline Evaluation
5. Prompt Engineering Experiments
6. Prompt Evaluation
7. Fine-Tuning (BART + LoRA)
8. Fine-Tuning Evaluation
9. Comparison & Visualization
```

### Deliverable
One clean Jupyter notebook

---

## Quick Reference: Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     FULL PAPER TEXT                         │
│   (Introduction, Methods, Results, Discussion, Conclusion)  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   YOUR MODEL   │
              └───────┬───────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  GENERATED SUMMARY    │
          └───────────┬───────────┘
                      │
                      │  COMPARE (ROUGE, BERTScore)
                      ▼
          ┌───────────────────────┐
          │  ACTUAL ABSTRACT      │  ← Written by paper authors
          │  (Ground Truth)       │     (FREE gold standard!)
          └───────────────────────┘
```

---

## Notes

- **Report Word Limit:** 3000 words max
- **Academic Integrity:** AI tools cannot write report content (proofreading must be stated)
- **All external sources/models must be cited**
