# Generative and Collaborative AI - Report Draft

**Word Limit:** 3000 words max  
**Status:** Draft - In Progress

---

## Task 1: Coursework Goal (5 marks)

**Target:** ~200 words

---

This project develops a medical research summarization system that condenses PubMed scientific articles into accessible summaries for patients and non-experts.

### Problem Statement

Patients and the general public struggle to understand clinical research publications. Medical papers contain complex terminology, statistical analyses, and domain-specific language that creates a knowledge gap between published findings and public understanding. This gap can lead to misinformation, misinterpretation of health advice, and reduced patient engagement with evidence-based medicine.

### Objective

Build and evaluate a fine-tuned summarization model that produces accurate, readable summaries of medical research papers. The project compares three approaches: baseline pre-trained generation, prompt engineering, and domain-specific fine-tuning using LoRA.

### Success Criteria

- ROUGE-L score > 0.40 on held-out test set
- BERTScore F1 > 0.80
- Generated summaries preserve key medical findings without hallucination

### Risk Assessment

The primary risks are hallucination (generating unsupported medical claims) and oversimplification (losing critical information like dosages or contraindications). These are addressed through evaluation metrics, extractive constraints, and explicit disclaimers that outputs are not medical advice.

---

## Task 2: Literature Review (10 marks)

**Target:** ~600-800 words

---

This section reviews four key papers that inform the technical decisions in this project: model architecture, fine-tuning methodology, domain-specific challenges, and ethical considerations.

### 1. GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)

Radford et al. introduced GPT-2, a large-scale transformer-based language model trained on diverse internet text. The model uses a decoder-only architecture and is trained with a simple objective: predict the next word given all previous words. This unsupervised pre-training enables the model to perform various tasks through prompting without task-specific fine-tuning.

**Strengths:** GPT-2 demonstrates strong zero-shot and few-shot capabilities across tasks including summarization, translation, and question answering. Its autoregressive nature allows it to generate coherent, fluent text that responds to prompt variations.

**Limitations:** As a decoder-only model, GPT-2 lacks bidirectional context understanding. It processes text left-to-right, which can limit comprehension of long documents. The model is also prone to hallucination, generating plausible-sounding but unsupported claims.

**Design Takeaway:** GPT-2 is selected as the base model because it responds to natural language prompts, enabling systematic comparison of prompt engineering techniques. This aligns with the coursework emphasis on demonstrating prompt-based improvements. While encoder-decoder models like BART achieve higher summarization quality, GPT-2's prompt sensitivity makes it ideal for demonstrating the full pipeline from baseline to fine-tuned performance.

---

### 2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022)

Hu et al. proposed LoRA, a parameter-efficient fine-tuning method that freezes pre-trained weights and injects small trainable low-rank matrices into transformer layers. This represents weight updates as a product of two small matrices (ΔW = BA), drastically reducing trainable parameters.

**Strengths:** LoRA reduces trainable parameters by 10,000x (e.g., from 175B to 37.7M for GPT-3) while matching or exceeding full fine-tuning quality. It introduces no additional inference latency since trained matrices can be merged with frozen weights.

**Limitations:** The choice of which weight matrices to adapt is heuristic-based, and batching inputs across different LoRA configurations is non-trivial.

**Design Takeaway:** LoRA enables fine-tuning GPT-2 on consumer hardware. Without it, adapting even a 124M parameter model on medical data would require significant computational resources.

---

### 3. Evaluation of LLMs in Medical Text Summarization (Balde et al., 2025)

Balde et al. investigated the challenges of medical text summarization, focusing on out-of-vocabulary (OOV) medical terminology. They found that standard tokenizers over-fragment medical terms (splitting them into 13-20% more subword pieces than general text), causing semantic loss and generation errors.

**Strengths:** The paper demonstrated that vocabulary adaptation improved ROUGE-L by 18.75% for Llama-2 and 14.82% for Llama-3.1. Human evaluators found adapted models produced more relevant (93.34% vs 70%) and faithful summaries.

**Limitations:** Results were limited to 7B/8B parameter models, and adaptation was less helpful for extractive summarization tasks.

**Design Takeaway:** This paper justifies fine-tuning on domain-specific data. Since GPT-2 was pre-trained on general internet text, fine-tuning on SUMPUBMED teaches it medical vocabulary and domain conventions.

---

### 4. Hallucination Detection and Mitigation Framework (Liu et al., 2025)

Liu et al. addressed the critical problem of hallucination in summarization — when models generate content not supported by source documents. They proposed a Question-Sorting-Evaluation (Q-S-E) framework that detects factual inconsistencies and iteratively refines summaries.

**Strengths:** Tested on PubMed data, achieving the highest factual consistency (FactCC: 37.78) among baselines including ChatGPT and GPT-3. The framework provides transparency by showing exactly which summary segments were corrected.

**Limitations:** Iterative refinement has diminishing returns beyond approximately 6 iterations, and temporal accuracy remains challenging.

**Design Takeaway:** Hallucination is a critical risk for medical summarization. This paper informs the ethics section — we acknowledge the risk and discuss mitigations including disclaimers, fact-checking against source documents, and the necessity of human review for clinical applications.

---

### Summary

These four papers collectively justify the project pipeline:

1. **GPT-2** provides a prompt-responsive model suitable for demonstrating iterative improvements
2. **LoRA** makes fine-tuning feasible on limited hardware
3. **Medical OOV research** demonstrates why domain-specific training matters
4. **Hallucination research** highlights risks requiring ethical consideration

Together, they establish an evidence-based approach to medical text summarization that balances performance with practical and ethical constraints.

---

## Task 3: Generate Baseline (5 marks)

**Target:** ~150 words

---

### Model Configuration

The baseline uses GPT-2 (124M parameters) from Hugging Face Transformers v4.40.0 with the following inference parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max input tokens | 800 | Leave room for generation within 1024 context window |
| Max new tokens | 150 | Sufficient for abstract-length summaries |
| Num beams | 4 | Beam search for coherent output |
| No repeat ngram size | 3 | Prevent repetitive phrases |
| Do sample | False | Deterministic output for reproducibility |

### Prompt Design

Four baseline prompts were tested, each with a simple variation to observe how minor wording changes affect output:

1. **"Summarize this text:"** — Minimal instruction to test default behaviour
2. **"Summarize the following medical research:"** — Adds domain context
3. **"Provide a brief summary:"** — Length cue to encourage conciseness
4. **"TL;DR:"** — Common internet shorthand GPT-2 recognises from training

### Outputs

Eight representative summaries were generated (two per prompt) using fixed sample indices [0, 100, 500, 1000, 1500, 2000, 2500, 3000] from the test set for reproducibility. Outputs demonstrated that GPT-2 generates fluent text but often includes hallucinated content and generic filler phrases like "Acknowledgments" sections not present in the source.

---

## Task 4: Evaluate Baseline (5 marks)

**Target:** ~150 words

---

### Metrics

Two complementary metrics were used:

- **ROUGE** (1, 2, L): Measures word overlap between generated and reference text
- **BERTScore**: Measures semantic similarity using contextual embeddings

### Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| ROUGE-1 | 0.2907 | 29% single word overlap |
| ROUGE-2 | 0.0437 | 4% two-word phrase overlap |
| ROUGE-L | 0.1529 | 15% longest matching sequence |
| BERTScore F1 | 0.8156 | 82% semantic similarity |

### Analysis

The low ROUGE scores indicate GPT-2 does not reproduce the exact phrasing of ground truth abstracts. This is expected — GPT-2 is a general language model, not a summarisation specialist. The relatively higher BERTScore (0.82) suggests generated text is semantically related to the source, even when exact words differ.

Notably, the "TL;DR" prompt achieved the highest ROUGE-L scores (0.171, 0.209), likely because this pattern appears frequently in GPT-2's training data. This observation informs prompt engineering strategies.

The baseline establishes a starting point: ROUGE-L of 0.15 and BERTScore of 0.82 are the benchmarks to improve through prompt engineering and fine-tuning.

---

## Task 5: Prompt Engineering (15 marks)

**Target:** ~400-500 words

---

### Approach: Systematic Ablation Study

Rather than testing random prompt variations, this section uses a **controlled ablation study** where each prompt version adds exactly one element to the previous version. This isolates the effect of each technique and reveals which components contribute to (or harm) performance.

### Prompt Versions

| Version | Addition | Cumulative Template |
|---------|----------|---------------------|
| v1_role | Role framing | "You are a medical researcher. Summarize this research:" |
| v2_audience | + Audience targeting | Adds "...for a patient with no medical background" |
| v3_constraints | + Constraints | Adds "Only include factual claims. Do not include acknowledgments or references." |
| v4_structure | + Structured output | Adds "Summarize in 3 points: 1) What was studied 2) Key findings 3) Implications" |
| v5_fewshot | + Few-shot example | Adds a worked example before the input |

### Rationale for Each Technique

1. **Role framing (v1):** Primes the model to adopt domain-appropriate vocabulary and tone. Based on research showing LLMs respond to persona instructions.

2. **Audience targeting (v2):** Aims to reduce jargon and increase accessibility by specifying the intended reader.

3. **Constraints (v3):** Addresses observed baseline problems — GPT-2 frequently generated "Acknowledgments" sections and unsupported claims. Explicit constraints attempt to suppress these.

4. **Structured output (v4):** Provides a template that guides the model toward organized, focused summaries rather than freeform generation.

5. **Few-shot example (v5):** The standard technique for improving LLM output by demonstrating the desired format.

### Generation Parameters

All experiments used identical parameters to the baseline for fair comparison:
- Max new tokens: 150
- Beam search: 4 beams
- No repeat n-gram size: 3
- Deterministic (do_sample=False)

### Temperature Variation

Additionally, temperature was tested on the best-performing prompt (v4_structure) to explore the creativity-accuracy tradeoff:

| Temperature | ROUGE-L | Observation |
|-------------|---------|-------------|
| 0.0 | 0.1010 | Most deterministic but lowest score |
| 0.3 | 0.1118 | Slight improvement |
| 0.7 | 0.1185 | More variation |
| 1.0 | 0.1278 | Highest with sampling |

Note: Temperature testing required disabling beam search (num_beams=1, do_sample=True), which fundamentally changes generation behaviour. These results are not directly comparable to the main ablation study.

### Key Finding: Stacking Can Hurt

The critical insight is that **prompt stacking does not guarantee improvement**. When v1_role performed poorly (dropping from baseline 0.1529 to 0.1424 ROUGE-L), all subsequent versions inherited this degraded starting point. The few-shot example (v5) performed worst of all — likely because GPT-2's limited context window was consumed by the example, leaving less room for the actual input, and the model sometimes "continued" the example rather than summarizing the new text.

---

## Task 6: Evaluate Prompts (5 marks)

**Target:** ~200-250 words

---

### Ablation Results

| Version | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Δ R-L vs Previous |
|---------|---------|---------|---------|--------------|-------------------|
| Baseline | 0.2907 | 0.0437 | 0.1529 | 0.8156 | — |
| v1_role | 0.2503 | 0.0389 | 0.1424 | 0.8071 | -0.0105 |
| v2_audience | 0.2588 | 0.0360 | 0.1442 | 0.8102 | +0.0019 |
| v3_constraints | 0.2608 | 0.0305 | 0.1392 | 0.8122 | -0.0050 |
| v4_structure | 0.2917 | 0.0416 | 0.1539 | 0.8167 | +0.0146 |
| v5_fewshot | 0.1966 | 0.0194 | 0.1229 | 0.7932 | -0.0310 |

### Analysis

**Best performer: v4_structure** achieved ROUGE-L 0.1539, a 0.6% improvement over baseline. While modest, this represents the ceiling of what prompt engineering alone can achieve with GPT-2 on this task.

**Worst performer: v5_fewshot** (ROUGE-L 0.1229) demonstrates that few-shot prompting is not universally beneficial. For GPT-2 with its 1024-token context limit, including an example consumed ~200 tokens, leaving less room for the input document. The model also showed confusion, sometimes continuing the example's content rather than summarizing the new input.

**Role framing backfired:** v1_role dropped performance from baseline, suggesting that "You are a medical researcher" may have biased GPT-2 toward overly technical or verbose output not matching the ground truth abstracts.

**Constraints had mixed effects:** While v3_constraints successfully reduced "Acknowledgments" generation (a qualitative improvement), it slightly decreased ROUGE-L, possibly because the constraint text consumed context tokens.

### Non-Stacked Validation

To verify whether stacking caused the poor results, individual techniques were tested in isolation:

| Prompt | ROUGE-L | vs Baseline |
|--------|---------|-------------|
| Baseline (TL;DR) | 0.1529 | — |
| Constraints only | 0.1423 | -0.0106 |
| Structure only | 0.1472 | -0.0058 |

**Both techniques performed worse than baseline when used alone.** This confirms that the issue is not stacking — the techniques themselves do not help GPT-2. The simple "TL;DR" prompt outperforms all engineered variants, likely because this pattern appears frequently in GPT-2's training data.

### Conclusion

Prompt engineering achieved marginal improvement (+0.6% ROUGE-L) but cannot compensate for GPT-2's fundamental limitations as a general-purpose language model not trained for summarization. Fine-tuning on domain-specific data (Task 7) is expected to yield larger gains.

---

## Task 7: Fine-Tune Generator (15 marks)

*To be completed*

---

## Task 8: Evaluate Fine-Tuning (10 marks)

*To be completed*

---

## Task 9: Compare & Contrast (15 marks)

*To be completed*

---

## Task 10: Ethical Considerations (5 marks)

*To be completed*

---

## References

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2106.09685

3. Balde, et al. (2025). Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings. *Findings of the Association for Computational Linguistics: ACL 2025*. https://arxiv.org/abs/2505.21242

4. Liu, et al. (2025). A hallucination detection and mitigation framework for faithful text summarization using LLMs. *Scientific Reports*, 15, 2360. https://www.nature.com/articles/s41598-025-31075-1

5. Gupta, V., Bharti, P., Nokhiz, P., & Karnick, H. (2021). SumPubMed: Summarization Dataset of PubMed Scientific Articles. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, 292–303. https://aclanthology.org/2021.acl-srw.30/
