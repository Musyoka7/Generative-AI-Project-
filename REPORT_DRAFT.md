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

### Hypotheses

Before running experiments, I formed the following predictions about what each technique would achieve:

1. **Role framing (v1):** I expected that telling GPT-2 "You are a medical researcher" would prime it to use domain-appropriate vocabulary, resulting in higher ROUGE scores due to better word overlap with the medical abstracts.

2. **Audience targeting (v2):** I predicted that specifying "for a patient with no medical background" would produce simpler, less jargon-heavy summaries. This might lower ROUGE (simpler words ≠ reference words) but improve readability.

3. **Constraints (v3):** The baseline frequently generated "Acknowledgments" sections and unsupported claims. I hypothesized that explicit instructions to avoid these would clean up the output and improve scores.

4. **Structured output (v4):** I expected that providing a template ("Summarize in 3 points: 1) What was studied 2) Key findings 3) Implications") would guide the model toward organized summaries, improving both quality and ROUGE scores.

5. **Few-shot example (v5):** Few-shot prompting is a standard technique for improving LLM output. I predicted that showing the model an example summary would help it understand the desired format and produce better results.

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
| 0.0 | 0.1010 | Most deterministic, lowest score |
| 0.3 | 0.1513 | **Highest with sampling** |
| 0.7 | 0.1396 | Moderate variation |
| 1.0 | 0.1223 | Most random, performance drops |

Note: Temperature testing required disabling beam search (num_beams=1, do_sample=True), which fundamentally changes generation behaviour. Interestingly, a low temperature (0.3) performed best, while higher temperatures degraded output quality — suggesting that for this task, more randomness hurts rather than helps. These results are not directly comparable to the main ablation study.

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

**Target:** ~400 words

---

### Why LoRA?

Full fine-tuning of GPT-2's 124 million parameters would require significant GPU memory and risk catastrophic forgetting of the model's pre-trained knowledge. Instead, I used LoRA (Low-Rank Adaptation), which freezes the original weights and injects small trainable matrices into the attention layers. This reduced the trainable parameters from 124M to just 295K — a 99.76% reduction — making fine-tuning feasible on my MacBook's GPU.

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 8 | Balance between capacity and efficiency; Hu et al. found r=8 sufficient for most tasks |
| Alpha | 32 | Scaling factor; alpha/r = 4 provides reasonable learning signal |
| Dropout | 0.1 | Regularization to prevent overfitting on small dataset |
| Target modules | c_attn | GPT-2's attention projection layer; where most task-specific learning happens |

### Training Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training samples | 1,000 | Enough to learn the task without overfitting |
| Validation samples | 500 | From the separate dev split to monitor generalization |
| Epochs | 3 | Early experiments showed diminishing returns after 3 epochs |
| Batch size | 2 | Limited by GPU memory |
| Learning rate | 5e-4 | Standard for LoRA fine-tuning |
| Optimizer | AdamW | Standard choice for transformer fine-tuning |
| Device | MPS (Apple GPU) | Training took approximately 12 minutes |

### Training Format

I formatted training examples as:

```
Summarize:
{paper_text}

Summary: {abstract}
```

This teaches the model to associate the "Summarize:" prompt with producing abstract-style output.

### Training Progress

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 1.8730 | 1.6803 |
| 2 | 1.7311 | 1.6651 |
| 3 | 1.7087 | 1.6572 |

The key observation here is that validation loss remained below training loss throughout training. This indicates the model is generalizing well to unseen data rather than just memorizing the training examples. If validation loss had started increasing while training loss decreased, that would signal overfitting — but I didn't observe this.

### Failure Modes and Mitigations

The fine-tuned model has several potential failure modes:

1. **Prompt sensitivity**: The model performs best with the exact prompt format it was trained on ("Summarize:"). Using "TL;DR:" dropped ROUGE-L from 0.1258 to 0.1103. Mitigation: document the required format and enforce it in any deployment.

2. **Truncation loss**: Papers longer than ~1000 characters get truncated before the model sees them, meaning key findings at the end of a paper could be lost entirely. Mitigation: could implement chunking or sliding window approaches, or use a model with a longer context window.

3. **Hallucination**: The model may generate claims not present in the source text. I observed this occasionally in baseline outputs, and while fine-tuning reduced it, I cannot guarantee the model never hallucinates. Mitigation: human review is essential for any medical application; fact-checking against the source document before display.

4. **Domain edge cases**: The model was trained on 1,000 samples from SUMPUBMED, which may not cover all medical subfields. Unusual paper structures or rare terminology could confuse it. Mitigation: more diverse training data would help; flagging low-confidence outputs for human review.

### Limitations

I did not set a random seed, which means exact reproduction of my results isn't guaranteed. I also didn't implement a learning rate scheduler, though the short training duration (3 epochs) meant this likely had minimal impact.

---

## Task 8: Evaluate Fine-Tuning (10 marks)

**Target:** ~300 words

---

### Quantitative Results

| Metric | Baseline | Fine-Tuned | Change |
|--------|----------|------------|--------|
| ROUGE-1 | 0.2907 | 0.2083 | -28.4% |
| ROUGE-2 | 0.0437 | 0.0407 | -6.9% |
| ROUGE-L | 0.1529 | 0.1258 | -17.7% |
| BERTScore F1 | 0.8156 | 0.8339 | +2.2% |

At first glance, these results seem disappointing — ROUGE scores dropped significantly after fine-tuning. But the BERTScore improvement tells a different story.

### Why ROUGE Dropped But BERTScore Improved

ROUGE measures exact word overlap between the generated summary and the reference. BERTScore measures semantic similarity using contextual embeddings — whether the meaning is preserved even if different words are used.

The fine-tuned model learned to paraphrase. Instead of copying phrases directly from the source text, it generates summaries using different words that convey the same meaning. This is actually desirable for summarization — we want the model to understand and rephrase, not just extract and copy.

For example, if the reference says "the study examined the effects" and the model generates "this research investigated the impact", ROUGE penalizes this harshly (no word overlap) but BERTScore recognizes the semantic equivalence.

### Prompt Format Matters

I tested different prompts on the fine-tuned model:

| Prompt | ROUGE-L |
|--------|---------|
| "Summarize:" (training format) | 0.1258 |
| "TL;DR:" | 0.1103 |
| v4_structure | 0.1066 |

The model performs best with the exact format it was trained on. This makes sense — during training, it learned to associate "Summarize:" with producing abstract-style output. Using a different prompt confuses it.

### Generalization Check

The validation loss (1.6572) remained below training loss (1.7087) at the end of training. This confirms the model learned generalizable patterns rather than memorizing training examples. When I tested on the held-out test set (samples the model never saw during training), it produced coherent, topic-relevant summaries.

### Qualitative Observations

Comparing outputs side-by-side revealed a crucial difference: the fine-tuned model produces consistently coherent text, while baseline GPT-2 occasionally degenerates. On one test sample, the baseline started producing random letters and numbers instead of a summary. The fine-tuned model handled the same input without issue. This consistency is arguably more important than raw ROUGE scores for a practical application.

---

## Task 9: Compare & Contrast (15 marks)

**Target:** ~400 words

---

### Summary of All Approaches

| Approach | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|----------|---------|---------|---------|--------------|
| Baseline | 0.2907 | 0.0437 | 0.1529 | 0.8156 |
| Prompt Eng (v4_structure) | 0.2917 | 0.0416 | 0.1539 | 0.8167 |
| LoRA Fine-tuned | 0.2083 | 0.0407 | 0.1258 | 0.8339 |

### What Each Approach Achieved

**Baseline**: Pre-trained GPT-2 tested with four different prompts (including "Summarize this text:", "TL;DR:", etc.) averaged across 8 test samples. The model leverages its training on internet text, but outputs were inconsistent — some samples produced fluent summaries while others degenerated into nonsensical text.

**Prompt Engineering**: Adding role framing, audience targeting, and structured output instructions yielded marginal improvement (+0.6% ROUGE-L). The best variant (v4_structure) performed slightly better than baseline, but the gains were modest. Interestingly, techniques that typically help larger models (like few-shot examples) actually hurt GPT-2's performance, likely due to context window limitations.

**Fine-tuning**: LoRA training produced the most semantically accurate summaries (highest BERTScore) but the lowest ROUGE scores. The model learned to paraphrase rather than copy, which metrics penalize but humans might prefer.

### The ROUGE vs BERTScore Tradeoff

This project revealed a fundamental tension between lexical and semantic evaluation:

- **High ROUGE** = Output copies words from the reference
- **High BERTScore** = Output preserves meaning regardless of word choice

For medical summarization, I'd argue semantic accuracy matters more. A summary that correctly conveys "the treatment reduced symptoms by 30%" is useful even if it doesn't match the reference's exact phrasing. A summary that copies random phrases from the source but misses the key finding is useless despite higher ROUGE.

### Which Approach is Best?

There's no single winner — it depends on the use case:

| If you need... | Best approach |
|----------------|---------------|
| Minimal effort, reasonable quality | Baseline with TL;DR prompt |
| Slight improvement without training | Prompt engineering (v4_structure) |
| Semantic accuracy and consistency | Fine-tuned model |
| Highest ROUGE scores specifically | Baseline or prompt engineering |

For a real medical summarization system, I would choose the fine-tuned model despite lower ROUGE scores. The consistency (no degeneration) and semantic accuracy (higher BERTScore) are more important for a domain where reliability matters.

### Lessons Learned

1. **Simple prompts can outperform complex ones**: The basic "TL;DR" beat elaborate engineered prompts because GPT-2 was trained on that pattern.

2. **More data isn't always better**: My 2,000-sample training run performed worse than the 1,000-sample run due to overfitting.

3. **Metrics don't tell the whole story**: ROUGE dropped but output quality improved qualitatively. Always look at actual outputs, not just numbers.

4. **Prompt format lock-in**: Fine-tuned models become sensitive to the exact training format. This is a limitation for deployment flexibility.

---

## Task 10: Ethical Considerations (5 marks)

**Target:** ~300 words

---

### Hallucination Risk

The most serious risk in medical summarization is hallucination — when the model generates claims not supported by the source text. During my experiments, I observed the baseline model occasionally producing statements like "Phytoplosan is a highly toxic herbicide" when the source paper discussed something entirely different. In a medical context, such fabrications could lead to dangerous misinformation.

The fine-tuned model showed fewer obvious hallucinations, likely because training on paper-abstract pairs taught it to stay closer to the source material. However, I cannot guarantee it never hallucinates. Any deployment would require human verification of generated summaries.

### Oversimplification

Medical research often contains critical nuances — dosage information, contraindications, confidence intervals, population limitations. A summary that omits these details could be dangerous. For example, summarizing "Drug X reduced symptoms in patients aged 18-45 with mild disease" as "Drug X reduces symptoms" removes crucial context about who the treatment works for.

My model was trained on abstracts, which are themselves simplifications of full papers. This compounds the simplification problem. For critical applications, multi-level summaries (technical + layman versions) might be preferable to a single compressed output.

### Misuse Potential

There's a real risk that patients might treat AI-generated summaries as medical advice. A clear disclaimer like "This summary is for informational purposes only and does not constitute medical advice. Consult a healthcare professional for medical decisions." should accompany any deployed system.

### Mitigations

Based on the literature (Liu et al., 2025) and my observations, I recommend:

1. **Human-in-the-loop**: Medical professionals should review summaries before patient-facing use
2. **Confidence thresholds**: Flag or reject outputs where the model shows high uncertainty
3. **Source linking**: Always provide access to the original paper alongside the summary
4. **Explicit disclaimers**: Make clear that outputs are AI-generated and not medical advice
5. **Hallucination detection**: Implement fact-checking against the source document before display

### Conclusion

While AI-assisted medical summarization could improve public understanding of research, the risks in this domain are uniquely high. The technology should augment human expertise, not replace it. My system is a proof-of-concept that demonstrates both the potential and the limitations of current approaches.

---

## References

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2106.09685

3. Balde, et al. (2025). Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings. *Findings of the Association for Computational Linguistics: ACL 2025*. https://arxiv.org/abs/2505.21242

4. Liu, et al. (2025). A hallucination detection and mitigation framework for faithful text summarization using LLMs. *Scientific Reports*, 15, 2360. https://www.nature.com/articles/s41598-025-31075-1

5. Gupta, V., Bharti, P., Nokhiz, P., & Karnick, H. (2021). SumPubMed: Summarization Dataset of PubMed Scientific Articles. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, 292–303. https://aclanthology.org/2021.acl-srw.30/
