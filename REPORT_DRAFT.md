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

### 1. BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., 2020)

Lewis et al. introduced BART, a denoising autoencoder combining a bidirectional encoder (similar to BERT) with an autoregressive decoder (similar to GPT). The model is pre-trained by corrupting text with various noising functions (token masking, deletion, text infilling, sentence permutation) and learning to reconstruct the original document.

**Strengths:** BART achieved state-of-the-art results on summarization benchmarks, improving by 6 ROUGE points on XSum. The encoder-decoder architecture enables full comprehension of input documents before generation, making it particularly effective for abstractive summarization.

**Limitations:** Performance varies across noising objectives, and the model can occasionally generate claims not fully supported by source documents.

**Design Takeaway:** BART is selected as the base model because its architecture is purpose-built for summarization, unlike decoder-only models which lack bidirectional input understanding.

---

### 2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022)

Hu et al. proposed LoRA, a parameter-efficient fine-tuning method that freezes pre-trained weights and injects small trainable low-rank matrices into transformer layers. This represents weight updates as a product of two small matrices (ΔW = BA), drastically reducing trainable parameters.

**Strengths:** LoRA reduces trainable parameters by 10,000x (e.g., from 175B to 37.7M for GPT-3) while matching or exceeding full fine-tuning quality. It introduces no additional inference latency since trained matrices can be merged with frozen weights.

**Limitations:** The choice of which weight matrices to adapt is heuristic-based, and batching inputs across different LoRA configurations is non-trivial.

**Design Takeaway:** LoRA enables fine-tuning BART on consumer hardware. Without it, training a 139M parameter model on medical data would require expensive GPU infrastructure.

---

### 3. Evaluation of LLMs in Medical Text Summarization (Balde et al., 2025)

Balde et al. investigated the challenges of medical text summarization, focusing on out-of-vocabulary (OOV) medical terminology. They found that standard tokenizers over-fragment medical terms (splitting them into 13-20% more subword pieces than general text), causing semantic loss and generation errors.

**Strengths:** The paper demonstrated that vocabulary adaptation improved ROUGE-L by 18.75% for Llama-2 and 14.82% for Llama-3.1. Human evaluators found adapted models produced more relevant (93.34% vs 70%) and faithful summaries.

**Limitations:** Results were limited to 7B/8B parameter models, and adaptation was less helpful for extractive summarization tasks.

**Design Takeaway:** This paper justifies fine-tuning on domain-specific data. Since BART was pre-trained on general text, fine-tuning on SUMPUBMED teaches it medical vocabulary and domain conventions.

---

### 4. Hallucination Detection and Mitigation Framework (Liu et al., 2025)

Liu et al. addressed the critical problem of hallucination in summarization — when models generate content not supported by source documents. They proposed a Question-Sorting-Evaluation (Q-S-E) framework that detects factual inconsistencies and iteratively refines summaries.

**Strengths:** Tested on PubMed data, achieving the highest factual consistency (FactCC: 37.78) among baselines including ChatGPT and GPT-3. The framework provides transparency by showing exactly which summary segments were corrected.

**Limitations:** Iterative refinement has diminishing returns beyond approximately 6 iterations, and temporal accuracy remains challenging.

**Design Takeaway:** Hallucination is a critical risk for medical summarization. This paper informs the ethics section — we acknowledge the risk and discuss mitigations including disclaimers, fact-checking against source documents, and the necessity of human review for clinical applications.

---

### Summary

These four papers collectively justify the project pipeline:

1. **BART** provides an architecture optimized for summarization
2. **LoRA** makes fine-tuning feasible on limited hardware
3. **Medical OOV research** demonstrates why domain-specific training matters
4. **Hallucination research** highlights risks requiring ethical consideration

Together, they establish an evidence-based approach to medical text summarization that balances performance with practical and ethical constraints.

---

## Task 3: Generate Baseline (5 marks)

*To be completed*

---

## Task 4: Evaluate Baseline (5 marks)

*To be completed*

---

## Task 5: Prompt Engineering (15 marks)

*To be completed*

---

## Task 6: Evaluate Prompts (5 marks)

*To be completed*

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

1. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 7871–7880. https://arxiv.org/abs/1910.13461

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2106.09685

3. Balde, et al. (2025). Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings. *Findings of the Association for Computational Linguistics: ACL 2025*. https://arxiv.org/abs/2505.21242

4. Liu, et al. (2025). A hallucination detection and mitigation framework for faithful text summarization using LLMs. *Scientific Reports*, 15, 2360. https://www.nature.com/articles/s41598-025-31075-1

5. Gupta, V., Bharti, P., Nokhiz, P., & Karnick, H. (2021). SumPubMed: Summarization Dataset of PubMed Scientific Articles. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*, 292–303. https://aclanthology.org/2021.acl-srw.30/
