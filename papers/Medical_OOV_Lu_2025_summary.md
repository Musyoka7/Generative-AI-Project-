# Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings

**Authors:** Balde et al. (2025)  
**Venue:** ACL Findings 2025  
**Link:** [arXiv](https://arxiv.org/abs/2505.21242)

---

## 1. Methodology

### Problem Addressed
The paper addresses the lack of fine-grained evaluation of Large Language Models (LLMs) in "difficult" medical summarization settings—specifically scenarios with high concentrations of **Out-of-Vocabulary (OOV)** words and **High Novelty** terms.

### Definition of OOV
OOV is defined as medical terms that are **over-fragmented** by standard tokenizers (split into more than one token, or "more than thrice" for "Difficult-OOV"). This fragmentation leads to:
- Loss of semantic meaning
- Increased complexity of generating correct medical terms

### Vocabulary Adaptation Approach
Three-step process:
1. **Candidate Generation:** Extracting tokens from domain-specific corpora (PubMed Abstract Collection) and target task datasets
2. **Selection:** Using a utility function (e.g., fragment score—the average number of subwords per word) to choose the most beneficial tokens
3. **Continual Pretraining (CPT):** Learning new embeddings for added tokens using LoRA via two strategies:
   - **End-to-End:** Simultaneous training
   - **Two-Stage:** Training embeddings first, then unfreezing LoRA adapters
   - **Embedding Initialization:** New token embeddings are initialized as the average of their existing subword embeddings

### Specific Adaptation Methods
- **MEDVOC-LLM:** A cleaned version of the state-of-the-art MEDVOC, removing non-medical/noisy tokens
- **ScafFix (Proposed):** A new strategy that directly adds whole medical words and uses a custom AdaptBPE scheme to prevent "scaffolding tokens" (under-trained intermediate subwords)

### LoRA Configuration
Continual pretraining applies LoRA to all linear modules (attention and MLP) with a fixed configuration of **rank 32** and **alpha 64**.

---

## 2. Strengths

- **Quantifiable Improvements:** Vocabulary adaptation achieved an average performance boost in ROUGE-L of:
  - **18.75%** for Llama-2
  - **14.82%** for Llama-3.1 across medical datasets
- **Human-Verified Quality:** Medical experts found adapted models produced more:
  - **Relevant:** 93.34% vs. 70% for BASE
  - **Faithful:** 93.34% vs. 83.34% for BASE
- **Key Insights for Medical Summarization:**
  - Even "large-vocabulary" models like Llama-3.1 (128K tokens) still suffer from severe over-fragmentation of medical terms
  - ScafFix is particularly effective for "High Novelty" settings because it preserves morphological boundaries better than standard BPE

---

## 3. Limitations

- **Hardware & Scale Constraints:** Due to resource limits, only 7B/8B parameter models tested; effects on larger variants (13B/70B) unexplored
- **Extractive vs. Abstractive Limits:** Adaptation was less helpful for extractive summaries or datasets with low novelty/OOV (e.g., BioASQ-S), where models can simply copy text
- **Safety & Deployment:** Models still exhibit hallucinations and lack the safety alignment required for high-stakes clinical deployment
- **LoRA vs. Full Tuning:** Results were generated using parameter-efficient tuning (LoRA); full-scale fine-tuning might yield different outcomes
- **Language Scope:** The study focuses on English medical text, while much prior vocabulary expansion work targets non-English settings

---

## 4. Key Findings

### Impact of Medical Terms on General Models
- Medical terms significantly increase the "fragment score" (average tokens per word) by **13% to 20%** compared to general text
- This fragmentation causes "semantic loss" and forces models to generate more subwords, increasing the probability of errors

### Vocabulary Adaptation Results
- Adapted models outperformed the BASE model and CPT-Only (continual pretraining without vocab expansion) in **7 out of 8** overall evaluation settings
- **ScafFix** reduced the fragment score by **30.83%** over BASE while using the smallest amount of added vocabulary
- Generalization tests showed these adaptation benefits hold true across different model families, including Mistral and Qwen-2

---

## Design Takeaway

This paper justifies fine-tuning on domain-specific medical data. General language models break medical terminology into too many pieces (13-20% more fragmentation than normal text), causing semantic loss and generation errors. Since BART was pre-trained on general internet text, it will struggle with medical vocabulary. Fine-tuning on SUMPUBMED (32,000+ medical paper-abstract pairs) addresses this by teaching the model medical terminology and writing patterns. The paper's finding that even large-vocabulary models suffer from this problem reinforces that domain-specific training is necessary.
