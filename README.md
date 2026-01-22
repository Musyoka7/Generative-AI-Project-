# Medical Research Summarization with GPT-2

A coursework project for **Generative and Collaborative AI** that develops and evaluates a medical research summarization system using GPT-2, prompt engineering, and LoRA fine-tuning.

## Project Overview

This project addresses the challenge of making medical research accessible to patients and non-experts. Medical papers contain complex terminology and domain-specific language that creates a knowledge gap between published findings and public understanding.

### Approach

Three summarization approaches are compared:

| Approach | Description | ROUGE-L | BERTScore F1 |
|----------|-------------|---------|--------------|
| **Baseline** | Pre-trained GPT-2 with simple prompts | 0.1529 | 0.8156 |
| **Prompt Engineering** | Systematic ablation of role, audience, constraints, structure | 0.1539 | 0.8167 |
| **LoRA Fine-tuning** | Parameter-efficient fine-tuning on medical abstracts | 0.1258 | 0.8339 |

### Key Finding

Fine-tuning **decreased ROUGE but increased BERTScore**. This indicates the model learned to paraphrase (preserving meaning with different words) rather than copying exact phrases. For medical summarization, semantic accuracy may be more important than lexical overlap.

## Repository Structure

```
.
├── Code.ipynb                 # Main Jupyter notebook with all experiments
├── REPORT_DRAFT.md            # Written report (3000 words max)
├── lora_gpt2_medical/         # Saved LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── papers/                    # Literature review sources
│   ├── LoRA_Hu_2022.pdf
│   ├── Hallucination_2025.pdf
│   ├── Medical_OOV_Lu_2025.pdf
│   └── ...
├── Generative and Collaborative AI.pdf  # Coursework specification
└── PROJECT_PLAN.md            # Initial project planning
```

## Setup & Requirements

### Prerequisites

- Python 3.11+


### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets peft rouge-score bert-score numpy pandas tqdm matplotlib
```

### Dataset

The project uses [SumPubMed](https://huggingface.co/datasets/scientific_papers) — a dataset of PubMed scientific articles with their abstracts as ground truth summaries.

```python
from datasets import load_dataset
dataset = load_dataset("ccdv/pubmed-summarization")
```

## Running the Notebook

Open `Code.ipynb` in Jupyter and run cells sequentially. The notebook is organized by task:

| Task | Description | Key Output |
|------|-------------|------------|
| Task 3 | Generate baseline summaries | 8 outputs with 4 different prompts |
| Task 4 | Evaluate baseline | ROUGE & BERTScore metrics |
| Task 5 | Prompt engineering | 5 prompt versions + temperature tests |
| Task 6 | Evaluate prompts | Ablation table showing v4_structure wins |
| Task 7 | LoRA fine-tuning | Trained adapter saved to `lora_gpt2_medical/` |
| Task 8 | Evaluate fine-tuned model | Comparison with baseline |
| Task 9 | Compare all approaches | Comprehensive analysis |

### Hardware Notes

- **Device**: Runs on MPS (Apple Silicon) or CUDA
- **Training time**: ~12 minutes for 3 epochs on M1 Mac
- **Memory**: Batch size 2 to fit in GPU memory

## Model Configuration

### LoRA Settings

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target modules | c_attn |
| Trainable parameters | 294,912 (0.24% of total) |

### Training Settings

| Parameter | Value |
|-----------|-------|
| Training samples | 1,000 |
| Validation samples | 500 |
| Epochs | 3 |
| Learning rate | 5e-4 |
| Batch size | 2 |

## Results Summary

### Prompt Engineering Ablation

| Version | Change | ROUGE-L | Effect |
|---------|--------|---------|--------|
| Baseline | — | 0.1529 | — |
| v1_role | Role framing | 0.1424 | -0.0105 |
| v2_audience | + Audience | 0.1442 | +0.0019 |
| v3_constraints | + Constraints | 0.1392 | -0.0050 |
| v4_structure | + Structure | 0.1539 | +0.0146 |
| v5_fewshot | + Few-shot | 0.1229 | -0.0310 |

**Key insight**: Prompt stacking doesn't guarantee improvement. Few-shot prompting actually hurt performance due to context window limitations.

### Fine-tuning Results

| Metric | Baseline | Fine-Tuned | Change |
|--------|----------|------------|--------|
| ROUGE-L | 0.1529 | 0.1258 | -17.7% |
| BERTScore F1 | 0.8156 | 0.8339 | +2.2% |

The model learned to paraphrase rather than copy, improving semantic accuracy at the cost of lexical overlap.

## Ethical Considerations

- **Hallucination risk**: Models may generate unsupported medical claims
- **Oversimplification**: Critical nuances (dosages, contraindications) may be lost
- **Not medical advice**: Any deployment requires human review and clear disclaimers

## References

1. Radford et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
2. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
3. Balde et al. (2025). Evaluation of LLMs in Medical Text Summarization. ACL.
4. Liu et al. (2025). Hallucination Detection and Mitigation Framework. Scientific Reports.
5. Gupta et al. (2021). SumPubMed: Summarization Dataset of PubMed Scientific Articles. ACL-SRW.


