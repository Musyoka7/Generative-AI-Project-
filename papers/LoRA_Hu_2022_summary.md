# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Hu et al. (2022)  
**Venue:** ICLR 2022  
**Link:** [arXiv](https://arxiv.org/abs/2106.09685)

---

## 1. Methodology

### Core Concept
LoRA freezes the pre-trained model weights (W₀) and injects trainable rank decomposition matrices into each layer of the Transformer architecture.

### Mechanism
It represents the weight update ΔW as a product of two low-rank matrices:
```
W₀ + ΔW = W₀ + BA
```
Where:
- B ∈ ℝ^(d × r)
- A ∈ ℝ^(r × k)
- Rank r << min(d, k)

### Forward Pass
The modified forward pass is: `h = W₀x + BAx`. Both the frozen and trainable branches receive the same input, and their outputs are summed coordinate-wise.

### Initialization
- Matrix A uses random Gaussian initialization
- Matrix B is initialized to zero
- This ensures ΔW = BA is zero at the start of training

### Key Technical Components
- **Rank (r):** The dimension of the low-rank matrices. Experiments show that a very low rank (e.g., r=1 or r=2) is often sufficient
- **Alpha (α):** A scaling factor for ΔW. The update is scaled by α/r

### Difference from Full Fine-Tuning
Full fine-tuning updates all model parameters, whereas LoRA only updates the low-rank matrices, requiring far fewer trainable parameters. LoRA can "recover" full fine-tuning expressivity as r increases.

---

## 2. Strengths

- **Memory Efficiency:** Reduces VRAM consumption during training by up to **3x** (e.g., from 1.2TB to 350GB for GPT-3 175B)
- **Storage Efficiency:** Reduces checkpoint sizes by roughly **10,000x** (e.g., from 350GB to 35MB for GPT-3)
- **Training Speed:** Increases training throughput by approximately **25%** as gradients do not need to be calculated for the vast majority of parameters
- **Inference Latency:** Introduces **no additional inference latency** because the trained low-rank matrices can be merged with the frozen weights upon deployment
- **Quality:** Performs on-par with or better than full fine-tuning across various models (RoBERTa, DeBERTa, GPT-2, GPT-3) and tasks

---

## 3. Limitations

- **Batching Constraints:** Not straightforward to batch inputs for different tasks (each with different LoRA modules) in a single forward pass if weights are merged
- **Heuristic Selection:** The choice of which weight matrices to apply LoRA to is currently based on heuristics (primarily attention weights)

---

## 4. Key Findings

### Parameter Reduction
For GPT-3 175B, LoRA reduces trainable parameters:
- From **175 Billion** to **37.7 Million** (r=8) — 10,000x reduction
- Or down to **4.7 Million** (r=1)

### Performance Comparison
| Model | Task | LoRA | Full Fine-Tuning |
|-------|------|------|------------------|
| GPT-3 175B | WikiSQL | 73.4 | 73.8 |
| GPT-3 175B | MultiNLI-m | 91.7 | 89.5 |
| GPT-3 175B | SAMSum | 53.8 | 52.0 |
| RoBERTa-base | GLUE Avg | 87.2 | 86.4 |

RoBERTa-base LoRA: only 0.3M trainable parameters vs 125M for full fine-tuning.

### Intrinsic Rank Finding
The paper finds that weight matrix updates during adaptation have a very low "intrinsic rank," explaining why a small r works so effectively.

---

## Design Takeaway

LoRA is used for fine-tuning in this project because it enables training on consumer hardware without sacrificing quality. Instead of updating all 139 million parameters in BART (requiring expensive GPUs), LoRA freezes the original weights and only trains ~300,000 new parameters — a 10,000x reduction. The paper demonstrates that this achieves performance equal to or better than full fine-tuning across multiple benchmarks. This makes fine-tuning BART on the SUMPUBMED dataset feasible on a laptop.
