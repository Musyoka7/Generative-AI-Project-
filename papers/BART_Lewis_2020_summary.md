# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

**Authors:** Lewis et al. (2020)  
**Venue:** ACL 2020  
**Link:** [arXiv](https://arxiv.org/abs/1910.13461)

---

## 1. Methodology

### Architecture
BART is a denoising autoencoder built with a standard sequence-to-sequence Transformer architecture. It combines:
- **Bidirectional encoder** (similar to BERT)
- **Autoregressive (left-to-right) decoder** (similar to GPT)
Additional architectural details from the paper:
- Uses GeLU activations and parameter initialization from `N(0, 0.02)`
- Base model uses 6 encoder/6 decoder layers; large uses 12/12
- Decoder includes cross-attention over the encoder, and BART has ~10% more parameters than a similarly sized BERT

### Pre-training Approach (Denoising)
The model is trained to reconstruct original text from documents that have been corrupted with a noising function. It optimizes the negative log-likelihood of the original document.

### Noising Techniques
BART uses several transformations, including:
- **Token Masking/Deletion:** Replacing random tokens with `[MASK]` or deleting them entirely
- **Text Infilling:** Replacing arbitrary length spans of text (including zero length) with a single `[MASK]` token
- **Sentence Permutation:** Shuffling the order of sentences in a document
- **Document Rotation:** Rotating the document to start at a random token
The best-performing pre-training setup combines **text infilling + sentence permutation** (used in the large model experiments).

### Tokenization
Documents are tokenized with the same byte-pair encoding (BPE) as GPT-2.

### Difference from GPT/BERT
- Unlike BERT, which predicts masked tokens independently, BART's autoregressive decoder allows it to model dependencies between predicted tokens
- Unlike GPT, which only has a left-to-right context, BART's bidirectional encoder allows it to see the full corrupted input

---

## 2. Strengths

- **Generation Excellence:** Particularly effective for text generation tasks (summarization, dialogue, abstractive QA) where the model must manipulate input information
- **Flexibility:** The sequence-to-sequence setup allows for arbitrary noise transformations, including those that change the total length of the text
- **Versatility:** Matches the performance of RoBERTa on natural language comprehension tasks (GLUE, SQuAD) while outperforming it on generation
- **Qualitative Outputs:** Generations are fluent and highly abstractive, often integrating evidence across the input document

---

## 3. Limitations

- **Task Dependency:** Performance of specific noising objectives varies significantly across tasks; for example, rotating documents or permuting sentences performs poorly in isolation
- **Loose Constraints:** In tasks like ELI5 where the output is only loosely constrained by the input, pure language models can still outperform BART
- **Overfitting:** In machine translation experiments, the approach was found to be prone to overfitting and less effective without the addition of back-translation data
- **Occasional Unsupported Claims:** Qualitative analysis includes an example where a generated claim is not supported by the source

---

## 4. Key Findings

### Summarization Benchmarks
Achieved new state-of-the-art results on **CNN/DailyMail** and **XSum**. Notably, it improved performance by **6 ROUGE points** over previous work on the highly abstractive XSum dataset.

### Benchmark Performance
| Dataset | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|---------|---------|---------|
| XSum | 45.14 | 22.27 | 37.25 |
| CNN/DM | 44.16 | 21.28 | 40.90 |

### Other Results
- **Machine Translation:** Provided a 1.1 BLEU increase over a strong back-translation baseline on the WMT Romanian-English benchmark
- **Comprehension:** Achieved comparable results to RoBERTa on GLUE and SQuAD, proving that the generative decoder does not degrade discriminative performance
- **Ablation Insight:** Text infilling is the most consistently strong noising function; token deletion can outperform masking on generation tasks

---

## Design Takeaway

BART is chosen as the base model for this project because its encoder-decoder architecture is specifically designed for summarization tasks. Unlike decoder-only models like GPT-2, BART first reads and understands the entire input document (encoder), then generates the summary (decoder). This approach achieved +6 ROUGE improvement over prior state-of-the-art on the XSum benchmark, demonstrating its effectiveness for abstractive summarization. The pre-training objective (reconstructing corrupted text) closely aligns with the summarization task of condensing and rewriting information.
