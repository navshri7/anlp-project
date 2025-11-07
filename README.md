# Memory-Enhanced Transformer for Dialect-to-Standard Language Translation

This repository contains the code and results for a project on translating low-resource Bengali dialects (Chittagonian and Barendri) into Standard Bangla. The core of this work is a novel **Memory-Enhanced Transformer model** that improves upon a standard fine-tuned baseline by integrating a specialized memory module. This module is trained to focus on specific, challenging tokens, leading to significant gains in translation quality and impressive generalization to unseen dialects.

## Table of Contents
- [Project Overview](#project-overview)
- [Results](#results)
  - [In-Domain Performance: Chittagonian-to-Bangla](#in-domain-performance-chittagonian-to-bangla)
  - [Zero-Shot Transfer Performance: Barendri-to-Bangla](#zero-shot-transfer-performance-barendri-to-bangla)
  - [Overall Key Findings](#overall-key-findings)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Installation](#2-installation)
  - [3. Data Preparation](#3-data-preparation)
  - [4. Training and Evaluation](#4-training-and-evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Codebase Description](#codebase-description)
- [Ablation Studies](#ablation-studies)
- [Datasets](#datasets)
- [Citation](#citation)

## Project Overview

Translating between closely related language variants, such as dialects and a standard language, presents unique challenges. While general-purpose models can be fine-tuned for this task, they often struggle with dialect-specific vocabulary, idiomatic expressions, and rare words.

This project introduces a **Memory-Enhanced Transformer** to address these challenges. The methodology consists of a two-stage process:

1.  **Baseline Model Training**: We first fine-tune a standard `google/mt5-small` model on a source dialect (Chittagonian). This serves as our `model_backbone`.
2.  **Memory Enhancement**: We then augment this baseline model with an external memory module. This involves:
    *   **Feature Extraction**: Analyzing the trained baseline to extract rich token-level features, including embeddings, cross-attention scores, and corpus frequency.
    *   **Memory Module Training**: Training a small Feed-Forward Neural Network (FFNN) on subsets of these tokens. We perform ablation studies to determine which tokens are most beneficial to "memorize."
    *   **Integration**: Integrating the trained FFNN memory module into the main transformer architecture. A learnable gate controls the flow of information from the memory, allowing the model to dynamically decide when to rely on its specialized knowledge.

This approach allows the model to better handle the "long tail" of the vocabulary distribution, effectively giving it an expert system for translating difficult or infrequent words.

## Results

We evaluated our models on two distinct tasks: an in-domain task (Chittagonian, the dialect it was trained on) and a zero-shot transfer task (Barendri, a dialect it has never seen).

### In-Domain Performance: Chittagonian-to-Bangla

This table shows the performance of models trained and tested on the Chittagonian dataset. The memory-enhanced models show a substantial improvement over the baseline.

| Model                                | SacreBLEU (↑)    | METEOR (↑)     | ROUGE-1 (↑) | ROUGE-2 (↑) | ROUGE-L (↑)    |
| ------------------------------------ | ---------------- | -------------- | ----------- | ----------- | -------------- |
| Baseline (`model_backbone`)          | 21.72            | 0.3422         | 0.7206      | 0.5741      | 0.6862         |
| **Memory-Enhanced (freq_asc_p25)**   | **25.17** (+3.45) | **0.4191**     | **0.7630**  | **0.6356**  | **0.7307**     |
| Memory-Enhanced (freq_asc_p50)       | 24.50            | 0.4176         | 0.7643      | 0.6375      | 0.7324         |
| Memory-Enhanced (attn_desc_p25)      | 24.43            | 0.4009         | 0.7437      | 0.6197      | 0.7128         |
| Memory-Enhanced (attn_desc_p75)      | 24.27            | 0.4168         | 0.7628      | 0.6357      | 0.7298         |
| Memory-Enhanced (freq_asc_p75)       | 24.17            | 0.4030         | 0.7517      | 0.6244      | 0.7202         |
| Memory-Enhanced (attn_desc_p50)      | 24.14            | 0.4011         | 0.7457      | 0.6224      | 0.7144         |

### Zero-Shot Transfer Performance: Barendri-to-Bangla

To test generalization, we took the models trained *only* on Chittagonian and evaluated them directly on the Barendri test set without any further training. This is a challenging **zero-shot cross-dialectal transfer** task.

| Model (Trained on Chittagonian)      | SacreBLEU (↑)    | METEOR (↑)     | ROUGE-1 (↑) | ROUGE-2 (↑) | ROUGE-L (↑)    |
| ------------------------------------ | ---------------- | -------------- | ----------- | ----------- | -------------- |
| Baseline (`model_backbone`)          | 16.61            | 0.3500         | 0.8107      | 0.6512      | **0.7730**     |
| **Memory-Enhanced (freq_asc_p25)**   | **17.50** (+0.89) | **0.3542**     | 0.8101      | **0.6561**  | 0.7653         |
| Memory-Enhanced (attn_desc_p75)      | 17.42            | 0.3523         | 0.8103      | 0.6549      | 0.7652         |
| Memory-Enhanced (freq_asc_p50)       | 17.14            | **0.3542**     | **0.8132**  | **0.6581**  | 0.7688         |
| Memory-Enhanced (attn_desc_p50)      | 17.03            | 0.3510         | 0.8090      | 0.6526      | 0.7637         |
| Memory-Enhanced (freq_asc_p75)       | 16.91            | 0.3511         | 0.8084      | 0.6540      | 0.7643         |
| Memory-Enhanced (attn_desc_p25)      | 16.72            | 0.3448         | 0.8075      | 0.6482      | 0.7613         |

### Overall Key Findings

1.  **Massive In-Domain Improvement**: The memory module provides a substantial **+3.45 BLEU** point increase on the Chittagonian dialect, proving its effectiveness for the primary task.

2.  **Successful Zero-Shot Transfer**: The memory-enhanced model **outperforms the baseline's BLEU score on an unseen dialect (Barendri)**, demonstrating that it has learned a generalizable strategy for handling dialectal variance, not just memorized Chittagonian-specific words.

3.  **Handling Rare Words is Key**: The best-performing model (`freq_asc_p25`) in *both* scenarios is the one whose memory is built on the **25% least frequent tokens**. This is a critical insight: the model's performance boost comes from an improved ability to translate rare, out-of-vocabulary, or dialect-specific terms.

4.  **Robust Architecture**: Almost all ablations of the memory-enhanced model outperform the baseline, showing the method is robust and consistently beneficial.

### Result Plots
Result plots for both the Chittagonian and Barendri datasets are included in the repository.

## How to Run the Pipeline

Follow these steps to replicate the experiments from data preprocessing to final evaluation.

### 1. Prerequisites
*   Python 3.8+
*   PyTorch 1.10+
*   An NVIDIA GPU with CUDA support is highly recommended for training.

### 2. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
