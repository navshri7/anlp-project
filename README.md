# Adaptive Memory-based Transformer for Low-Resource Dialect Translation

This repository contains the code and results for a project on translating low-resource dialects such as Chittagonian and Barendri into Standard Bangla. The core of this work is a novel **Memory-Enhanced Transformer model** that improves upon a standard fine-tuned baseline by integrating a specialized memory module. This module is trained to focus on specific, challenging tokens, leading to significant gains in translation quality and impressive generalization to unseen dialects.

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
```
Your `requirements.txt` file should contain:
```
transformers
torch
numpy
evaluate
sacrebleu
scikit-learn
sentencepiece
```

### 3. Data Preparation
This phase cleans the raw data and partitions it for the two main training stages. For each dataset (e.g., Chittagonian, Barendri), you will need to run these scripts.

1.  **`preprocess.py`**: Place your raw dataset files in a designated folder. Run this script to perform initial cleaning and normalization.
    ```bash
    python preprocess.py --data_dir path/to/raw/data
    ```
2.  **`merge.py`**: This script merges multiple cleaned data files into a single corpus.
    ```bash
    python merge.py --input_dir path/to/cleaned/data --output_file merged_data.json
    ```
3.  **`partition.py`**: This script splits the merged corpus into two sets:
    *   `pre_training/`: Data used to train the baseline `model_backbone`.
    *   `post_training/`: Data used to fine-tune the `memory_transformer`.
    ```bash
    python partition.py --input_file merged_data.json
    ```
After these steps, your directory should contain `pre_training` and `post_training` folders, each with `data_train.json`, `data_val.json`, and `data_test.json`.

### 4. Training and Evaluation
This is the core pipeline for building and evaluating the models.

1.  **Train the Baseline Model**: This script fine-tunes the `google/mt5-small` model on the `pre_training` dataset. The best model will be saved to `./chittagong-translation-model`.
    ```bash
    python model_backbone.py
    ```

2.  **Extract Token Features**: Using the trained baseline model, this script analyzes the training data to extract detailed features (embeddings, attention scores, frequency) for every token. The output is a large `extracted_token_features.json` file.
    ```bash
    python embeddings_extract.py
    ```

3.  **Train Memory Modules**: This script runs all the ablation studies. It reads `extracted_token_features.json`, filters tokens based on each ablation's criteria, and trains a dedicated FFNN memory module for each. The trained modules are saved in the `memory_models/` directory.
    ```bash
    python memory_module.py
    ```

4.  **Train and Evaluate the Memory-Enhanced Transformer**: This is the final step. The script iterates through each trained memory module, integrates it with the baseline model, fine-tunes the combined architecture, and evaluates the final performance.
    ```bash
    python memory_transformer.py
    ```

## Pre-trained Models

The best-performing baseline model (`model_backbone`) and the best Memory-Enhanced Transformer (`freq_asc_p25`) are available for download.

**[Download Models Here]**(https://iiithydresearch-my.sharepoint.com/:f:/g/personal/vamshavardhanreddy_b_research_iiit_ac_in/EkDRGUqD1EJBlGOA0kTCnmIBM2UcpSAQeOsVdixvhwIYyQ?e=px2iw5)

**Instructions:**
*   Download and unzip the files.
*   Place the baseline model folder (e.g., `chittagong-translation-model`) in the root directory of the project.
*   Place the memory transformer model files (e.g., `best_freq_asc_p25.pt`) inside the `memory_transformer_models/` directory.

## Codebase Description
- **Data Processing**: `preprocess.py`, `merge.py`, `partition.py`
  - Scripts for cleaning, consolidating, and splitting the raw text data for the two-stage training process.
- **`model_backbone.py`**: Handles the fine-tuning of the standard `google/mt5-small` model to establish the baseline performance.
- **`embeddings_extract.py`**: A feature extraction pipeline that computes embeddings, cross-attention norms, and frequencies for tokens in the training corpus.
- **`memory_module.py`**: Defines the FFNN memory architecture and trains multiple modules based on the ablation configurations.
- **`memory_transformer.py`**: The core script that defines the integrated memory-enhanced architecture, connects the baseline and memory with a learnable gate, and handles the final stage of training and evaluation.

## Ablation Studies

To understand what knowledge is most useful for the memory module, we conducted several ablation studies based on two key token properties:

1.  **Frequency**: Does the model benefit more from memorizing common words or rare words?
    *   `freq_asc_p25/p50/p75`: Memory trained on the 25%/50%/75% **least** frequent tokens.
2.  **Cross-Attention**: Does the model benefit from memorizing tokens that it already "focuses on" during translation?
    *   `attn_desc_p25/p50/p75`: Memory trained on the 25%/50%/75% of tokens with the **highest** cross-attention scores.

Our results clearly indicate that focusing on **low-frequency tokens (`freq_asc`)** provides the most significant and generalizable performance boost.

## Datasets

This work utilizes two newly created datasets:
1.  **Chittagonian-to-Bangla**: A corpus of parallel sentences from the Chittagonian dialect and Standard Bangla which is present in ANLP_Project file.
2.  **Barendri-to-Bangla**: A corpus of parallel sentences from the Barendri dialect and Standard Bangla.

