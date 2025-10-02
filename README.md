# ANLP-Project
Dataset 5050 sentences each in Standard and Barendri Bangla - Parallel Corpora


Completed Tasks:
1. Dataset - 5k+ parallel corpora created and cleaned
2. Implementation - Working code for Chittagong Dialect Completed

How to run:
1. for every set in the data, first run ```preprocess.py```
2. run ```merge.py```
3. run ```partition.py```
4. run ```model_backbone.py```
5. run ```embeddings_extract.py```
6. run ```memory_module.py```
7. run ```memory_transformer.py```

Points to note:
1. Keep all limit/max_samples related values as ```None``` for full dataset training
2. Check paths before proceeding

Upcoming tasks:
1. Hyperparameter Finetuning for baselines on Chittagong Dialect
2. Ablations - Memory Threshold (Compute VS Metrics tradeoff), Memory Architecture, Memory Trigger Key, Other Hyperparameters
3. Obtain baselines on Barendri Dialect
