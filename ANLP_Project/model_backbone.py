#model_backbone.py
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, 
    TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
)
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from evaluate import load
import sacrebleu
import os


# ==================== FIXED HYPERPARAMETERS ====================
MODEL_NAME = "google/mt5-small"

# --- Data & Length Configuration ---
MAX_LENGTH = 128
TRAIN_SAMPLES = None  # Use None for all data
VAL_SAMPLES = None
TEST_SAMPLES = None

# --- Training Schedule ---
NUM_EPOCHS = 30
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

# --- Stable Learning Rate & Optimization ---
LEARNING_RATE = 3e-5
FP16_ENABLED = False
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
LABEL_SMOOTHING = 0.1

# --- Logging & Checkpointing ---
LOGGING_STEPS = 50
EVAL_STEPS = 300
SAVE_STEPS = 300
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 5

# --- Generation ---
GENERATION_MAX_LENGTH = 64
NUM_BEAMS = 5

# ==================== LOAD MODEL & DATA ====================
print("Loading mt5-small...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Loading datasets...")
with open('pre_training/data_train.json', 'r', encoding='utf-8') as f: 
    train_data = json.load(f)
with open('pre_training/data_val.json', 'r', encoding='utf-8') as f: 
    val_data = json.load(f)
with open('pre_training/data_test.json', 'r', encoding='utf-8') as f: 
    test_data = json.load(f)

def extract_data(data):
    inputs, targets = [], []
    for item in data:
        if "Dialect" in item and "Standard" in item:
            dialect, standard = item["Dialect"].strip(), item["Standard"].strip()
            if dialect and standard and dialect != standard:
                inputs.append(dialect)
                targets.append(standard)
    return inputs, targets

train_inputs, train_targets = extract_data(train_data if TRAIN_SAMPLES is None else train_data[:TRAIN_SAMPLES])
val_inputs, val_targets = extract_data(val_data if VAL_SAMPLES is None else val_data[:VAL_SAMPLES])
test_inputs, test_targets = extract_data(test_data if TEST_SAMPLES is None else test_data[:TEST_SAMPLES])

print(f"\nDataset Statistics:")
print(f"   Training:   {len(train_inputs):>6,} pairs")
print(f"   Validation: {len(val_inputs):>6,} pairs")
print(f"   Test:       {len(test_inputs):>6,} pairs\n")

if len(train_inputs) == 0:
    raise ValueError("ERROR: No training samples! Check your data or TRAIN_SAMPLES setting.")

# ==================== DATASET CLASS ====================
class TranslationDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=MAX_LENGTH):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        model_inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True)
        labels = self.tokenizer(text_target=target_text, max_length=self.max_length, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

print("Creating datasets...")
train_dataset = TranslationDataset(train_inputs, train_targets, tokenizer)
val_dataset = TranslationDataset(val_inputs, val_targets, tokenizer)
test_dataset = TranslationDataset(test_inputs, test_targets, tokenizer)
print("Datasets created successfully.\n")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding='longest'
)

# ==================== TRAINING CONFIGURATION ====================
training_args = TrainingArguments(
    output_dir="./results",
    gradient_accumulation_steps=2,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=0.06,
    weight_decay=WEIGHT_DECAY,
    label_smoothing_factor=LABEL_SMOOTHING,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    fp16=FP16_ENABLED,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)


# ==================== TRAINING ====================
# Define the path where the final model will be saved
SAVED_MODEL_PATH = "./Barendri-translation-model"

# Check if the model has already been trained and saved
if os.path.exists(SAVED_MODEL_PATH) and os.path.exists(os.path.join(SAVED_MODEL_PATH, 'config.json')):
    print(f"\nFound existing trained model at '{SAVED_MODEL_PATH}'.")
    print("Skipping training and loading model from disk.")
    
    # Load the tokenizer and model from the saved path
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_PATH)
    
else:
    # If the model doesn't exist, run the full training process
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Device:              {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Training samples:    {len(train_inputs):,}")
    print(f"Batch size:          {TRAIN_BATCH_SIZE}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"FP16:                {FP16_ENABLED}")
    print(f"{'='*70}\n")

    # This will train the model and save the best checkpoint in './results'
    trainer.train()

    print("\nTraining complete!")
    
    # The trainer automatically loads the best model at the end, so we save that one
    model.save_pretrained(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    print(f"Model saved to {SAVED_MODEL_PATH}\n")

# ==================== EVALUATION FUNCTION ====================
def evaluate_on_test_set(model, tokenizer, test_dataset):
    print(f"\n{'='*70}\nEVALUATING ON TEST SET\n{'='*70}\n")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False,
        collate_fn=data_collator
    )
    
    predictions, references, inputs_text = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=GENERATION_MAX_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
            for i in range(len(outputs)):
                input_text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                ref_labels = batch["labels"][i].clone()
                ref_labels[ref_labels == -100] = tokenizer.pad_token_id
                ref_text = tokenizer.decode(ref_labels, skip_special_tokens=True)
                
                inputs_text.append(input_text)
                predictions.append(pred_text)
                references.append(ref_text)
    
    return inputs_text, predictions, references

print("\nRunning test set evaluation...")
inputs_text, predictions, references = evaluate_on_test_set(model, tokenizer, test_dataset)

# ==================== DEBUG & COMPUTE METRICS ====================
print(f"\n{'='*70}\nDEBUGGING SAMPLES\n{'='*70}\n")
for i in range(min(5, len(predictions))):
    print(f"Sample {i+1}:")
    print(f"  Input:      {inputs_text[i]}")
    print(f"  Predicted:  {predictions[i]}")
    print(f"  Reference:  {references[i]}")
    print()

print(f"\n{'='*70}\nFINAL METRICS\n{'='*70}\n")

# Check for empty predictions
empty_preds = sum(1 for p in predictions if not p.strip())
empty_refs = sum(1 for r in references if not r.strip())
print(f"Empty predictions: {empty_preds}/{len(predictions)}")
print(f"Empty references:  {empty_refs}/{len(references)}\n")

# Compute SacreBLEU (most reliable for translation)
bleu_score = sacrebleu.corpus_bleu(predictions, [references])
print(f"SacreBLEU: {bleu_score.score:.2f}")

# Compute METEOR
meteor = load("meteor").compute(predictions=predictions, references=references)
print(f"METEOR:    {meteor['meteor']:.4f}")

# Compute ROUGE with multiple tokenization strategies
print("\nROUGE Scores (with different tokenizers):")

# Strategy 1: Default (word-level, may not work well for Bangla)
try:
    rouge = load("rouge")
    rouge_default = rouge.compute(predictions=predictions, references=references)
    print(f"   Default tokenizer:")
    print(f"      ROUGE-1: {rouge_default['rouge1']:.4f}")
    print(f"      ROUGE-2: {rouge_default['rouge2']:.4f}")
    print(f"      ROUGE-L: {rouge_default['rougeL']:.4f}")
except Exception as e:
    print(f"   Default tokenizer: Failed ({e})")

# Strategy 2: Character-level (better for Bangla)
try:
    rouge = load("rouge")
    rouge_char = rouge.compute(
        predictions=predictions, 
        references=references,
        tokenizer=lambda x: list(x)  # Character-level
    )
    print(f"   Character-level tokenizer:")
    print(f"      ROUGE-1: {rouge_char['rouge1']:.4f}")
    print(f"      ROUGE-2: {rouge_char['rouge2']:.4f}")
    print(f"      ROUGE-L: {rouge_char['rougeL']:.4f}")
    rouge_l_final = rouge_char['rougeL']
except Exception as e:
    print(f"   Character-level: Failed ({e})")
    rouge_l_final = 0.0

# Strategy 3: Space-split (simple word tokenizer)
try:
    rouge = load("rouge")
    rouge_space = rouge.compute(
        predictions=predictions, 
        references=references,
        tokenizer=lambda x: x.split()
    )
    print(f"   Space-split tokenizer:")
    print(f"      ROUGE-1: {rouge_space['rouge1']:.4f}")
    print(f"      ROUGE-2: {rouge_space['rouge2']:.4f}")
    print(f"      ROUGE-L: {rouge_space['rougeL']:.4f}")
except Exception as e:
    print(f"   Space-split: Failed ({e})")

print(f"\n{'='*70}")
print("\nRecommendation: Use character-level ROUGE for Bangla evaluation")
print(f"\n{'='*70}\n")

# Save results with all metrics
eval_results = {
    "metrics": {
        "sacrebleu": bleu_score.score, 
        "meteor": meteor['meteor'], 
        "rouge_l_char": rouge_l_final,
    }, 
    "translations": [
        {"input": inp, "prediction": pred, "reference": ref} 
        for inp, pred, ref in zip(inputs_text, predictions, references)
    ]
}

with open("./evaluation_results.json", "w", encoding='utf-8') as f: 
    json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
print("Results saved to evaluation_results.json")
print("\n✨ Evaluation complete!")