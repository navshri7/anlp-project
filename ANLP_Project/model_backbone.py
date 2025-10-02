from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import numpy as np
from evaluate import load
import sacrebleu

# HYPERPARAMETERS AS MACROS
MODEL_NAME = "google/mt5-small"
MAX_LENGTH = 512
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 100
TEST_SAMPLES = 100
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 500
GENERATION_MAX_LENGTH = 512
NUM_BEAMS = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load and prepare data from separate files
with open('pre_training/data_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('pre_training/data_val.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

with open('pre_training/data_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Extract inputs and targets from each dataset
def extract_data(data):
    inputs = []
    targets = []
    for item in data:
        if "Dialect" in item:
            inputs.append(item["Dialect"])
        if "Standard" in item:
            targets.append(item["Standard"])
    return inputs, targets
if TRAIN_SAMPLES != None:
    train_inputs, train_targets = extract_data(train_data[:TRAIN_SAMPLES])
else:
    train_inputs, train_targets = extract_data(train_data)
if VAL_SAMPLES != None:
    val_inputs, val_targets = extract_data(val_data[:VAL_SAMPLES])
else:
    val_inputs, val_targets = extract_data(val_data)
if TEST_SAMPLES != None:
    test_inputs, test_targets = extract_data(test_data[:TEST_SAMPLES])
else:
    test_inputs, test_targets = extract_data(test_data)

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
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten()
        }

# Create datasets
train_dataset = TranslationDataset(train_inputs, train_targets, tokenizer)
val_dataset = TranslationDataset(val_inputs, val_targets, tokenizer)
test_dataset = TranslationDataset(test_inputs, test_targets, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./chittagong-translation-model")
tokenizer.save_pretrained("./chittagong-translation-model")

# Evaluation on test set
def evaluate_on_test_set(model, tokenizer, test_dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    predictions = []
    references = []
    inputs_text = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=GENERATION_MAX_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )
            
            # Decode predictions, references, and inputs
            for i in range(len(outputs)):
                input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                ref_text = tokenizer.decode(labels[i], skip_special_tokens=True)
                
                inputs_text.append(input_text)
                predictions.append(pred_text)
                references.append(ref_text)
    
    return inputs_text, predictions, references

# Run evaluation
print("Evaluating on test set...")
inputs_text, predictions, references = evaluate_on_test_set(model, tokenizer, test_dataset)

# Calculate SacreBLEU
bleu_score = sacrebleu.corpus_bleu(predictions, [references])
print(f"SacreBLEU: {bleu_score.score:.2f}")

# Calculate METEOR
meteor = load("meteor")
meteor_score = meteor.compute(predictions=predictions, references=references)
print(f"METEOR: {meteor_score['meteor']:.4f}")

# Calculate ROUGE-L
rouge = load("rouge")
rouge_score = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")

# Save evaluation results with translations
eval_results = {
    "metrics": {
        "sacrebleu": bleu_score.score,
        "meteor": meteor_score['meteor'],
        "rouge_l": rouge_score['rougeL'],
        "num_test_samples": len(test_dataset)
    },
    "translations": [
        {
            "input": inp,
            "prediction": pred,
            "reference": ref
        }
        for inp, pred, ref in zip(inputs_text, predictions, references)
    ]
}

with open("./evaluation_results.json", "w", encoding='utf-8') as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

print(f"Evaluation complete. Results saved to evaluation_results.json")