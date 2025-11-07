# memory_transformer.py — OPTIMIZED FOR BEST RESULTS
import os
import json
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import sacrebleu
from evaluate import load

# Import from existing files
from memory_module import MemoryFFNN, load_token_features, filter_valid_entries

# =====================
# OPTIMIZED HYPERPARAMETERS - FOR BEST RESULTS
# =====================
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5  # Increased from 3e-5 for better adaptation
NUM_EPOCHS = 30  # Increased from 15 to match baseline
VALIDATION_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
GENERATION_MAX_LENGTH = 64
NUM_BEAMS = 5  # CRITICAL: Match baseline!
EARLY_STOPPING_PATIENCE = 5
PRINT_INTERVAL = 100
SAMPLE_TRANSLATIONS_TO_SHOW = 5

# Use ALL data (no limits)
TRAIN_LIMIT = None
VAL_LIMIT = None
TEST_LIMIT = None

# MODEL PATHS
MODEL_PATH = "chittagong-translation-model"  # Pre-trained baseline
DATA_DIR = "post_training"  # For integrated training
OUTPUT_DIR = "memory_transformer_models"

# =====================
# CRITICAL TRAINING CONFIGURATION
# =====================
FREEZE_BASE = False  # MUST be False to improve beyond baseline
MEM_GATE_INIT = 0.1  # Start with LOW memory influence, let it learn
ACCUM_STEPS = 2
WARMUP_RATIO = 0.1  # Increased warmup
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1  # Match baseline


def _torch_device_str():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================
# Memory module wrapper
# =====================
class MemoryModule(nn.Module):
    """Memory module for storing and retrieving token embeddings."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_ffnn = MemoryFFNN(input_dim, hidden_dim, output_dim)

    def load_trained_weights(self, model_path: str):
        state_dict = torch.load(model_path, map_location='cpu')
        self.memory_ffnn.load_state_dict(state_dict)
        print(f"Loaded memory weights from {model_path}")

    def forward(self, query_embedding: torch.Tensor, memory_embeddings_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embedding: [batch, D_in]
            memory_embeddings_norm: [M, D_in] (already L2-normalized)
        Returns:
            memory_output: [batch, D_out]
            similarity_scores: [batch, M]
        """
        query_norm = F.normalize(query_embedding, p=2, dim=-1)
        similarity_scores = torch.matmul(query_norm, memory_embeddings_norm.T)
        similarity_scores = F.softmax(similarity_scores, dim=-1)
        memory_output = self.memory_ffnn(query_embedding)
        return memory_output, similarity_scores


# ======================================
# Tensor-backed Dataset + fast collate
# ======================================
class PostTrainingDataset(Dataset):
    """Pre-tokenized, tensor-backed dataset."""
    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH, limit: Optional[int] = None):
        print(f"Loading and pre-tokenizing data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if limit:
            data = data[:limit]

        src_ids, src_mask, tgt_ids = [], [], []
        for item in data:
            source = item.get('Dialect', '')
            target = item.get('Standard', '')

            s = tokenizer(
                source,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            t = tokenizer(
                target,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            src_ids.append(s['input_ids'].squeeze(0))
            src_mask.append(s['attention_mask'].squeeze(0))
            tgt_ids.append(t['input_ids'].squeeze(0))

        self.src_ids = torch.stack(src_ids).long().contiguous()
        self.src_mask = torch.stack(src_mask).long().contiguous()
        self.tgt_ids = torch.stack(tgt_ids).long().contiguous()
        print(f"Pre-tokenizing complete. Loaded {len(self)} samples")

    def __len__(self):
        return self.src_ids.size(0)

    def __getitem__(self, idx):
        return (self.src_ids[idx], self.src_mask[idx], self.tgt_ids[idx])


def fast_collate(batch):
    x, m, y = zip(*batch)
    return {
        'source_input_ids': torch.stack(x, 0),
        'source_attention_mask': torch.stack(m, 0),
        'target_input_ids': torch.stack(y, 0),
    }


# =============================
# Memory-Enhanced Transformer
# =============================
class MemoryTransformer(nn.Module):
    """Integrated Memory Transformer for translation."""
    def __init__(self, model_path: str, memory_config_path: str, token_features_path: str, memory_model_path: str):
        super().__init__()

        print(f"Loading base model from {model_path}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        print(f"Loading memory configuration from {memory_config_path}...")
        with open(memory_config_path, 'r') as f:
            self.memory_config = json.load(f)

        print(f"Loading token features from {token_features_path}...")
        self.token_features = load_token_features(token_features_path)

        self.memory_module = MemoryModule(
            input_dim=self.memory_config['model_architecture']['input_dim'],
            hidden_dim=self.memory_config['model_architecture']['hidden_dim'],
            output_dim=self.memory_config['model_architecture']['output_dim']
        )
        self.memory_module.load_trained_weights(memory_model_path)

        # Prepare memory embeddings
        self._prepare_memory_embeddings()

        # Projection layers
        self.hidden_size = getattr(self.model.config, 'd_model', 512)
        inp_dim = self.memory_config['model_architecture']['input_dim']
        out_dim = self.memory_config['model_architecture']['output_dim']
        
        self.to_mem = nn.Linear(self.hidden_size, inp_dim, bias=False)
        self.from_mem = nn.Linear(out_dim, self.hidden_size, bias=False)
        
        # CRITICAL: Learnable gate initialized LOW, constrained to [0, 1]
        self.memory_gate = nn.Parameter(torch.tensor(MEM_GATE_INIT))

        # Keep base model trainable
        if FREEZE_BASE:
            for p in self.model.parameters():
                p.requires_grad = False
            print("[WARNING] Base model frozen - this will hurt performance!")
        else:
            print("[Init] Base model trainable ✓")

    def _prepare_memory_embeddings(self):
        """Prepare memory embeddings from token features."""
        memory_embeddings = []
        self.token_to_idx = {}
        valid_entries = filter_valid_entries(self.token_features)
        target_dim = self.memory_config['model_architecture']['input_dim']

        for token_data in valid_entries:
            if 'input_embedding' in token_data and token_data['input_embedding']:
                emb = torch.tensor(token_data['input_embedding'], dtype=torch.float32)
                if emb.numel() < target_dim:
                    emb = torch.cat([emb, torch.zeros(target_dim - emb.numel())])
                elif emb.numel() > target_dim:
                    emb = emb[:target_dim]
                memory_embeddings.append(emb)
                self.token_to_idx[token_data['clean_token']] = len(memory_embeddings) - 1

        if memory_embeddings:
            mem = torch.stack(memory_embeddings)
        else:
            mem = torch.randn(100, target_dim)

        # Register as buffers
        with torch.no_grad():
            self.register_buffer("memory_embeddings", mem, persistent=False)
            self.register_buffer("memory_embeddings_norm", F.normalize(mem, p=2, dim=-1), persistent=False)
        print(f"Loaded {self.memory_embeddings.shape[0]} memory embeddings")

    def encode_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return encoder_outputs.last_hidden_state

    def decode_with_memory(self, encoder_hidden_states: torch.Tensor,
                           target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Decode with GENTLE memory integration."""
        # Standard decoder forward
        decoder_outputs = self.model.decoder(
            input_ids=target_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        hs = decoder_outputs.last_hidden_state  # [B, S, H]

        # Project to memory space
        q = self.to_mem(hs)  # [B, S, D_in]
        
        # Compute similarities (vectorized)
        qn = F.normalize(q, p=2, dim=-1)
        sims = torch.einsum('bsd,md->bsm', qn, self.memory_embeddings_norm)  # [B, S, M]
        
        # Use top-k for efficiency
        topk = 256
        if topk > 0 and topk < sims.size(-1):
            topv, _ = sims.topk(topk, dim=-1)
            sims_soft = F.softmax(topv, dim=-1)
        else:
            sims_soft = F.softmax(sims, dim=-1)
        
        # Memory FFNN
        B, S, D_in = q.shape
        mem_out = self.memory_module.memory_ffnn(q.reshape(B * S, D_in))
        mem_out = mem_out.view(B, S, -1)
        mem_proj = self.from_mem(mem_out)  # [B, S, H]

        # LEARNABLE gating with sigmoid constraint
        gate = torch.sigmoid(self.memory_gate)
        combined = gate * mem_proj + (1.0 - gate) * hs

        return combined

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        encoder_hidden_states = self.encode_with_memory(input_ids, attention_mask)
        
        if target_ids is not None:
            decoder_input_ids = target_ids[:, :-1]
            memory_enhanced_decoder_states = self.decode_with_memory(
                encoder_hidden_states, decoder_input_ids, attention_mask
            )
            logits = self.model.lm_head(memory_enhanced_decoder_states)
            return {
                'logits': logits,
                'encoder_hidden_states': encoder_hidden_states
            }
        else:
            return {
                'encoder_hidden_states': encoder_hidden_states
            }


# =============================
# Datasets / Dataloaders
# =============================
def load_datasets(data_dir: str, tokenizer, train_limit=None, test_limit=None, val_limit=None):
    """Load datasets."""
    train_dataset = PostTrainingDataset(f"{data_dir}/data_train.json", tokenizer, limit=train_limit)
    val_dataset = PostTrainingDataset(f"{data_dir}/data_val.json", tokenizer, limit=val_limit)
    test_dataset = PostTrainingDataset(f"{data_dir}/data_test.json", tokenizer, limit=test_limit)

    # Windows-safe settings
    num_workers = 0 if os.name == 'nt' else 2
    pin = torch.cuda.is_available()

    common = dict(
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=fast_collate,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **common)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# =============================
# Training
# =============================
def train_memory_transformer(model: MemoryTransformer, train_loader: DataLoader,
                             val_loader: DataLoader, ablation_name: str, num_epochs: int = NUM_EPOCHS):
    device = torch.device(_torch_device_str())
    model.to(device)
    print(f"Using device: {device}")
    
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Warmup + Cosine scheduler
    total_steps = len(train_loader) * num_epochs // ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id, label_smoothing=LABEL_SMOOTHING)

    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            source_ids = batch['source_input_ids'].to(device, non_blocking=True)
            source_mask = batch['source_attention_mask'].to(device, non_blocking=True)
            target_ids = batch['target_input_ids'].to(device, non_blocking=True)

            outputs = model(source_ids, source_mask, target_ids)
            logits = outputs['logits']
            labels = target_ids[:, 1:].contiguous()
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / ACCUM_STEPS
            
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            batch_count += 1

            if batch_idx % PRINT_INTERVAL == 0:
                gate_val = torch.sigmoid(model.memory_gate).item()
                print(f'  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item()*ACCUM_STEPS:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                      f'MemGate: {gate_val:.3f}')

        avg_loss = total_loss / max(1, batch_count)

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Avg Training Loss: {avg_loss:.4f}")
        print(f"    Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"    Memory Gate: {torch.sigmoid(model.memory_gate).item():.3f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                source_ids = batch['source_input_ids'].to(device, non_blocking=True)
                source_mask = batch['source_attention_mask'].to(device, non_blocking=True)
                target_ids = batch['target_input_ids'].to(device, non_blocking=True)

                outputs = model(source_ids, source_mask, target_ids)
                logits = outputs['logits']
                labels = target_ids[:, 1:].contiguous()
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / max(1, val_batch_count)
        print(f'    Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_path = os.path.join(OUTPUT_DIR, f"best_{ablation_name}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'    New best model saved (val_loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'    No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'\n  Early stopping triggered after epoch {epoch+1}')
            break

        print()

    # Load best model
    best_model_path = os.path.join(OUTPUT_DIR, f"best_{ablation_name}.pt")
    model.load_state_dict(torch.load(best_model_path))
    print(f"\nLoaded best model from {best_model_path}")
    
    return best_val_loss


# =============================
# Evaluation
# =============================
# =============================
# Evaluation (MATCHES BASELINE)
# =============================
def evaluate_model(model: MemoryTransformer, test_loader: DataLoader, ablation_name: str):
    """
    Comprehensive evaluation on the test set using a beam search strategy
    identical to the model_backbone.py script for a fair comparison.
    """
    model.eval()
    device = torch.device(_torch_device_str())
    model.to(device)

    print("\n" + "="*70)
    print(f"EVALUATING MODEL: {ablation_name}")
    print("="*70)

    all_predictions = []
    all_references = []
    all_inputs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(test_loader)}...")

            source_ids = batch['source_input_ids'].to(device, non_blocking=True)
            source_mask = batch['source_attention_mask'].to(device, non_blocking=True)
            # We don't need target_ids for generation, but we'll use it for decoding the reference text later
            target_ids = batch['target_input_ids'] 

            # =================================================================
            # CRITICAL: This .generate() call is now identical to model_backbone.py
            # =================================================================
            # --- THIS IS THE CORRECTED LINE ---
            generated_ids = model.model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=GENERATION_MAX_LENGTH,
                num_beams=NUM_BEAMS,
                # Adding these back in from the baseline for the best quality
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            # =================================================================

            for i in range(source_ids.shape[0]):
                source_text = model.tokenizer.decode(source_ids[i], skip_special_tokens=True)
                generated_text = model.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                # Decode the reference text from the original target_ids
                reference_text = model.tokenizer.decode(target_ids[i], skip_special_tokens=True)

                all_inputs.append(source_text)
                all_predictions.append(generated_text)
                all_references.append(reference_text)

    # --- The rest of the function (metric calculation and saving) is already correct ---

    # Calculate metrics
    print("\nCalculating metrics...")

    # SacreBLEU
    bleu_score = sacrebleu.corpus_bleu(all_predictions, [all_references])
    print(f"SacreBLEU: {bleu_score.score:.2f}")

    # METEOR
    meteor = load("meteor")
    meteor_score = meteor.compute(predictions=all_predictions, references=all_references)
    print(f"METEOR: {meteor_score['meteor']:.4f}")

    # ROUGE (character-level for Bangla)
    rouge = load("rouge")
    rouge_score_char = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        tokenizer=lambda x: list(x)
    )
    print(f"\nROUGE Scores (character-level):")
    print(f"   ROUGE-1: {rouge_score_char['rouge1']:.4f}")
    print(f"   ROUGE-2: {rouge_score_char['rouge2']:.4f}")
    print(f"   ROUGE-L: {rouge_score_char['rougeL']:.4f}")

    metrics = {
        "sacrebleu": float(bleu_score.score),
        "meteor": float(meteor_score['meteor']),
        "rouge_1_char": float(rouge_score_char['rouge1']),
        "rouge_2_char": float(rouge_score_char['rouge2']),
        "rouge_l_char": float(rouge_score_char['rougeL']),
        "num_test_samples": len(all_predictions)
    }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"results_{ablation_name}.json")
    eval_results = {
        "ablation_name": ablation_name,
        "metrics": metrics,
        "sample_translations": [
            {"input": inp, "prediction": pred, "reference": ref}
            for inp, pred, ref in zip(all_inputs[:20], all_predictions[:20], all_references[:20])
        ]
    }

    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")

    # Print sample translations
    print(f"\n{'='*70}")
    print("SAMPLE TRANSLATIONS")
    print("="*70)
    for i in range(min(SAMPLE_TRANSLATIONS_TO_SHOW, len(all_inputs))):
        print(f"\nSample {i+1}:")
        print(f"  Dialect:   {all_inputs[i]}")
        print(f"  Reference: {all_references[i]}")
        print(f"  Generated: {all_predictions[i]}")

    return metrics
# =============================
# Main
# =============================
def run_ablation(ablation_name: str):
    """Run complete pipeline for one ablation."""
    print("\n" + "="*70)
    print(f"RUNNING ABLATION: {ablation_name}")
    print("="*70 + "\n")

    memory_model_path = f"memory_models/{ablation_name}.pt"
    memory_config_path = f"memory_models/{ablation_name}_info.json"
    token_features_path = "extracted_token_features.json"

    if not os.path.exists(memory_model_path):
        print(f"Memory model not found: {memory_model_path}")
        return None

    memory_transformer = MemoryTransformer(
        model_path=MODEL_PATH,
        memory_config_path=memory_config_path,
        token_features_path=token_features_path,
        memory_model_path=memory_model_path
    )

    train_loader, val_loader, test_loader = load_datasets(
        DATA_DIR, memory_transformer.tokenizer, TRAIN_LIMIT, TEST_LIMIT, VAL_LIMIT
    )

    # =================================================================
    # START OF THE NEW SNIPPET
    # =================================================================
    best_model_path = os.path.join(OUTPUT_DIR, f"best_{ablation_name}.pt")

    if os.path.exists(best_model_path):
        print(f"\nFound existing trained model at '{best_model_path}'.")
        print("Skipping training and loading weights directly for evaluation.")
        
        # Load the pre-trained weights directly into the model
        memory_transformer.load_state_dict(torch.load(best_model_path))
        
        # Set a placeholder value since we didn't re-train
        val_loss = -1.0 
    else:
        # If no checkpoint exists, run the training process
        print("\nStarting training...")
        val_loss = train_memory_transformer(memory_transformer, train_loader, val_loader, ablation_name, NUM_EPOCHS)
    # =================================================================
    # END OF THE NEW SNIPPET
    # =================================================================

    print("\nStarting evaluation...")
    metrics = evaluate_model(memory_transformer, test_loader, ablation_name)

    return {
        "ablation": ablation_name,
        "val_loss": val_loss,
        **metrics
    }


def main():
    """Run all ablations."""
    print("\n" + "="*70)
    print("MEMORY-ENHANCED TRANSFORMER - OPTIMIZED FOR BEST RESULTS")
    print("="*70 + "\n")

    ablation_dir = "memory_models"
    if not os.path.exists(ablation_dir):
        print(f"Memory models directory not found: {ablation_dir}")
        return

    ablation_files = [f for f in os.listdir(ablation_dir) if f.endswith('.pt')]
    ablations = [f.replace('.pt', '') for f in ablation_files]

    if not ablations:
        print("No ablation models found!")
        return

    print(f"Found {len(ablations)} ablations:")
    for i, abl in enumerate(ablations, 1):
        print(f"  {i}. {abl}")

    all_results = []
    for ablation in ablations:
        try:
            result = run_ablation(ablation)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nError running ablation {ablation}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if all_results:
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY - ALL ABLATIONS")
        print("="*70 + "\n")

        all_results_sorted = sorted(all_results, key=lambda x: x.get('sacrebleu', 0), reverse=True)

        print(f"{'Rank':<6} {'Ablation':<25} {'BLEU':<8} {'METEOR':<8} {'ROUGE-L':<8}")
        print("-"*70)
        for rank, result in enumerate(all_results_sorted, 1):
            print(f"{rank:<6} {result['ablation']:<25} {result['sacrebleu']:<8.2f} "
                  f"{result['meteor']:<8.4f} {result['rouge_l_char']:<8.4f}")

        print("\n" + "="*70)
        best = all_results_sorted[0]
        print(f"BEST ABLATION: {best['ablation']}")
        print(f"   BLEU: {best['sacrebleu']:.2f}")
        print(f"   METEOR: {best['meteor']:.4f}")
        print(f"   ROUGE-L: {best['rouge_l_char']:.4f}")
        print("="*70)

        # Save final summary
        summary_path = os.path.join(OUTPUT_DIR, "final_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "all_results": all_results_sorted,
                "best_ablation": best
            }, f, indent=2)
        print(f"\nFinal summary saved to: {summary_path}")

    print("\nAll ablations completed!")


if __name__ == "__main__":
    main()