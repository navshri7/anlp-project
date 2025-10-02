import torch
import json
import numpy as np
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

# Import from existing files
from memory_module import MemoryFFNN, load_token_features, filter_valid_entries
import sacrebleu
from evaluate import load

# HYPERPARAMETERS AS MACROS
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
VALIDATION_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
GENERATION_MAX_LENGTH = 128
NUM_BEAMS = 4
MEMORY_FREQUENCY_THRESHOLD = 50
FALLBACK_MEMORY_SIZE = 100
TRAIN_LIMIT = 100
VAL_LIMIT = 100
TEST_LIMIT = 100
PRINT_INTERVAL = 50
SAMPLE_TRANSLATIONS_TO_SHOW = 3

# MODEL PATHS
MODEL_PATH = "chittagong-translation-model"
MEMORY_CONFIG_PATH = "memory_model/memory_model_frequency_desc_top10_info.json"
TOKEN_FEATURES_PATH = "extracted_token_features.json"
MEMORY_MODEL_PATH = "memory_model/memory_model_frequency_desc_top10.pt"
DATA_DIR = "post_training"
OUTPUT_MODEL_PATH = "memory_transformer_integrated.pt"
EVALUATION_RESULTS_PATH = "./memory_transformer_evaluation_results.json"

class MemoryModule(nn.Module):
    """Memory module for storing and retrieving token embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Use the trained memory FFNN
        self.memory_ffnn = MemoryFFNN(input_dim, hidden_dim, output_dim)
        
        # Similarity computation
        self.similarity_layer = nn.Linear(input_dim, input_dim)
        
    def load_trained_weights(self, model_path: str):
        """Load trained memory model weights."""
        state_dict = torch.load(model_path, map_location='cpu')
        self.memory_ffnn.load_state_dict(state_dict)
        
    def forward(self, query_embedding: torch.Tensor, memory_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embedding: [batch_size, input_dim]
            memory_embeddings: [memory_size, input_dim]
        Returns:
            memory_output: [batch_size, output_dim]
            similarity_scores: [batch_size, memory_size]
        """
        # Compute similarity scores using cosine similarity
        query_norm = F.normalize(query_embedding, p=2, dim=-1)
        memory_norm = F.normalize(memory_embeddings, p=2, dim=-1)
        similarity_scores = torch.matmul(query_norm, memory_norm.T)  # [batch_size, memory_size]
        similarity_scores = F.softmax(similarity_scores, dim=-1)
        
        # Get memory response using trained FFNN
        memory_output = self.memory_ffnn(query_embedding)  # [batch_size, output_dim]
        
        return memory_output, similarity_scores

class PostTrainingDataset(Dataset):
    """Dataset for post-training data from JSON files."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH, limit = None):
        # Load JSON data instead of pickle
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if limit:
            self.data = self.data[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Extract dialect and standard text
        source = item.get('Dialect', '')
        target = item.get('Standard', '')
        
        source_encoded = self.tokenizer(
            source, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        target_encoded = self.tokenizer(
            target, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'source_input_ids': source_encoded['input_ids'].squeeze(),
            'source_attention_mask': source_encoded['attention_mask'].squeeze(),
            'target_input_ids': target_encoded['input_ids'].squeeze(),
            'target_attention_mask': target_encoded['attention_mask'].squeeze()
        }

class MemoryTransformer(nn.Module):
    """Integrated Memory Transformer for translation."""
    
    def __init__(self, model_path: str, memory_config_path: str, token_features_path: str, memory_model_path: str):
        super().__init__()
        
        # Load pre-trained model and tokenizer from chittagong-translation-model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # Load memory configuration
        with open(memory_config_path, 'r') as f:
            self.memory_config = json.load(f)
        
        # Load token features for memory
        self.token_features = load_token_features(token_features_path)
        
        # Initialize memory module
        self.memory_module = MemoryModule(
            input_dim=self.memory_config['model_architecture']['input_dim'],
            hidden_dim=self.memory_config['model_architecture']['hidden_dim'],
            output_dim=self.memory_config['model_architecture']['output_dim']
        )
        
        # Load trained memory weights
        self.memory_module.load_trained_weights(memory_model_path)
        
        # Prepare memory embeddings
        self._prepare_memory_embeddings()
        
        # # Freeze MT5 encoder (optional)
        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False
    
    def _prepare_memory_embeddings(self):
        """Prepare memory embeddings from token features."""
        memory_embeddings = []
        self.token_to_idx = {}
        
        # Filter tokens based on frequency threshold from config
        threshold = self.memory_config.get('threshold', MEMORY_FREQUENCY_THRESHOLD)
        valid_entries = filter_valid_entries(self.token_features)
        
        for token_data in valid_entries:
            if token_data.get('frequency', 0) >= threshold:
                if 'input_embedding' in token_data and token_data['input_embedding']:
                    embedding = torch.tensor(token_data['input_embedding'], dtype=torch.float32)
                    # Pad or truncate to match input_dim
                    target_dim = self.memory_config['model_architecture']['input_dim']
                    if len(embedding) < target_dim:
                        padding = torch.zeros(target_dim - len(embedding))
                        embedding = torch.cat([embedding, padding])
                    elif len(embedding) > target_dim:
                        embedding = embedding[:target_dim]
                    
                    memory_embeddings.append(embedding)
                    self.token_to_idx[token_data['clean_token']] = len(memory_embeddings) - 1
        
        if memory_embeddings:
            self.memory_embeddings = torch.stack(memory_embeddings)
        else:
            # Fallback: create random embeddings
            self.memory_embeddings = torch.randn(FALLBACK_MEMORY_SIZE, self.memory_config['model_architecture']['input_dim'])
        
        print(f"Loaded {len(self.memory_embeddings)} memory embeddings")
    
    def encode_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input using model encoder and get embeddings."""
        # Get encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    
    def decode_with_memory(self, encoder_hidden_states: torch.Tensor, target_ids: torch.Tensor, 
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Decode with memory integration at decoder level."""
        # Get regular decoder outputs
        decoder_outputs = self.model.decoder(
            input_ids=target_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        
        decoder_hidden_states = decoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = decoder_hidden_states.shape
        
        # Process each token position
        memory_enhanced_states = []
        
        for i in range(seq_len):
            token_embedding = decoder_hidden_states[:, i, :]  # [batch_size, hidden_size]
            
            # Adjust embedding dimension to match memory input
            if hidden_size != self.memory_config['model_architecture']['input_dim']:
                token_embedding_adjusted = F.adaptive_avg_pool1d(
                    token_embedding.unsqueeze(1), 
                    self.memory_config['model_architecture']['input_dim']
                ).squeeze(1)
            else:
                token_embedding_adjusted = token_embedding
            
            # Get memory response and similarity scores
            memory_output, similarity_scores = self.memory_module(
                token_embedding_adjusted, 
                self.memory_embeddings.to(token_embedding.device)
            )
            
            # Compute overall similarity score (max similarity as weight)
            max_similarity = torch.max(similarity_scores, dim=1)[0]  # [batch_size]
            
            # Adjust memory output dimension to match decoder hidden size
            if memory_output.shape[-1] != hidden_size:
                memory_output_adjusted = F.adaptive_avg_pool1d(
                    memory_output.unsqueeze(1), 
                    hidden_size
                ).squeeze(1)
            else:
                memory_output_adjusted = memory_output
            
            # Weighted combination: memory_weight * memory_output + (1 - memory_weight) * decoder_output
            combined_embedding = (
                max_similarity.unsqueeze(1) * memory_output_adjusted + 
                (1 - max_similarity.unsqueeze(1)) * decoder_hidden_states[:, i, :]
            )
            
            memory_enhanced_states.append(combined_embedding)
        
        # Stack back to sequence
        memory_enhanced_sequence = torch.stack(memory_enhanced_states, dim=1)
        
        return memory_enhanced_sequence
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                target_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with memory integration."""
        
        # Encode input
        encoder_hidden_states = self.encode_with_memory(input_ids, attention_mask)
        
        if target_ids is not None:
            # Training mode - decode with memory integration
            decoder_input_ids = target_ids[:, :-1]  # Shift right for decoder input
            
            # Get memory-enhanced decoder states
            memory_enhanced_decoder_states = self.decode_with_memory(
                encoder_hidden_states, decoder_input_ids, attention_mask
            )
            
            # Get logits using the language model head
            logits = self.model.lm_head(memory_enhanced_decoder_states)
            
            return {
                'logits': logits,
                'encoder_hidden_states': encoder_hidden_states
            }
        else:
            # Inference mode
            return {
                'encoder_hidden_states': encoder_hidden_states
            }
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                 max_length: int = GENERATION_MAX_LENGTH, num_beams: int = NUM_BEAMS) -> torch.Tensor:
        """Generate translation with memory enhancement."""
        
        # For generation, we'll use a custom generation loop
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode input
        encoder_hidden_states = self.encode_with_memory(input_ids, attention_mask)
        
        # Initialize decoder input with start token
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.tokenizer.pad_token_id, 
            device=device, 
            dtype=torch.long
        )
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get memory-enhanced decoder output
            memory_enhanced_states = self.decode_with_memory(
                encoder_hidden_states, decoder_input_ids, attention_mask
            )
            
            # Get logits for next token
            logits = self.model.lm_head(memory_enhanced_states[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            
            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for end token
            if torch.all(next_token == self.tokenizer.eos_token_id):
                break
        
        return decoder_input_ids

def load_datasets(data_dir: str, tokenizer, train_limit=None, test_limit=None, val_limit=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load train, validation, and test datasets from post_training_data."""
    
    # Load from JSON files in post_training directory
    if train_limit:
        train_dataset = PostTrainingDataset(f"{data_dir}/data_train.json", tokenizer, limit=train_limit)
    else:
        train_dataset = PostTrainingDataset(f"{data_dir}/data_train.json", tokenizer)
    if val_limit:
        val_dataset = PostTrainingDataset(f"{data_dir}/data_val.json", tokenizer, limit=val_limit)
    else:
        val_dataset = PostTrainingDataset(f"{data_dir}/data_val.json", tokenizer)
    if test_limit:
        test_dataset = PostTrainingDataset(f"{data_dir}/data_test.json", tokenizer, limit=test_limit)
    else:
        test_dataset = PostTrainingDataset(f"{data_dir}/data_test.json", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_memory_transformer(model: MemoryTransformer, train_loader: DataLoader, 
                           val_loader: DataLoader, num_epochs: int = NUM_EPOCHS):
    """Train the memory transformer model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to device
            source_ids = batch['source_input_ids'].to(device)
            source_mask = batch['source_attention_mask'].to(device)
            target_ids = batch['target_input_ids'].to(device)
            
            # Forward pass
            outputs = model(source_ids, source_mask, target_ids)
            logits = outputs['logits']
            
            # Prepare labels (shift left)
            labels = target_ids[:, 1:]
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % PRINT_INTERVAL == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                source_ids = batch['source_input_ids'].to(device)
                source_mask = batch['source_attention_mask'].to(device)
                target_ids = batch['target_input_ids'].to(device)
                
                outputs = model(source_ids, source_mask, target_ids)
                loss = criterion(outputs['logits'].reshape(-1, outputs['logits'].size(-1)), 
                               target_ids[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

def main():
    """Main function to initialize and train the memory transformer."""
    
    # Initialize model
    memory_transformer = MemoryTransformer(
        model_path=MODEL_PATH,
        memory_config_path=MEMORY_CONFIG_PATH,
        token_features_path=TOKEN_FEATURES_PATH,
        memory_model_path=MEMORY_MODEL_PATH
    )
    
    # Load datasets
    train_loader, val_loader, test_loader = load_datasets(DATA_DIR, memory_transformer.tokenizer, TRAIN_LIMIT, TEST_LIMIT, VAL_LIMIT)
    
    # Train model
    train_memory_transformer(memory_transformer, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Save model
    torch.save(memory_transformer.state_dict(), OUTPUT_MODEL_PATH)
    print("Model saved successfully!")
    
    # Test generation
    memory_transformer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_transformer.to(device)
    
    print("\n--- Testing Memory-Enhanced Translation ---")
    # Comprehensive evaluation on test set
    all_predictions = []
    all_references = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            source_ids = batch['source_input_ids'].to(device)
            source_mask = batch['source_attention_mask'].to(device)
            target_ids = batch['target_input_ids'].to(device)
            
            # Generate translations
            generated_ids = memory_transformer.generate(source_ids, source_mask, max_length=GENERATION_MAX_LENGTH)
            
            # Decode texts for each sample in batch
            for i in range(source_ids.shape[0]):
                source_text = memory_transformer.tokenizer.decode(source_ids[i], skip_special_tokens=True)
                generated_text = memory_transformer.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                target_text = memory_transformer.tokenizer.decode(target_ids[i], skip_special_tokens=True)
                
                all_inputs.append(source_text)
                all_predictions.append(generated_text)
                all_references.append(target_text)
    
    # Calculate SacreBLEU
    bleu_score = sacrebleu.corpus_bleu(all_predictions, [all_references])
    print(f"SacreBLEU: {bleu_score.score:.2f}")
    
    # Calculate METEOR
    meteor = load("meteor")
    meteor_score = meteor.compute(predictions=all_predictions, references=all_references)
    print(f"METEOR: {meteor_score['meteor']:.4f}")
    
    # Calculate ROUGE-L
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    
    # Save evaluation results
    eval_results = {
        "metrics": {
            "sacrebleu": bleu_score.score,
            "meteor": meteor_score['meteor'],
            "rouge_l": rouge_score['rougeL'],
            "num_test_samples": len(all_predictions)
        },
        "translations": [
            {
                "input": inp,
                "prediction": pred,
                "reference": ref
            }
            for inp, pred, ref in zip(all_inputs, all_predictions, all_references)
        ]
    }
    
    with open(EVALUATION_RESULTS_PATH, "w", encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete. Results saved to {EVALUATION_RESULTS_PATH}")
    
    # Print a few sample translations
    print("\n--- Sample Translations ---")
    for i in range(min(SAMPLE_TRANSLATIONS_TO_SHOW, len(all_inputs))):
        print(f"Source (Dialect): {all_inputs[i]}")
        print(f"Target (Standard): {all_references[i]}")
        print(f"Generated: {all_predictions[i]}")
        print("-" * 50)

if __name__ == "__main__":
    main()