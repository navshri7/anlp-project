import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import re
from torch.nn import functional as F

# MACROS - Hyperparameters
MODEL_PATH = "./chittagong-translation-model"
DATA_PATH = "data/data.json"
TRAIN_DATA_PATH = "data/data_train.json"
MAX_LENGTH = 512
PADDING = True
TRUNCATION = True
ADD_SPECIAL_TOKENS = False
MIN_TOKEN_LENGTH = 0
MAX_SAMPLES_DEFAULT = 20
OUTPUT_PATH_DEFAULT = "extracted_features.json"
TOKEN_FEATURES_OUTPUT = "extracted_token_features.json"
SAMPLE_TEXT_DEFAULT = "সিগারেট অর নলি"
TOP_TOKENS_DISPLAY = 10
PROGRESS_DISPLAY_LENGTH = 50

# Special tokens to filter out
SPECIAL_TOKENS = ['<pad>', '</s>', '<unk>', '<s>']
SUBWORD_PREFIX = '▁'

class EmbeddingExtractor:
    def __init__(self, model_path=MODEL_PATH, data_path=DATA_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract all dialect texts
        self.dialect_texts = [item["Dialect"] for item in self.data]
        
        # Calculate token frequencies
        self.token_frequencies = self._calculate_token_frequencies()
    
    def _calculate_token_frequencies(self):
        """Calculate frequency of each token in the dataset using the model's tokenizer"""
        all_tokens = []
        
        for text in self.dialect_texts:
            # Use the model's tokenizer to get proper tokens
            tokenized = self.tokenizer(
                text,
                add_special_tokens=ADD_SPECIAL_TOKENS,
                return_tensors="pt",
                truncation=TRUNCATION,
                max_length=MAX_LENGTH
            )
            
            # Convert token IDs back to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
            
            # Clean tokens and filter out special tokens
            for token in tokens:
                # Remove subword prefix and clean token
                clean_token = token.replace(SUBWORD_PREFIX, '').strip()
                
                # Skip empty tokens, special tokens, and very short tokens
                if (clean_token and 
                    not token.startswith('<') and 
                    not token.endswith('>') and
                    clean_token not in SPECIAL_TOKENS and
                    len(clean_token) > MIN_TOKEN_LENGTH):
                    all_tokens.append(clean_token.lower())
        
        return Counter(all_tokens)
    
    def extract_embeddings_and_attention(self, input_text, target_text=None):
        """Extract embeddings and cross-attention scores for input text"""
        
        # If no target provided, use the input as target (for demonstration)
        if target_text is None:
            target_text = input_text
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=PADDING,
            truncation=TRUNCATION,
            max_length=MAX_LENGTH
        )
        
        target_encoding = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=PADDING,
            truncation=TRUNCATION,
            max_length=MAX_LENGTH
        )
        
        with torch.no_grad():
            # Forward pass with output attentions
            outputs = self.model(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                decoder_input_ids=target_encoding["input_ids"],
                output_attentions=True,
                output_hidden_states=True
            )
            
            # Extract encoder embeddings (input embeddings)
            encoder_embeddings = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # Extract decoder embeddings (target embeddings)
            decoder_embeddings = outputs.decoder_hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Extract cross-attention scores
            cross_attentions = outputs.cross_attentions  # List of attention matrices from each layer
            
            # Use the last layer's cross-attention
            last_cross_attention = cross_attentions[-1]  # [batch_size, num_heads, target_seq_len, source_seq_len]
            
            # Average across attention heads
            avg_cross_attention = last_cross_attention.mean(dim=1)  # [batch_size, target_seq_len, source_seq_len]
        
        return {
            "input_tokens": self.tokenizer.convert_ids_to_tokens(input_encoding["input_ids"][0]),
            "target_tokens": self.tokenizer.convert_ids_to_tokens(target_encoding["input_ids"][0]),
            "encoder_embeddings": encoder_embeddings[0],  # [seq_len, hidden_size]
            "decoder_embeddings": decoder_embeddings[0],  # [seq_len, hidden_size]
            "cross_attention": avg_cross_attention[0],    # [target_seq_len, source_seq_len]
            "input_text": input_text,
            "target_text": target_text
        }
    
    def get_token_frequency(self, token):
        """Get frequency of a token in the dataset"""
        return self.token_frequencies.get(token.lower(), 0)
    
    def extract_token_level_features(self, input_text, target_text=None):
        """Extract features for each token in the input text"""
        results = self.extract_embeddings_and_attention(input_text, target_text)
        
        token_features = []
        input_tokens = results["input_tokens"]
        
        for i, token in enumerate(input_tokens):
            if token not in SPECIAL_TOKENS:
                # Get token without special characters
                clean_token = token.replace(SUBWORD_PREFIX, '').strip()
                
                if clean_token:  # Skip empty tokens
                    token_feature = {
                        "token": token,
                        "clean_token": clean_token,
                        "position": i,
                        "input_embedding": results["encoder_embeddings"][i].numpy(),
                        "frequency": self.get_token_frequency(clean_token),
                        "cross_attention_scores": results["cross_attention"][:, i].numpy() if i < results["cross_attention"].shape[1] else None,
                        "cross_attention_mean": float(np.mean(results["cross_attention"][:, i].numpy())) if i < results["cross_attention"].shape[1] else None,
                        "cross_attention_max": float(np.max(results["cross_attention"][:, i].numpy())) if i < results["cross_attention"].shape[1] else None,
                        "cross_attention_2norm": float(np.linalg.norm(results["cross_attention"][:, i].numpy())) if i < results["cross_attention"].shape[1] else None,
                        "cross_attention_lennorm": float(np.linalg.norm(results["cross_attention"][:, i].numpy(), ord=len(results["cross_attention"][:, i].numpy()))) if i < results["cross_attention"].shape[1] else None,
                        "cross_attention_min": float(np.min(results["cross_attention"][:, i].numpy())) if i < results["cross_attention"].shape[1] else None
                    }
                    
                    # Add corresponding target embedding if available
                    if i < len(results["decoder_embeddings"]):
                        token_feature["target_embedding"] = results["decoder_embeddings"][i].numpy()
                    
                    token_features.append(token_feature)
        
        return token_features
    
    def process_all_data(self, max_samples=None):
        """Process all dialect texts and extract features"""
        all_features = []
        
        # If max_samples is None, process all data
        if max_samples is None:
            max_samples = len(self.dialect_texts)
        
        for i, text in enumerate(self.dialect_texts[:max_samples]):
            print(f"Processing sample {i+1}/{min(max_samples, len(self.dialect_texts))}: {text[:PROGRESS_DISPLAY_LENGTH]}...")
            
            try:
                token_features = self.extract_token_level_features(text)
                all_features.extend(token_features)
            except Exception as e:
                print(f"Error processing text: {text[:PROGRESS_DISPLAY_LENGTH]}... - {str(e)}")
                continue
        
        return all_features
    
    def save_features(self, features, output_path=OUTPUT_PATH_DEFAULT):
        """Save features to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_features = []
        for feature in features:
            json_feature = {}
            for key, value in feature.items():
                if isinstance(value, np.ndarray):
                    json_feature[key] = value.tolist()
                else:
                    json_feature[key] = value
            json_features.append(json_feature)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_features, f, ensure_ascii=False, indent=2)
        
        print(f"Features saved to {output_path}")

def main():
    # Initialize extractor
    extractor = EmbeddingExtractor(data_path=TRAIN_DATA_PATH)
    
    # Example: Extract features for a single text
    sample_text = SAMPLE_TEXT_DEFAULT
    print(f"Extracting features for: {sample_text}")
    
    token_features = extractor.extract_token_level_features(sample_text)
    
    print("\ntoken-level features:")
    for feature in token_features:
        print(f"Token: {feature['token']}")
        print(f"Clean token: {feature['clean_token']}")
        print(f"Frequency: {feature['frequency']}")
        print(f"Input embedding shape: {feature['input_embedding'].shape}")
        if 'target_embedding' in feature:
            print(f"Target embedding shape: {feature['target_embedding'].shape}")
        if feature['cross_attention_scores'] is not None:
            print(f"Cross-attention scores shape: {feature['cross_attention_scores'].shape}")
        print("-" * PROGRESS_DISPLAY_LENGTH)
    
    # Process all data (limited to first samples for efficiency)
    print("\nProcessing all data...")
    all_features = extractor.process_all_data(max_samples=MAX_SAMPLES_DEFAULT)
    
    # Save features
    extractor.save_features(all_features, TOKEN_FEATURES_OUTPUT)
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total tokens processed: {len(all_features)}")
    print(f"Unique tokens: {len(set(f['clean_token'] for f in all_features))}")
    
    # Top frequent tokens
    token_freq_pairs = [(f['clean_token'], f['frequency']) for f in all_features]
    unique_token_freqs = {}
    for token, freq in token_freq_pairs:
        if token not in unique_token_freqs:
            unique_token_freqs[token] = freq
    
    top_tokens = sorted(unique_token_freqs.items(), key=lambda x: x[1], reverse=True)[:TOP_TOKENS_DISPLAY]
    print(f"\nTop {TOP_TOKENS_DISPLAY} frequent tokens:")
    for token, freq in top_tokens:
        print(f"{token}: {freq}")

if __name__ == "__main__":
    main()