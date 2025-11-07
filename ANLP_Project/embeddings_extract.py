#embeddings_extract.py
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import re
from torch.nn import functional as F

# MACROS - Hyperparameters
MODEL_PATH = "./chittagong-translation-model"
TRAIN_DATA_PATH = "pre_training/data_train.json"
MAX_LENGTH = 512
PADDING = True
TRUNCATION = True
ADD_SPECIAL_TOKENS = False
MIN_TOKEN_LENGTH = 0
MAX_SAMPLES_DEFAULT = None  # Use all data
TOKEN_FEATURES_OUTPUT = "extracted_token_features.json"
SAMPLE_TEXT_DEFAULT = "সিগারেট অর নলি"
TOP_TOKENS_DISPLAY = 10
PROGRESS_DISPLAY_LENGTH = 50
BATCH_SIZE = 32  # Process in batches for efficiency

# Special tokens to filter out
SPECIAL_TOKENS = ['<pad>', '</s>', '<unk>', '<s>']
SUBWORD_PREFIX = '▁'

class EmbeddingExtractor:
    def __init__(self, model_path=MODEL_PATH, data_path=TRAIN_DATA_PATH):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples")
        
        # Extract all dialect texts
        self.dialect_texts = [item["Dialect"] for item in self.data if "Dialect" in item]
        self.standard_texts = [item["Standard"] for item in self.data if "Standard" in item]
        
        # Calculate token frequencies
        print("Calculating token frequencies...")
        self.token_frequencies = self._calculate_token_frequencies()
        print(f"Found {len(self.token_frequencies)} unique tokens")
    
    def _calculate_token_frequencies(self):
        """Calculate frequency of each token in the dataset using the model's tokenizer"""
        all_tokens = []
        
        for text in self.dialect_texts:
            tokenized = self.tokenizer(
                text,
                add_special_tokens=ADD_SPECIAL_TOKENS,
                return_tensors="pt",
                truncation=TRUNCATION,
                max_length=MAX_LENGTH
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
            
            for token in tokens:
                clean_token = token.replace(SUBWORD_PREFIX, '').strip()
                
                if (clean_token and 
                    not token.startswith('<') and 
                    not token.endswith('>') and
                    clean_token not in SPECIAL_TOKENS and
                    len(clean_token) > MIN_TOKEN_LENGTH):
                    all_tokens.append(clean_token.lower())
        
        return Counter(all_tokens)
    
    def extract_embeddings_and_attention(self, input_text, target_text):
        """Extract embeddings and cross-attention scores for input text"""
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=PADDING,
            truncation=TRUNCATION,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        target_encoding = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=PADDING,
            truncation=TRUNCATION,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                decoder_input_ids=target_encoding["input_ids"],
                output_attentions=True,
                output_hidden_states=True
            )
            
            encoder_embeddings = outputs.encoder_last_hidden_state
            decoder_embeddings = outputs.decoder_hidden_states[-1]
            cross_attentions = outputs.cross_attentions
            last_cross_attention = cross_attentions[-1]
            avg_cross_attention = last_cross_attention.mean(dim=1)
        
        return {
            "input_tokens": self.tokenizer.convert_ids_to_tokens(input_encoding["input_ids"][0]),
            "target_tokens": self.tokenizer.convert_ids_to_tokens(target_encoding["input_ids"][0]),
            "encoder_embeddings": encoder_embeddings[0].cpu(),
            "decoder_embeddings": decoder_embeddings[0].cpu(),
            "cross_attention": avg_cross_attention[0].cpu(),
            "input_text": input_text,
            "target_text": target_text
        }
    
    def get_token_frequency(self, token):
        """Get frequency of a token in the dataset"""
        return self.token_frequencies.get(token.lower(), 0)
    
    def extract_token_level_features(self, input_text, target_text):
        """Extract features for each token in the input text"""
        results = self.extract_embeddings_and_attention(input_text, target_text)
        
        token_features = []
        input_tokens = results["input_tokens"]
        
        for i, token in enumerate(input_tokens):
            if token not in SPECIAL_TOKENS:
                clean_token = token.replace(SUBWORD_PREFIX, '').strip()
                
                if clean_token:
                    # Get cross-attention scores for this position
                    cross_attn = results["cross_attention"][:, i].numpy() if i < results["cross_attention"].shape[1] else None
                    
                    token_feature = {
                        "token": token,
                        "clean_token": clean_token,
                        "position": i,
                        "input_embedding": results["encoder_embeddings"][i].numpy().tolist(),
                        "frequency": self.get_token_frequency(clean_token),
                    }
                    
                    # Add cross-attention statistics
                    if cross_attn is not None:
                        token_feature["cross_attention_scores"] = cross_attn.tolist()
                        token_feature["cross_attention_mean"] = float(np.mean(cross_attn))
                        token_feature["cross_attention_max"] = float(np.max(cross_attn))
                        token_feature["cross_attention_min"] = float(np.min(cross_attn))
                        token_feature["cross_attention_2norm"] = float(np.linalg.norm(cross_attn, ord=2))
                        # Avoid division by zero for length normalization
                        if len(cross_attn) > 0:
                            token_feature["cross_attention_lennorm"] = float(np.linalg.norm(cross_attn) / np.sqrt(len(cross_attn)))
                        else:
                            token_feature["cross_attention_lennorm"] = 0.0
                    
                    # Add corresponding target embedding if available
                    if i < len(results["decoder_embeddings"]):
                        token_feature["target_embedding"] = results["decoder_embeddings"][i].numpy().tolist()
                    
                    token_features.append(token_feature)
        
        return token_features
    
    def process_all_data(self, max_samples=None):
        """Process all dialect texts and extract features"""
        all_features = []
        
        if max_samples is None:
            max_samples = len(self.dialect_texts)
        
        total_samples = min(max_samples, len(self.dialect_texts))
        
        for i in range(total_samples):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{total_samples}...")
            
            try:
                input_text = self.dialect_texts[i]
                target_text = self.standard_texts[i] if i < len(self.standard_texts) else input_text
                
                token_features = self.extract_token_level_features(input_text, target_text)
                all_features.extend(token_features)
            except Exception as e:
                print(f"Error processing sample {i+1}: {str(e)}")
                continue
        
        return all_features
    
    def save_features(self, features, output_path=TOKEN_FEATURES_OUTPUT):
        """Save features to JSON file"""
        print(f"Saving {len(features)} token features to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        
        print(f"Features saved to {output_path}")

def main():
    # Initialize extractor
    extractor = EmbeddingExtractor()
    
    # Process all data
    print("\n" + "="*70)
    print("EXTRACTING TOKEN-LEVEL FEATURES FROM ALL DATA")
    print("="*70 + "\n")
    
    all_features = extractor.process_all_data(max_samples=MAX_SAMPLES_DEFAULT)
    
    # Save features
    extractor.save_features(all_features, TOKEN_FEATURES_OUTPUT)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total tokens processed: {len(all_features):,}")
    print(f"Unique tokens: {len(set(f['clean_token'] for f in all_features)):,}")
    
    # Top frequent tokens
    token_freq_pairs = [(f['clean_token'], f['frequency']) for f in all_features]
    unique_token_freqs = {}
    for token, freq in token_freq_pairs:
        if token not in unique_token_freqs:
            unique_token_freqs[token] = freq
    
    top_tokens = sorted(unique_token_freqs.items(), key=lambda x: x[1], reverse=True)[:TOP_TOKENS_DISPLAY]
    print(f"\nTop {TOP_TOKENS_DISPLAY} frequent tokens:")
    for token, freq in top_tokens:
        print(f"  {token}: {freq:,}")
    
    # Statistics on cross-attention
    tokens_with_attn = [f for f in all_features if 'cross_attention_2norm' in f]
    if tokens_with_attn:
        attn_norms = [f['cross_attention_2norm'] for f in tokens_with_attn]
        print(f"\nCross-attention L2 norm statistics:")
        print(f"  Mean: {np.mean(attn_norms):.4f}")
        print(f"  Std: {np.std(attn_norms):.4f}")
        print(f"  25th percentile: {np.percentile(attn_norms, 25):.4f}")
        print(f"  Median (50th): {np.percentile(attn_norms, 50):.4f}")
        print(f"  75th percentile: {np.percentile(attn_norms, 75):.4f}")
    
    print("\n" + "="*70)
    print("Feature extraction completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()