#memory_module.py
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch.nn as nn
import torch.optim as optim

# HYPERPARAMETERS AS MACROS
EPOCHS = 1000
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_HIDDEN_DIM = 128
MAX_HIDDEN_DIM = 512
HIDDEN_DIM_MULTIPLIER = 2
PATIENCE = 100
LOG_INTERVAL = 100

# ABLATION CONFIGURATIONS
ABLATIONS = [
    {"sort_key": "frequency", "order": "asc", "percentile": 25, "name": "freq_asc_p25"},
    {"sort_key": "frequency", "order": "asc", "percentile": 50, "name": "freq_asc_p50"},
    {"sort_key": "frequency", "order": "asc", "percentile": 75, "name": "freq_asc_p75"},
    {"sort_key": "cross_attention_2norm", "order": "desc", "percentile": 25, "name": "attn_desc_p25"},
    {"sort_key": "cross_attention_2norm", "order": "desc", "percentile": 50, "name": "attn_desc_p50"},
    {"sort_key": "cross_attention_2norm", "order": "desc", "percentile": 75, "name": "attn_desc_p75"},
]

class MemoryFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MemoryFFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def load_token_features(file_path):
    """Load token features from JSON file"""
    print(f"Loading token features from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} token features")
    return data

def filter_valid_entries(data):
    """Filter entries that have both input and target embeddings"""
    valid_entries = []
    for entry in data:
        if (entry.get('input_embedding') and 
            entry.get('target_embedding') and 
            len(entry['input_embedding']) > 0 and 
            len(entry['target_embedding']) > 0):
            valid_entries.append(entry)
    print(f"Found {len(valid_entries)} valid entries with embeddings")
    return valid_entries

def calculate_threshold_value(data, sort_key, percentile):
    """Calculate threshold value based on percentile"""
    values = [entry[sort_key] for entry in data if sort_key in entry and entry[sort_key] is not None]
    if not values:
        return None
    threshold = np.percentile(values, percentile)
    print(f"  {percentile}th percentile for {sort_key}: {threshold:.4f}")
    return threshold

def filter_by_percentile(data, sort_key, order, percentile):
    """Filter data based on percentile threshold"""
    # Get valid data with the sort key
    valid_data = [entry for entry in data if sort_key in entry and entry[sort_key] is not None]
    
    if not valid_data:
        print(f"  No valid data found for {sort_key}")
        return []
    
    # Calculate threshold
    threshold = calculate_threshold_value(valid_data, sort_key, percentile)
    
    if threshold is None:
        return []
    
    # Filter based on order and percentile
    if order == "asc":
        # For ascending (low values), take values <= threshold
        filtered = [entry for entry in valid_data if entry[sort_key] <= threshold]
        explanation = f"rare/low {sort_key}"
    else:
        # For descending (high values), take values >= threshold
        filtered = [entry for entry in valid_data if entry[sort_key] >= threshold]
        explanation = f"high {sort_key}"
    
    print(f"  Selected {len(filtered)} tokens with {explanation} (threshold: {threshold:.4f})")
    return filtered

def prepare_training_data(filtered_data):
    """Prepare input and target embeddings for training"""
    input_embeddings = []
    target_embeddings = []
    
    for entry in filtered_data:
        if (entry.get('input_embedding') and 
            entry.get('target_embedding') and 
            len(entry['input_embedding']) > 0 and 
            len(entry['target_embedding']) > 0):
            input_embeddings.append(entry['input_embedding'])
            target_embeddings.append(entry['target_embedding'])
    
    if not input_embeddings or not target_embeddings:
        raise ValueError("No valid input-target embedding pairs found!")
    
    X = torch.tensor(input_embeddings, dtype=torch.float32)
    y = torch.tensor(target_embeddings, dtype=torch.float32)
    
    return X, y

def train_memory_model(X, y, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Train the memory FFNN model with improved architecture"""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    hidden_dim = max(MIN_HIDDEN_DIM, min(MAX_HIDDEN_DIM, input_dim * HIDDEN_DIM_MULTIPLIER))
    
    print(f"  Model architecture: {input_dim} -> {hidden_dim} -> {hidden_dim} -> {output_dim}")
    
    model = MemoryFFNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Split data for training and validation
    if len(X) > 1:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        X_train, X_val, y_train, y_val = X, X, y, y
    
    print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if epoch % LOG_INTERVAL == 0:
            print(f"    Epoch {epoch:4d} | Train: {loss.item():.6f} | Val: {val_loss.item():.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    return model, best_val_loss

def save_model_and_info(model, ablation_config, val_loss, model_dir="memory_models"):
    """Save the trained model and associated information"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_name = ablation_config['name']
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save model info
    info = {
        "ablation_name": model_name,
        "sort_key": ablation_config['sort_key'],
        "order": ablation_config['order'],
        "percentile": ablation_config['percentile'],
        "validation_loss": float(val_loss),
        "model_architecture": {
            "input_dim": model.layers[0].in_features,
            "hidden_dim": model.layers[0].out_features,
            "output_dim": model.layers[-1].out_features
        }
    }
    
    info_path = os.path.join(model_dir, f"{model_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  Saved to: {model_path}")
    
    return model_path, info_path

def main():
    """Run all 6 ablation studies"""
    
    print("\n" + "="*70)
    print("MEMORY MODULE ABLATION STUDIES")
    print("="*70 + "\n")
    
    # Load data
    file_path = "extracted_token_features.json"
    
    try:
        data = load_token_features(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        print("Please run embeddings_extract.py first.")
        return
    
    # Filter entries with valid embeddings
    valid_data = filter_valid_entries(data)
    
    if not valid_data:
        print("No valid entries found. Please check your data.")
        return
    
    # Store results for comparison
    results = []
    
    # Run each ablation
    for i, ablation in enumerate(ABLATIONS, 1):
        print(f"\n{'='*70}")
        print(f"ABLATION {i}/6: {ablation['name']}")
        print(f"  Sort key: {ablation['sort_key']}")
        print(f"  Order: {ablation['order']}")
        print(f"  Percentile: {ablation['percentile']}")
        print("="*70)
        
        try:
            # Filter data based on ablation config
            filtered_data = filter_by_percentile(
                valid_data, 
                ablation['sort_key'], 
                ablation['order'], 
                ablation['percentile']
            )
            
            if not filtered_data:
                print(f" No data after filtering. Skipping this ablation.")
                continue
            
            # Prepare training data
            X, y = prepare_training_data(filtered_data)
            print(f"  Training data shape: X={X.shape}, y={y.shape}")
            
            # Train model
            print(f"\n  Training memory model...")
            model, val_loss = train_memory_model(X, y)
            
            # Save model
            model_path, info_path = save_model_and_info(model, ablation, val_loss)
            
            results.append({
                "ablation": ablation['name'],
                "val_loss": float(val_loss),
                "num_tokens": len(filtered_data),
                "model_path": model_path
            })
            
            print(f"  Ablation {i} completed successfully!")
            
        except Exception as e:
            print(f"  Error in ablation {i}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    
    if results:
        # Sort by validation loss
        results_sorted = sorted(results, key=lambda x: x['val_loss'])
        
        print(f"\n{'Rank':<6} {'Ablation':<20} {'Val Loss':<12} {'# Tokens':<10}")
        print("-"*70)
        for rank, result in enumerate(results_sorted, 1):
            print(f"{rank:<6} {result['ablation']:<20} {result['val_loss']:<12.6f} {result['num_tokens']:<10,}")
        
        print("\n" + "="*70)
        print(f"Best ablation: {results_sorted[0]['ablation']}")
        print(f"   Validation loss: {results_sorted[0]['val_loss']:.6f}")
        print(f"   Model path: {results_sorted[0]['model_path']}")
        print("="*70)
        
        # Save summary
        summary_path = "memory_models/ablation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "ablations": results_sorted,
                "best_ablation": results_sorted[0]['ablation']
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    print("\nAll ablations completed!")

if __name__ == "__main__":
    main()