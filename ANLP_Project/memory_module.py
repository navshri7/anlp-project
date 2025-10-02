import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import torch.nn as nn
import torch.optim as optim

# HYPERPARAMETERS AS MACROS
EPOCHS = 1
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_HIDDEN_DIM = 64
MAX_HIDDEN_DIM = 256
HIDDEN_DIM_MULTIPLIER = 2
PATIENCE = 100
LOG_INTERVAL = 100
SORT_KEY = "frequency"
# ["frequency", "cross_attention_mean", "cross_attention_max", "cross_attention_min", "cross_attention_2norm", "cross_attention_lennorm"]
THRESHOLD = 10        # Default threshold percentage
ORDER = "desc"        # Default order

class MemoryFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MemoryFFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def load_token_features(file_path):
    """Load token features from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
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
    return valid_entries

def get_user_preferences():
    """Get user preferences for sorting and threshold"""
    sort_keys = ["frequency", "cross_attention_mean", "cross_attention_max", 
                 "cross_attention_min", "cross_attention_2norm", "cross_attention_lennorm"]
    
    print("Available sorting keys:")
    for i, key in enumerate(sort_keys, 1):
        print(f"{i}. {key}")
    
    while True:
        try:
            choice = int(input("Choose a key (1-6): ")) - 1
            if 0 <= choice < len(sort_keys):
                selected_key = sort_keys[choice]
                break
            else:
                print("Invalid choice. Please select 1-6.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        order = input("Sort order (asc/desc): ").lower()
        if order in ['asc', 'desc']:
            reverse = (order == 'desc')
            break
        else:
            print("Please enter 'asc' or 'desc'.")
    
    while True:
        try:
            threshold = int(input("Enter threshold (number of top entries): "))
            if threshold > 0:
                break
            else:
                print("Threshold must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    return selected_key, reverse, threshold

def sort_and_filter_data(data, sort_key, reverse, threshold):
    """Sort data by specified key and return top entries based on percentage"""
    # Filter entries that have the sort key
    valid_data = [entry for entry in data if sort_key in entry and entry[sort_key] is not None]
    
    # Sort by the specified key
    sorted_data = sorted(valid_data, key=lambda x: x[sort_key], reverse=reverse)
    
    # Calculate number of entries based on percentage
    num_entries = max(1, int(len(sorted_data) * threshold / 100))
    
    # Return top percentage of entries
    return sorted_data[:num_entries]

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
    
    # Convert to tensors
    X = torch.tensor(input_embeddings, dtype=torch.float32)
    y = torch.tensor(target_embeddings, dtype=torch.float32)
    
    return X, y

def train_memory_model(X, y, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Train the memory FFNN model"""
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    hidden_dim = max(MIN_HIDDEN_DIM, min(MAX_HIDDEN_DIM, input_dim * HIDDEN_DIM_MULTIPLIER))
    
    model = MemoryFFNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Split data for training and validation
    if len(X) > 1:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        X_train, X_val, y_train, y_val = X, X, y, y
    
    print(f"Training model with {len(X_train)} samples...")
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % LOG_INTERVAL == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    return model

def save_model_and_info(model, sort_key, threshold, reverse, model_dir="memory_model"):
    """Save the trained model and associated information"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model filename
    order_str = "desc" if reverse else "asc"
    model_filename = f"memory_model_{sort_key}_{order_str}_top{threshold}.pt"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Save model info
    info = {
        "sort_key": sort_key,
        "threshold": threshold,
        "sort_order": order_str,
        "model_architecture": {
            "input_dim": model.layers[0].in_features,
            "hidden_dim": model.layers[0].out_features,
            "output_dim": model.layers[-1].out_features
        }
    }
    
    info_filename = f"memory_model_{sort_key}_{order_str}_top{threshold}_info.json"
    info_path = os.path.join(model_dir, info_filename)
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Model info saved to: {info_path}")

def main():
    sort_key = SORT_KEY
    threshold = THRESHOLD
    reverse = (ORDER == "desc")
    
    # Load data
    file_path = "extracted_token_features.json"
    print(f"Loading data from {file_path}...")
    
    try:
        data = load_token_features(file_path)
        print(f"Loaded {len(data)} entries.")
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error parsing JSON file {file_path}")
        return
    
    # Filter entries with valid embeddings
    valid_data = filter_valid_entries(data)
    print(f"Found {len(valid_data)} entries with valid input and target embeddings.")
    
    if not valid_data:
        print("No valid entries found. Please check your data.")
        return
    
    # Sort and filter data
    filtered_data = sort_and_filter_data(valid_data, sort_key, reverse, threshold)
    print(f"Selected top {len(filtered_data)} entries based on {sort_key} ({'descending' if reverse else 'ascending'}).")
    
    if not filtered_data:
        print("No data found after filtering. Please check your parameters.")
        return
    
    # Prepare training data
    try:
        X, y = prepare_training_data(filtered_data)
        print(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} input features, {y.shape[1]} output features.")
    except ValueError as e:
        print(f"Error preparing training data: {e}")
        return
    
    # Train model
    model = train_memory_model(X, y)
    
    # Save model
    save_model_and_info(model, sort_key, threshold, reverse)
    
    print("Memory module training completed successfully!")
if __name__ == "__main__":
    main()