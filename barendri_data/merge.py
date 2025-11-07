import json
import random
import glob
import os

# MACROS - Hyperparameters
TRAIN_SPLIT_RATIO = 0.7
TEST_SPLIT_RATIO = 0.15
VAL_SPLIT_RATIO = 0.15

# Key mapping configurations
STANDARD_KEYS = ["Bangla", " Bangla", "bangla_speech", "bangla_speech ", "Bangla Speech", "Bangla_Speech"]
DIALECT_KEYS = ["Chittagong", "Chittagong Dialect", "chittagong_speech", "chittagong_speech ", "chittagong_bangla_speech", "chittagong_bangla_speech "]
STANDARD_OUTPUT_KEY = "Standard"
DIALECT_OUTPUT_KEY = "Dialect"

# File output settings
JSON_INDENT = 2
ENCODING = 'utf-8'

def merge_multiple_json_files(file_paths, output_path):
    """Merge multiple JSON files into one single JSON file"""
    
    all_data = []
    
    # Read all JSON files
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding=ENCODING) as f:
                data = json.load(f)
                
                # Convert to list if it's a dict or other type
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.append(data)
                else:
                    all_data.append(data)
                    
            print(f"Loaded {file_path}: {len(data) if isinstance(data, list) else 1} records")
            
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping...")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}, skipping...")
    
    # Update keys in all data
    for item in all_data:
        if isinstance(item, dict):
            # Create a new dictionary with updated keys
            updated_item = {}
            for key, value in item.items():
                if key in STANDARD_KEYS:
                    updated_item[STANDARD_OUTPUT_KEY] = value
                elif key in DIALECT_KEYS:
                    updated_item[DIALECT_OUTPUT_KEY] = value
                else:
                    updated_item[key] = value
            # Update the item in place
            item.clear()
            item.update(updated_item)
    
    # Shuffle the merged data for randomness
    random.shuffle(all_data)
    
    # Split data into train, test, val partitions using macros
    total_size = len(all_data)
    train_size = int(TRAIN_SPLIT_RATIO * total_size)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    val_data = all_data[train_size + test_size:]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save train, test, val partitions
    base_name = os.path.splitext(output_path)[0]
    
    train_path = f"{base_name}_train.json"
    test_path = f"{base_name}_test.json"
    val_path = f"{base_name}_val.json"
    
    with open(train_path, 'w', encoding=ENCODING) as f:
        json.dump(train_data, f, indent=JSON_INDENT, ensure_ascii=False)
    
    with open(test_path, 'w', encoding=ENCODING) as f:
        json.dump(test_data, f, indent=JSON_INDENT, ensure_ascii=False)
    
    with open(val_path, 'w', encoding=ENCODING) as f:
        json.dump(val_data, f, indent=JSON_INDENT, ensure_ascii=False)
    
    print(f"Successfully merged {len(file_paths)} files into 3 partitions:")
    print(f"  Train: {train_path} ({len(train_data)} records)")
    print(f"  Test: {test_path} ({len(test_data)} records)")
    print(f"  Val: {val_path} ({len(val_data)} records)")
    print(f"  Total: {len(all_data)} records")

def merge_json_files_from_pattern(pattern, output_path):
    """Merge JSON files matching a pattern"""
    file_paths = glob.glob(pattern)
    if not file_paths:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(file_paths)} files matching pattern")
    merge_multiple_json_files(file_paths, output_path)

# Usage examples
if __name__ == "__main__":
    # Method 1: Specify individual files
    files_to_merge = [
        "data/data.json",
    ]
    output = "data/data.json"
    merge_multiple_json_files(files_to_merge, output)