import json
import os

SPLIT_RATIO = 0.5

def partition_data(split_ratio=0.5):
    # Create directories if they don't exist
    os.makedirs('pre_training', exist_ok=True)
    os.makedirs('post_training', exist_ok=True)
    
    # List of data files to partition
    data_files = ['data_train.json', 'data_test.json', 'data_val.json']
    
    for file_name in data_files:
        # Load the original data
        with open("data/"+file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Split data based on user-defined ratio
        split_point = int(len(data) * split_ratio)
        pre_training_data = data[:split_point]
        post_training_data = data[split_point:]
        
        # Save pre-training data
        pre_training_path = os.path.join('pre_training', file_name)
        with open(pre_training_path, 'w', encoding='utf-8') as f:
            json.dump(pre_training_data, f, indent=2, ensure_ascii=False)
        
        # Save post-training data
        post_training_path = os.path.join('post_training', file_name)
        with open(post_training_path, 'w', encoding='utf-8') as f:
            json.dump(post_training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Partitioned {file_name}: {len(pre_training_data)} items to pre_training, {len(post_training_data)} items to post_training")

if __name__ == "__main__":
    partition_data(split_ratio=SPLIT_RATIO)