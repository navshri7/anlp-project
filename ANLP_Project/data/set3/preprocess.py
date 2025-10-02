import json
import os

def merge_json_files():
    # Define file paths
    files = [
        "data/set3/Test/Chittagong Test Translation.json",
        "data/set3/Train/Chittagong Train Translation.json", 
        "data/set3/Validation/Chittagong Validation Translation.json"
    ]
    
    merged_data = []
    
    # Process each file
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filter data to keep only specified keys
                for item in data:
                    filtered_item = {
                        key: item[key] for key in ["bangla_speech ", "bangla_speech", "chittagong_bangla_speech ", "chittagong_bangla_speech"] 
                        if key in item
                    }
                    if filtered_item:
                        merged_data.append(filtered_item)
    # Save merged data
    output_path = "data/set3/processed_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged {len(merged_data)} items into {output_path}")

if __name__ == "__main__":
    merge_json_files()