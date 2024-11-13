import os
import pandas as pd
from collections import defaultdict

directory_path = 'data/AWID3_preprocessed'

files_with_non_zero_labels = [] 
total_label_counts = defaultdict(int) 

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue
        
        if 'Label' not in data.columns:
            print(f"File {filename} does not contain 'Label' column.")
            continue  
        
        label_counts = data['Label'].value_counts()
        print(file_path)
        for label, count in label_counts.items():
            percentage = (count / sum(label_counts)) * 100
            print(f"Label {label}: {count} ({percentage:.3f}%)")
            total_label_counts[label] += count
        
        if any(label_counts.index != 0):
            files_with_non_zero_labels.append(filename)

total_labels = sum(total_label_counts.values())

print("Files containing labels other than 0:", files_with_non_zero_labels)
print("\nTotal count of labels across all files (by label):")
for label, count in total_label_counts.items():
    percentage = (count / total_labels) * 100 if total_labels > 0 else 0
    print(f"Label {label}: {count} ({percentage:.3f}%)")
