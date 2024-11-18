import os
import pandas as pd
from sklearn.utils import resample

src_directory_path = 'dataset/AWID2_CSV_preprocessed/test'
target_directory_path = 'dataset/AWID2_CSV_preprocessed/test_balanced'


def balance_csv_file(src_path, target_path):
    df = pd.read_csv(src_path)

    majority_class = df[df['Label'] == 0]
    minority_class = df[df['Label'] != 0]

    majority_class_downsampled = resample(majority_class, 
                                          replace=False,
                                          n_samples=len(minority_class))

    balanced_df = pd.concat([majority_class_downsampled, minority_class])

    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)


    balanced_df.to_csv(target_path, index=False)
    print(f"| {target_path.split('/')[-1].split('.')[0]} | 0 | {len(majority_class_downsampled)} | {minority_class['Label'].head(1).iloc[0]} | {len(minority_class)} |")

def process_csvs(src_dir, target_dir):
    for file in os.listdir(src_dir):
        if file.endswith('.csv'):
            src_path = os.path.join(src_dir, file)
            target_path = os.path.join(target_dir, file)
            if os.path.exists(target_path):
                print(f'File {target_path} already exists')
                continue
            balance_csv_file(src_path, target_path)

if __name__ == "__main__":
    process_csvs(src_directory_path, target_directory_path)
