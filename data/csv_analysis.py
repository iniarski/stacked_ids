import pandas as pd
import os

from awid2_data_preprocessing import awid2_cols, awid2_usecols, awid2_rename, to_int

def parse_column_name(c):
    c = str(c).split("__")[-1]
    if c.endswith('.0'):
        return c[:-2]
    return c


def process_csv_files(files, categorical_columns, numerical_columns):

    cat_counts = {col: {} for col in categorical_columns}
    num_stats = {col: [] for col in numerical_columns}
    
    for file in files:
        try:
            df = pd.read_csv(file, names=awid2_cols, usecols=awid2_usecols)
            df.rename(
                columns=awid2_rename,
                #columns=parse_column_name,
                inplace=True
            )
            
            for col in categorical_columns:
                if col in df.columns:
                    value_counts = df[col].value_counts()
                    for category, count in value_counts.items():
                        if category not in cat_counts[col]:
                            cat_counts[col][category] = 0
                        cat_counts[col][category] += count

            for col in numerical_columns:
                if col in df.columns:
                    num_stats[col].append(pd.to_numeric(df[col], errors='coerce').dropna())
                else:
                    print(col, 'na')
        
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    print("### Categorical Column Summary")
    for col, counts in cat_counts.items():
        total = sum(counts.values())
        print(f"\n#### {col}")
        print("| Category | Count | Percentage |")
        print("|----------|-------|------------|")
        for category, count in counts.items():
            percentage = (count / total) * 100
            print(f"| {category} | {count} | {percentage:.2f}% |")
    
    # Summarize numerical columns
    print("\n### Numerical Column Summary")
    print("| Column | Mean | Variance | Min | 25th percentile | 50th percentile | 75th percentile | Max |")
    print("|--------|------|----------|-----|-----------------|-----------------|-----------------|-----|")
    for col, values_list in num_stats.items():
        if not values_list:
            continue
        combined_values = pd.concat(values_list)
        mean = combined_values.mean()
        var =  combined_values.var()
        _min = combined_values.min()
        q25 = combined_values.quantile(0.25)
        q50 = combined_values.quantile(0.5)
        q75 = combined_values.quantile(0.75)
        _max = combined_values.max()

        print(f'| {col}  | {mean:.4f}  | {_min:.4f}  | {var:.4f}  | {q25:.4f}  | {q50:.4f}  | {q75:.4f}  | {_max:.4f}  |')

if __name__ == "__main__":
    csv_dir = 'dataset/AWID2/AWID-CLS-F-Trn'
    files = os.listdir(csv_dir)
    from random import shuffle
    shuffle(files)
    files = files[:len(files) // 5]
    paths = [os.path.join(csv_dir, f) for f in files]
    
    categorical_columns = [
    'wlan.fc.type',
    'wlan.fc.subtype',
    'wlan.fc.ds',
    'radiotap.present.tsft',
    'radiotap.channel.flags.cck',
    'radiotap.channel.flags.ofdm',
    'wlan.fc.frag',
    'wlan.fc.retry',
    'wlan.fc.pwrmgt', 
    'wlan.fc.moredata', 
    'wlan.fc.protected', 
    'Label']
    numerical_columns = [
    'frame.len', 
    'radiotap.length',
    'frame.time_delta',
    'wlan.duration',
    'radiotap.dbm_antsignal',
    ]
    
    process_csv_files(paths, categorical_columns, numerical_columns)


    
    
    