import pandas as pd
import os

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
            df = pd.read_csv(file)
            df.rename(
                columns=parse_column_name,
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
                    num_stats[col].append(df[col].dropna())
        
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
    print("| Column | Mean | Variance | Median | 25th percentile | 50th percentile | 75th percentile |")
    print("|--------|------|----------|--------|-----------------|-----------------|-----------------|")
    for col, values_list in num_stats.items():
        if not values_list:
            continue
        combined_values = pd.concat(values_list)
        mean = combined_values.mean(),
        var =  combined_values.var(),
        q25 = combined_values.quantile(0.25),
        q50 = combined_values.quantile(0.5),
        q75 = combined_values.quantile(0.75),

        print(f'| {col}  | {mean[0]:.4f}  | {var[0]:.4f}  | {q25[0]:.4f}  | {q50[0]:.4f}  | {q75[0]:.4f}  |')

if __name__ == "__main__":
    csv_dir = 'dataset/AWID3_CSV_balanced'
    all_files = os.listdir(csv_dir)
    kr00k_files = list(filter(lambda f : f.startswith('Kr00k'), all_files))
    other_files = list(filter(lambda f : not f.startswith('Kr00k'), all_files))
    kr00k_paths = [os.path.join(csv_dir, f) for f in kr00k_files]
    other_paths = [os.path.join(csv_dir, f) for f in other_files]
    
    categorical_columns = [
    'wlan.fc.type_0',
    'wlan.fc.type_1',
    'wlan.fc.type_2',
    'wlan.fc.subtype_0',
    'wlan.fc.subtype_1',
    'wlan.fc.subtype_2',
    'wlan.fc.subtype_3',
    'wlan.fc.subtype_4',
    'wlan.fc.subtype_5',
    'wlan.fc.subtype_6',
    'wlan.fc.subtype_7',
    'wlan.fc.subtype_8',
    'wlan.fc.subtype_9',
    'wlan.fc.subtype_10',
    'wlan.fc.subtype_11',
    'wlan.fc.subtype_12',
    'wlan.fc.subtype_13',
    'wlan.fc.subtype_14',
    'wlan.fc.subtype_15',
    'wlan.fc.ds_0',
    'wlan.fc.ds_1',
    'wlan.fc.ds_2',
    'wlan.fc.ds_3',
    'radiotap.present.tsft',
    'radiotap.channel.flags.cck',
    'radiotap.channel.flags.ofdm',
    'wlan.fc.frag',
    'wlan.fc.retry',
    'wlan.fc.pwrmgt', 
    'wlan.fc.moredata', 
    'wlan.fc.protected', 
    '2ghz_spectrum',
    '5ghz_spectrum', 
    'Label']
    numerical_columns = [
    'frame.len', 
    'radiotap.length',
    'frame.time_delta',
    'wlan.duration',
    'radiotap.dbm_antsignal',
    'freq', ]

    print('Kr00k')    
    process_csv_files(kr00k_paths, categorical_columns, numerical_columns)
    print('Other')    
    process_csv_files(other_paths, categorical_columns, numerical_columns)
    
    
    