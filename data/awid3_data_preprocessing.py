import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

awid3_csv_path = 'dataset/AWID3_CSV_raw'
target_csv_dir = 'dataset/AWID3_CSV_preprocessed'

ranges = {
    'frame.len': (70, 3220),
    'radiotap.length': (48, 64),
    'frame.time_delta': (0.0, 0.001817),
    'wlan.duration': (0, 726),
    'radiotap.dbm_antsignal': (-255.0 / 3, -.0),
}


categories = {
    'wlan.fc.type': [0, 1, 2],
    'wlan.fc.subtype': list(range(16)),
    'wlan.fc.ds': list(range(4)),
}

label_mapping = {
    'Normal' : 0,
    'Deauth' : 1,
    'Disas': 1,
    '(Re)Assoc': 1,
    'RogueAP': 2,
    'Krack' : 2,
    'Kr00k' : 1,
    'Kr00K' : 1,
    'Evil_Twin' : 2,
}

tsft_present_mapping = {
    '0-0-0' : 0,
    '1-0-0' : 1
}

band_5GHz = (5160, 5885)

def init_preprocessor():
    transformers = []

    for feature, (min_val, max_val) in ranges.items():
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.scale_ = np.array([1 / (max_val - min_val)])
        scaler.min_ = np.array([-min_val / (max_val - min_val)])
        scaler.clip = True
        transformers.append((feature, scaler, [feature]))


    for feature, cats in categories.items():
        encoder = OneHotEncoder(categories=[cats], handle_unknown='ignore')
        transformers.append((feature, encoder, [feature]))

    binary_columns = [
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
        'freq'
    ]
    transformers.append(('bin', 'passthrough', binary_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def parse_ds(ds):
    if isinstance(ds, str):
        return int(ds, 16)  
    return -1

def parse_freq(freq):
    if 2412 <= freq <= 2472:
        return (freq - 2412) / (2472 - 2412)
    elif 5160 <= freq <= 5885:
        return (freq - 5160) / (5885 - 5160)
    
    return -1

def process_file(file_path, preprocessor, is_fitted):
    df = pd.read_csv(file_path, low_memory=False)

    df['radiotap.present.tsft'] = df['radiotap.present.tsft'].apply(lambda x: tsft_present_mapping.get(x, -1))
    df['radiotap.dbm_antsignal'] *= (1 / 3)
    df['wlan.fc.ds'] = df['wlan.fc.ds'].apply(parse_ds)
    df['2ghz_spectrum'] = df['radiotap.channel.freq'].apply(lambda freq: int(2412 <= freq <= 2472))
    df['5ghz_spectrum'] = df['radiotap.channel.freq'].apply(lambda freq: int(5160 <= freq <= 5885))
    df['freq'] = df['radiotap.channel.freq'].apply(parse_freq)

    X = df.drop('Label', axis=1)
    y = df['Label'].apply(lambda l: label_mapping.get(l, -1))

    if not is_fitted:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)
    transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())

    transformed_df['Label'] = y
    transformed_df = transformed_df[transformed_df['Label'] != -1]

    return transformed_df

def process_data():
    preprocessor = init_preprocessor()
    is_transformer_fitted = False

    for file in os.listdir(awid3_csv_path):
        target_csv = os.path.join(target_csv_dir, file)
        if (not file.endswith('.csv')) or os.path.exists(target_csv):
            continue
        file_path = os.path.join(awid3_csv_path, file)
        transformed_df = process_file(file_path, preprocessor, is_transformer_fitted)
        is_transformer_fitted = True

        transformed_df.to_csv(target_csv, index=False)

        print(f"Data saved to {target_csv}")


if __name__ == '__main__':
    process_data()
