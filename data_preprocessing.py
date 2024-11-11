import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

awid3_csv_path = 'data/AWID3_Dataset_CSV/CSV'
target_csv_dir = 'data/AWID3_preprocessed'

ranges = {
    'frame.len': (70, 3977),
    'radiotap.length': (48, 64),
    'frame.time_delta': (0.0, 3.323003),
    'wlan.duration': (0, 32581),
    'parsed_antsignal': (-259.0, -24.0),
}


categories = {
    'radiotap.channel.freq': [2417, 2472, 5180],
    'wlan.fc.type': [0, 1, 2],
    'wlan.fc.subtype': list(range(16)),
    'wlan.fc.ds': ['0x00000000', '0x00000001', '0x00000002', '0x00000003'],
    'radiotap.present.tsft': ['0-0-0', '1-0-0']
}

label_mapping = {
    'Normal' : 0,
    'Deauth' : 1,
    'Disas': 1,
    '(Re)Assoc': 1,
    'RogueAP': 2,
    'Krack' : 3,
    'Kr00k' : 3,
    'Evil_Twin' : 2,
}

def init_preprocessor():
    transformers = []

    for feature, (min_val, max_val) in ranges.items():
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.scale_ = np.array([1 / (max_val - min_val)]) 
        scaler.min_ = np.array([-min_val / (max_val - min_val)])
        transformers.append((feature, scaler, [feature]))


    for feature, cats in categories.items():
        encoder = OneHotEncoder(categories=[cats], handle_unknown='ignore')
        transformers.append((feature, encoder, [feature]))

    binary_columns = [
        'radiotap.channel.flags.cck',
        'radiotap.channel.flags.ofdm',
        'wlan.fc.frag',
        'wlan.fc.retry',
        'wlan.fc.pwrmgt',
        'wlan.fc.moredata',
        'wlan.fc.protected'
    ]
    transformers.append(('bin', 'passthrough', binary_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor


def parse_antsignal(value):
    try:
        return float(value)
    except ValueError:
        return float(value.split('-')[1]) * -1

def process_file(file_path, preprocessor, is_fitted):
    df = pd.read_csv(file_path, low_memory=False)

    df['parsed_antsignal'] = df['radiotap.dbm_antsignal'].apply(parse_antsignal)

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
    awid3_csv_dirs = os.listdir(awid3_csv_path)
    preprocessor = init_preprocessor()
    is_transformer_fitted = False

    for dir in awid3_csv_dirs:
        dir_path = os.path.join(awid3_csv_path, dir)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(awid3_csv_path, dir, file)
            transformed_df = process_file(file_path, preprocessor, is_transformer_fitted)
            is_transformer_fitted = True

            target_csv = os.path.join(target_csv_dir, file)
            transformed_df.to_csv(target_csv, index=False)

            print(f"Data saved to {target_csv}")


if __name__ == '__main__':
    process_data()
