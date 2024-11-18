import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

awid2_csv_path = '/home/filip/ids/dataset/AWID2/AWID-CLS-F-Tst'
target_csv_dir = 'dataset/AWID2_CSV_preprocessed/test'

awid2_cols = [
    "frame.interface_id","frame.dlt","frame.offset_shift","frame.time_epoch","frame.time_delta","frame.time_delta_displayed","frame.time_relative","frame.len","frame.cap_len","frame.marked","frame.ignored",
    "radiotap.version","radiotap.pad","radiotap.length","radiotap.present.tsft","radiotap.present.flags","radiotap.present.rate","radiotap.present.channel","radiotap.present.fhss","radiotap.present.dbm_antsignal","radiotap.present.dbm_antnoise","radiotap.present.lock_quality","radiotap.present.tx_attenuation","radiotap.present.db_tx_attenuation","radiotap.present.dbm_tx_power","radiotap.present.antenna","radiotap.present.db_antsignal","radiotap.present.db_antnoise","radiotap.present.rxflags","radiotap.present.xchannel","radiotap.present.mcs","radiotap.present.ampdu","radiotap.present.vht","radiotap.present.reserved","radiotap.present.rtap_ns","radiotap.present.vendor_ns","radiotap.present.ext",
    "radiotap.mactime","radiotap.flags.cfp","radiotap.flags.preamble","radiotap.flags.wep","radiotap.flags.frag","radiotap.flags.fcs","radiotap.flags.datapad","radiotap.flags.badfcs","radiotap.flags.shortgi",
    "radiotap.datarate",
    "radiotap.channel.freq","radiotap.channel.type.turbo","radiotap.channel.type.cck","radiotap.channel.type.ofdm","radiotap.channel.type.2ghz","radiotap.channel.type.5ghz","radiotap.channel.type.passive","radiotap.channel.type.dynamic","radiotap.channel.type.gfsk","radiotap.channel.type.gsm","radiotap.channel.type.sturbo","radiotap.channel.type.half","radiotap.channel.type.quarter",
    "radiotap.dbm_antsignal","radiotap.antenna","radiotap.rxflags.badplcp",
    "wlan.fc.type_subtype","wlan.fc.version","wlan.fc.type","wlan.fc.subtype",
    "wlan.fc.ds","wlan.fc.frag","wlan.fc.retry","wlan.fc.pwrmgt","wlan.fc.moredata","wlan.fc.protected","wlan.fc.order",
    "wlan.duration",
    "wlan.ra","wlan.da","wlan.ta","wlan.sa","wlan.bssid","wlan.frag","wlan.seq",
    "wlan.bar.type","wlan.ba.control.ackpolicy","wlan.ba.control.multitid","wlan.ba.control.cbitmap","wlan.bar.compressed.tidinfo","wlan.ba.bm","wlan.fcs_good",
    "wlan_mgt.fixed.capabilities.ess","wlan_mgt.fixed.capabilities.ibss","wlan_mgt.fixed.capabilities.cfpoll.ap","wlan_mgt.fixed.capabilities.privacy","wlan_mgt.fixed.capabilities.preamble","wlan_mgt.fixed.capabilities.pbcc","wlan_mgt.fixed.capabilities.agility","wlan_mgt.fixed.capabilities.spec_man","wlan_mgt.fixed.capabilities.short_slot_time","wlan_mgt.fixed.capabilities.apsd","wlan_mgt.fixed.capabilities.radio_measurement","wlan_mgt.fixed.capabilities.dsss_ofdm","wlan_mgt.fixed.capabilities.del_blk_ack","wlan_mgt.fixed.capabilities.imm_blk_ack","wlan_mgt.fixed.listen_ival","wlan_mgt.fixed.current_ap","wlan_mgt.fixed.status_code","wlan_mgt.fixed.timestamp","wlan_mgt.fixed.beacon","wlan_mgt.fixed.aid","wlan_mgt.fixed.reason_code","wlan_mgt.fixed.auth.alg","wlan_mgt.fixed.auth_seq","wlan_mgt.fixed.category_code","wlan_mgt.fixed.htact","wlan_mgt.fixed.chanwidth","wlan_mgt.fixed.fragment","wlan_mgt.fixed.sequence",
    "wlan_mgt.tagged.all","wlan_mgt.ssid","wlan_mgt.ds.current_channel","wlan_mgt.tim.dtim_count","wlan_mgt.tim.dtim_period","wlan_mgt.tim.bmapctl.multicast","wlan_mgt.tim.bmapctl.offset",
    "wlan_mgt.country_info.environment",
    "wlan_mgt.rsn.version","wlan_mgt.rsn.gcs.type","wlan_mgt.rsn.pcs.count","wlan_mgt.rsn.akms.count","wlan_mgt.rsn.akms.type","wlan_mgt.rsn.capabilities.preauth","wlan_mgt.rsn.capabilities.no_pairwise","wlan_mgt.rsn.capabilities.ptksa_replay_counter","wlan_mgt.rsn.capabilities.gtksa_replay_counter","wlan_mgt.rsn.capabilities.mfpr","wlan_mgt.rsn.capabilities.mfpc","wlan_mgt.rsn.capabilities.peerkey",
    "wlan_mgt.tcprep.trsmt_pow","wlan_mgt.tcprep.link_mrg",
    "wlan.wep.iv","wlan.wep.key","wlan.wep.icv","wlan.tkip.extiv","wlan.ccmp.extiv",
    "wlan.qos.tid","wlan.qos.priority","wlan.qos.eosp","wlan.qos.ack","wlan.qos.amsdupresent","wlan.qos.buf_state_indicated","wlan.qos.bit4","wlan.qos.txop_dur_req","wlan.qos.buf_state_indicated2",
    "data.len",
    "class"]
awid2_usecols = [
    "frame.time_delta",
    "frame.len",
    "radiotap.length",
    "radiotap.present.tsft",
    "radiotap.channel.freq",
    "radiotap.channel.type.cck",
    "radiotap.channel.type.ofdm",
    "radiotap.dbm_antsignal",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fc.ds",
    "wlan.fc.frag",
    "wlan.fc.retry",
    "wlan.fc.pwrmgt",
    "wlan.fc.moredata",
    "wlan.fc.protected",
    "wlan.fc.order",
    "wlan.duration",
    "class"]

awid2_rename = {
    'radiotap.channel.type.cck' : "radiotap.channel.flags.cck",
    'radiotap.channel.type.ofdm' : "radiotap.channel.flags.ofdm",
    'class' : "Label",
}

ranges = {
    'frame.len': (52, 3202),
    'radiotap.length': (38, 52),
    'frame.time_delta': (0.0, 0.001817),
    'wlan.duration': (0, 726),
    'radiotap.dbm_antsignal': (-85, -24),
}


categories = {
    'wlan.fc.type': [0, 1, 2],
    'wlan.fc.subtype': list(range(16)),
    'wlan.fc.ds': list(range(4)),
}

label_mapping = {
    'normal' : 0,
    'flooding' : 1,
    'impersonation' : 2,
}

def to_int(val):
    if isinstance(val, int):
        return val
    try:
        return int(val, 0)
    except ValueError:
        return -1
    
def to_float(val):
    if isinstance(val, float):
        return val
    try:
        return float(val)
    except ValueError:
        return -1.0

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
    try:
        if 2412 <= freq <= 2472:
            return (freq - 2412) / (2472 - 2412)
        elif 5160 <= freq <= 5885:
            return (freq - 5160) / (5885 - 5160)
    except ValueError:
        return -1
    return -1

def process_file(file_path, preprocessor, is_fitted):
    df = pd.read_csv(file_path, names=awid2_cols, usecols=awid2_usecols)
    
    if 'flooding' not in df['class'].values and 'impersonation' not in df['class'].values:
        print(df['class'].unique())
        return 'none'
    
    df.rename(
        columns=awid2_rename,
        inplace=True,
    )

    for col in [
        'radiotap.present.tsft',
        'radiotap.channel.flags.cck',
        'radiotap.channel.flags.ofdm',
        'wlan.fc.frag',
        'wlan.fc.retry',
        'wlan.fc.pwrmgt',
        'wlan.fc.moredata',
        'wlan.fc.protected']:
        df[col] = df[col].apply(to_int)

    df['wlan.fc.ds'] = df['wlan.fc.ds'].apply(parse_ds)
    #df['radiotap.channel.freq'] = df['radiotap.channel.freq'].replace('?', np.nan)
    for col in ranges.keys():
        df[col] = df[col].replace('?', np.nan)
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

    for file in os.listdir(awid2_csv_path):
        target_csv = os.path.join(target_csv_dir, file + '.csv')
        if os.path.exists(target_csv):
            continue
        file_path = os.path.join(awid2_csv_path, file)
        transformed_df = process_file(file_path, preprocessor, is_transformer_fitted)
        if isinstance(transformed_df, str):
            print("no attacks in ", file)
            continue
        is_transformer_fitted = True

        transformed_df.to_csv(target_csv, index=False)

        print(f"Data saved to {target_csv}")


if __name__ == '__main__':
    process_data()
