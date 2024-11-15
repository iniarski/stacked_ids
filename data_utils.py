import tensorflow as tf

features = [
    'frame.len', 'radiotap.length', 'frame.time_delta', 'wlan.duration', 'parsed_antsignal',
    'radiotap.channel.freq_2417', 'radiotap.channel.freq_2472', 'radiotap.channel.freq_5180',
    'wlan.fc.type_0', 'wlan.fc.type_1', 'wlan.fc.type_2', 'wlan.fc.subtype_0', 'wlan.fc.subtype_1',
    'wlan.fc.subtype_2', 'wlan.fc.subtype_3', 'wlan.fc.subtype_4', 'wlan.fc.subtype_5',
    'wlan.fc.subtype_6', 'wlan.fc.subtype_7', 'wlan.fc.subtype_8', 'wlan.fc.subtype_9',
    'wlan.fc.subtype_10', 'wlan.fc.subtype_11', 'wlan.fc.subtype_12', 'wlan.fc.subtype_13',
    'wlan.fc.subtype_14', 'wlan.fc.subtype_15', 'wlan.fc.ds_0x00000000', 'wlan.fc.ds_0x00000001',
    'wlan.fc.ds_0x00000002', 'wlan.fc.ds_0x00000003', 'radiotap.present.tsft_0-0-0',
    'radiotap.present.tsft_1-0-0', 'radiotap.channel.flags.cck', 'radiotap.channel.flags.ofdm',
    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
    'Label'
]

feature_description = {feat: tf.io.FixedLenFeature([], tf.float32) for feat in features}

def parse_record(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    label = parsed_features.pop('Label')
    label = 0 if label == 0 else 1
    features = tf.stack(list(parsed_features.values()))
    return features, tf.reshape(label,(1,))

def sequence_has_attack(x, y):
    return tf.math.not_equal(tf.reduce_max(y), 0)

def create_sequential_dataset(tfrecords_files, seq_length = 64, seq_shift = 56, batch_size = 32, filter_out_normal = True, shuffle=True):
    raw_dataset = tf.data.TFRecordDataset(tfrecords_files)
    parsed_dataset = raw_dataset.map(parse_record)

    features = parsed_dataset.map(lambda x, y: x)
    labels = parsed_dataset.map(lambda x, y: y)

    feature_sequences = features.window(size=seq_length, shift=seq_shift, drop_remainder=True)
    label_sequences = labels.window(size=seq_length, shift=seq_shift, drop_remainder=True)

    feature_sequences = feature_sequences.flat_map(lambda x: x.batch(seq_length))
    label_sequences = label_sequences.flat_map(lambda x: x.batch(seq_length))

    sequence_dataset = tf.data.Dataset.zip((feature_sequences, label_sequences))
    if filter_out_normal:
        sequence_dataset = sequence_dataset.filter(sequence_has_attack)
    if shuffle:
        sequence_dataset = sequence_dataset.shuffle(100000)
    sequence_dataset = sequence_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return sequence_dataset