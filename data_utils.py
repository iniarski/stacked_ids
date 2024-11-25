import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import math

features = [
    'frame.len',
    'radiotap.length',
    'frame.time_delta',
    'wlan.duration',
    'radiotap.dbm_antsignal',
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
    'wlan.fc.frag', 'wlan.fc.retry',
    'wlan.fc.pwrmgt',
    'wlan.fc.moredata',
    'wlan.fc.protected',
    '2ghz_spectrum',
    '5ghz_spectrum',
    'freq',
    'Label']


feature_description = {feat: tf.io.FixedLenFeature([], tf.float32) for feat in features}

awid3_attacks = [
    'Deauth',
    'Disas',
    '(Re)Assoc',
    'RogueAP',
    'Krack',
    'Kr00k',
    'Evil_Twin'
]

def train_test_split(file_names : list[str], train_ratio : float = 0.8, shuffle : bool = True, repeat_rare : bool = False) -> (list[str], list[str]):
    test_set = []
    train_set = []

    # sorting files an using random seed for consistent splitting
    random.seed(42)
    file_names.sort()

    for attack_name in awid3_attacks:
        attack_files = list(filter(lambda name: name.startswith(attack_name) and name.endswith('.tfrecords'), file_names))
        if len(attack_files) == 0:
            print(f'No files for {attack_name}')
            continue
        n_test = max(1, int(len(attack_files) * (1 - train_ratio)))
        if shuffle:
            random.shuffle(attack_files)
        test_files = attack_files[:n_test]
        train_files = attack_files[n_test:]
        if attack_name == 'RogueAP' and repeat_rare:
            train_files = 3 * train_files
        if attack_name == 'Krack' and repeat_rare:
            train_files = 3 * train_files
        for f in test_files:
            test_set.append(f)
        for f in train_files:
            train_set.append(f)

    if shuffle:
        random.shuffle(test_set)
        random.shuffle(train_set)

    return train_set, test_set


def parse_binary_record(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    label = parsed_features.pop('Label')
    label = 0 if label == 0 else 1
    features = tf.stack(list(parsed_features.values()))
    return features, tf.reshape(label,(1,))

def parse_multiclass_record(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    label = parsed_features.pop('Label')
    label = tf.cast(label, tf.int8)
    label = tf.one_hot(label, depth=3)
    features = tf.stack(list(parsed_features.values()))
    return features, label

@tf.function
def binary_sequence_has_attack(x, y):
    return tf.math.not_equal(tf.reduce_max(y), 0)

@tf.function
def multiclass_sequence_has_attack(x, y):
    reduced  = tf.math.equal(tf.reduce_max(y, axis=0), tf.ones((3, )))
    return tf.math.logical_or(reduced[1], reduced[2])

def create_binary_sequential_dataset(
        tfrecords_files : list[str],
        seq_length : int = 32,
        seq_shift : int = 30,
        batch : bool = True,
        batch_size : int = 32,
        filter_out_normal : bool = True,
        shuffle : bool = True,
        shuffle_buffer : int = 2048
        ) -> tf.data.Dataset :

    raw_dataset = tf.data.TFRecordDataset(tfrecords_files, num_parallel_reads=tf.data.AUTOTUNE)
    parsed_dataset = raw_dataset.map(parse_binary_record)

    features = parsed_dataset.map(lambda x, y: x)
    labels = parsed_dataset.map(lambda x, y: y)

    feature_sequences = features.window(size=seq_length, shift=seq_shift, drop_remainder=True)
    label_sequences = labels.window(size=seq_length, shift=seq_shift, drop_remainder=True)

    feature_sequences = feature_sequences.flat_map(lambda x: x.batch(seq_length))
    label_sequences = label_sequences.flat_map(lambda x: x.batch(seq_length))

    sequence_dataset = tf.data.Dataset.zip((feature_sequences, label_sequences))
    if filter_out_normal:
        sequence_dataset = sequence_dataset.filter(binary_sequence_has_attack)
    if shuffle:
        sequence_dataset = sequence_dataset.shuffle(shuffle_buffer)
    if batch:
        sequence_dataset = sequence_dataset.batch(batch_size)
    sequence_dataset = sequence_dataset.prefetch(tf.data.AUTOTUNE)

    return sequence_dataset

def create_multiclass_sequential_dataset(
        tfrecords_files : list[str],
        seq_length : int = 32,
        seq_shift : int = 30,
        batch : bool = True,
        batch_size : int = 32,
        filter_out_normal : bool = True,
        shuffle : bool = True,
        shuffle_buffer : int = 2048,
        ) -> tf.data.Dataset :

    raw_dataset = tf.data.TFRecordDataset(tfrecords_files, num_parallel_reads=tf.data.AUTOTUNE)
    parsed_dataset = raw_dataset.map(parse_multiclass_record)

    features = parsed_dataset.map(lambda x, y: x)
    labels = parsed_dataset.map(lambda x, y: y)

    feature_sequences = features.window(size=seq_length, shift=seq_shift, drop_remainder=True)
    label_sequences = labels.window(size=seq_length, shift=seq_shift, drop_remainder=True)

    feature_sequences = feature_sequences.flat_map(lambda x: x.batch(seq_length))
    label_sequences = label_sequences.flat_map(lambda x: x.batch(seq_length))

    sequence_dataset = tf.data.Dataset.zip((feature_sequences, label_sequences))
    if filter_out_normal:
        sequence_dataset = sequence_dataset.filter(multiclass_sequence_has_attack)
    if shuffle:
        sequence_dataset = sequence_dataset.shuffle(shuffle_buffer)
    if batch:
        sequence_dataset = sequence_dataset.batch(batch_size)
    sequence_dataset = sequence_dataset.prefetch(tf.data.AUTOTUNE)

    return sequence_dataset


def create_binary_dataset(
        tfrecords_files : list[int],
        batch_size : int = 50,
        shuffle : bool = True,
        shuffle_buffer : int = 10000
        ) -> tf.data.Dataset:

    raw_ds = tf.data.TFRecordDataset(tfrecords_files)
    ds = raw_ds.map(parse_binary_record)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_multiclass_dataset(
        tfrecords_files : list[int],
        batch_size : int = 50,
        shuffle : bool = True,
        shuffle_buffer : int = 10000
        ) -> tf.data.Dataset:

    raw_ds = tf.data.TFRecordDataset(tfrecords_files)
    ds = raw_ds.map(parse_multiclass_record)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def step_training(
        train_files : list[str],
        test_files : list[str],
        model,
        dataset_callback,
        epochs_per_step : int = 5,
        val_freq : int = 1,
        training_callbacks = None,
        n_initial_files : int = 15,
        increment : float = 0.1
    ):
    test_ds = dataset_callback(test_files)
    n_files = n_initial_files

    current_train_files = []
    histories = []

    while len(current_train_files) < len(train_files):
        current_train_files = train_files[:min(len(train_files), n_files)]
        random.shuffle(current_train_files)
        for file in current_train_files:
            print(file.split('/')[-1].split('.')[0], end=',')
        print()
        n_files += math.ceil(n_files * increment)
        train_ds = dataset_callback(current_train_files)
        history = model.fit(
            train_ds,
            validation_data = test_ds,
            epochs = epochs_per_step,
            validation_freq = val_freq,
            callbacks = training_callbacks,
        )
        random.shuffle(train_files)
        histories.append(history.history)
    return histories



def per_attack_test(model, dataset_callback, train_ratio=0.8, tfrecords_dir='dataset/AWID3_tfrecords'):
    _, all_test_files = train_test_split(os.listdir(tfrecords_dir), train_ratio=train_ratio)


    per_attack_files = [list(filter(lambda f : f.startswith(attack), all_test_files)) for attack in awid3_attacks]
    for files in per_attack_files:
        test_set = [os.path.join(tfrecords_dir, f) for f in files]
        for file in files:
            print(file.split('.')[0], end=' , ')
        print()
        ds = dataset_callback(test_set)
        model.evaluate(ds)
