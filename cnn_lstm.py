import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, Reshape

model_path = 'saved_models/cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='accuracy',
    mode='max',
    verbose=1
)

def CNN_LSTM_model(sequence_length = 64, n_features=40):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        model = tf.keras.models.Sequential([
        Reshape((-1, n_features)),
        Conv1D(60, 1, activation='relu'),
        Dropout(0.2),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(240, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),
        Reshape((sequence_length, )),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -3
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
                    )

        return model

def create_sequence_dataset(tfrecord_paths, sequence_length=2048, sequence_shift=1, batch_size=32):
    # Load the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)
    parsed_dataset = raw_dataset.map(data_utils.parse_record)


    # Separate features and labels
    features = parsed_dataset.map(lambda x, y: x)
    labels = parsed_dataset.map(lambda x, y: y)

    # Apply sliding window to accumulate sequences of timepoints
    feature_sequences = features.window(size=sequence_length, shift=sequence_shift, drop_remainder=True)
    label_sequences = labels.window(size=sequence_length, shift=sequence_shift, drop_remainder=True)

    # Flatten the windowed datasets and zip them together
    feature_sequences = feature_sequences.flat_map(lambda x: x.batch(sequence_length))
    label_sequences = label_sequences.flat_map(lambda x: x.batch(sequence_length))
    label_sequences = label_sequences.map(lambda y: 0 if tf.reduce_max(y, axis=-1) == 0 else 1)
    label_sequences = label_sequences.map(lambda y: tf.reshape(y, (1, )))

    # Combine features and labels into a single dataset and batch
    sequence_dataset = tf.data.Dataset.zip((feature_sequences, label_sequences))
    sequence_dataset = sequence_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return sequence_dataset

def main():
    labels = [0, 0]
    batch_size = 1000
    tfrecords_dir='data/AWID3_tfrecords'
    tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')]
    dataset = create_sequence_dataset(tfrecords_files, sequence_shift=512, batch_size=1000)
    i = 0
    for (X, y) in dataset:
        print(i)
        anomalies = tf.reduce_sum(y)
        labels[0] += batch_size - anomalies
        labels[1] += anomalies
        print(labels)
        i+=1
    model = CNN()
    model.fit(dataset, epochs=5)

if __name__ == '__main__':
    main()