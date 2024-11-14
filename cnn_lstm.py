import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input

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
        Conv1D(60, 1, activation='relu'),
        Dropout(0.2),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(240, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),

      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -5
        )

        model.compile(optimizer=optimizer,
                    loss='binary_focal_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    tfrecords_dir='data/AWID3_tfrecords'
    tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')] 
    dataset = data_utils.create_sequential_dataset(tfrecords_files)
    model = CNN_LSTM_model()
    model.fit(dataset, epochs=15)

if __name__ == '__main__':
    main()