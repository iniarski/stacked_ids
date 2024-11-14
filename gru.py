import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Dropout, Reshape

model_path = 'saved_models/gru.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='accuracy',
    mode='max',
    verbose=1
)

def GRU_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        GRU(128, activation='tanh', return_sequences=True),
        Dropout(0.2),
        GRU(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -5
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model


def main():
    tfrecords_dir='data/AWID3_tfrecords'
    tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')] 
    dataset = data_utils.create_sequential_dataset(tfrecords_files)
    model = GRU_model()
    model.fit(dataset, epochs=15)

if __name__ == '__main__':
    main()