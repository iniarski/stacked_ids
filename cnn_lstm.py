import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout
import random

model_path = 'saved_models/cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='accuracy',
    mode='max',
    verbose=1
)

def CNN_LSTM_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(128, 5, activation='relu', padding='same'),
        Dropout(0.2),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dropout(0.2),
        LSTM(256, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(256, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(240, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(120, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),

      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4
        )

        loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=0.07,
                gamma = 2
        )

        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)

    tfrecords_dir='data/AWID3_tfrecords'

    tfrecords = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')]
    epochs = 15

    random.shuffle(tfrecords)
    test_ratio = 0.8
    train_size = int(len(tfrecords) * test_ratio)
    train_data = tfrecords[:train_size]
    test_data = tfrecords[train_size:]


    print("Training set:")
    train_names = [t.split('/')[-1].split('.')[0] for t in train_data]
    train_names.sort()
    for t in train_names:
        print(t)

    print("\nTest set:")
    test_names = [t.split('/')[-1].split('.')[0] for t in test_data]
    test_names.sort()
    for t in test_names:
        print(t)

    sequence_length = 128
    sequence_shift = 8
    batch_size = 50

    import data_utils
    train_ds = data_utils.create_sequential_dataset(train_data, seq_length=sequence_length, seq_shift=sequence_shift, batch_size=batch_size)
    test_ds = data_utils.create_sequential_dataset(test_data, seq_length=sequence_length, seq_shift=sequence_shift, batch_size=batch_size)


    
    model = CNN_LSTM_model()
    model.fit(train_ds,
               epochs=epochs,
               callbacks=[checkpoint_callback],)

    model.summary()

    model.evaluate(test_ds)

if __name__ == '__main__':
    main()