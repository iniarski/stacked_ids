import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, MaxPooling1D, Bidirectional, Reshape
import random


model_path = 'saved_models/binary_cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_recall',
    mode='max',
    verbose=1
)

def binary_CNN_LSTM_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(128, 5, activation='relu', padding='same'),
        Dropout(0.2),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4.5
        )

        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.2,
            gamma=2
        )

        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)
    import data_utils


    tfrecords_dir='dataset/AWID3_tfrecords'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    full_train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    epochs = 4
    batch_size = 16
    histories = []
    model = binary_CNN_LSTM_model()
    
    if model.built:
        data_utils.evaluate_for_attack_types(model, data_utils.create_binary_sequential_dataset)
        return

    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    test_ds = data_utils.create_binary_sequential_dataset(test_set,batch_size=batch_size, shuffle=False, filter_out_normal=False)
    train_ds = data_utils.create_binary_sequential_dataset(train_set, seq_length=20, seq_shift=16, batch_size=batch_size)

    
    n_files = 4
    increment = 1

    while n_files < len(full_train_files):
        train_files = full_train_files[:min(n_files, len(full_train_files))]
        random.shuffle(train_files)
        train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
        train_ds = data_utils.create_binary_sequential_dataset(train_set, seq_length=20, seq_shift=16, batch_size=batch_size)
        n_files+=increment
        increment+=1
        history = model.fit(
        train_ds,
        validation_data = test_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        validation_freq=epochs // 2
        )

        histories.append(history.history)

    model.summary()

    print(histories)


if __name__ == '__main__':
    main()
