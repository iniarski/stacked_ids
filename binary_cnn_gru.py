import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Dropout
import random


model_path = 'saved_models/binary_cnn_gru.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=False,
    monitor='recall',
    mode='max',
    verbose=1
)

def binary_CNN_GRU_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(60, 5, activation='relu', padding='same'),
        Dropout(0.2),
        GRU(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        GRU(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(240, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4
        )


        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)
    import data_utils


    tfrecords_dir='dataset/AWID3_tfrecords'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    train_files = list(filter(lambda f : not f.startswith('Kr00k'), train_files))
    test_files = list(filter(lambda f : not f.startswith('Kr00k'), test_files))

    epochs = 20
    batch_size = 100

    model = binary_CNN_GRU_model()

    train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    train_ds = data_utils.create_binary_sequential_dataset(train_set, batch_size=batch_size)
    test_ds = data_utils.create_binary_sequential_dataset(test_set,batch_size=batch_size, shuffle=False)

    model.fit(
    train_ds,
    validation_data = test_ds,
    epochs=epochs,
    callbacks=[checkpoint_callback]
    )

    model.summary()


if __name__ == '__main__':
    main()
