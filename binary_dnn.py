import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization
import random

model_path = 'saved_models/binary_dnn.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='recall',
    mode='max',
    verbose=1
)

def binary_DNN_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        initializer = tf.keras.initializers.HeUniform()
        model = tf.keras.models.Sequential([
        Dense(30, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.25),
        Dense(20, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.25),
        Dense(16, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.25),
        Dense(12, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.25),
        Dense(6, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.25),
        Dense(1, activation='sigmoid', kernel_initializer=initializer)
      ])

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.01,   
            momentum = 0.9
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
                    )

        return model

def main():
    import data_utils
    random.seed(42)



    tfrecords_dir='dataset/AWID3_tfrecords'

    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files = data_utils.train_test_split(tfrecords_files, train_ratio)
    train_files = list(filter(lambda f : not f.startswith('Kr00k'), train_files))
    test_files = list(filter(lambda f : not f.startswith('Kr00k'), test_files))
    epochs = 25

    train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    batch_size = 50

    raw_train_ds = tf.data.TFRecordDataset(train_set)
    train_ds = raw_train_ds.map(data_utils.parse_binary_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    raw_test_ds = tf.data.TFRecordDataset(test_set)
    test_ds = raw_test_ds.map(data_utils.parse_binary_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)



    model = binary_DNN_model()
    model.fit(train_ds,
              validation_data=test_ds,
              epochs=epochs,
              callbacks = [checkpoint_callback],)

    model.summary()


if __name__ == '__main__':
    main()