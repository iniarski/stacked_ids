import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization
import random

model_path = 'saved_models/dnn.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='f1_score',
    mode='max',
    verbose=1
)

def DNN_model():
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
                    loss='binary_focal_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
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

    sequence_length = 64
    sequence_shift = 56
    batch_size = 200

    import data_utils
    raw_train_ds = tf.data.TFRecordDataset(train_data)
    train_ds = raw_train_ds.map(data_utils.parse_record).shuffle(100000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    raw_test_ds = tf.data.TFRecordDataset(test_data)
    test_ds = raw_train_ds.map(data_utils.parse_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    class_weight = {
        0 : 1 / 0.92972 / 2.0,
        1 : 1 / (1 - 0.92972) / 2.0
    }

    model = DNN_model()
    model.fit(train_ds,
              epochs=epochs,
              callbacks = [checkpoint_callback],
              class_weight=class_weight)

    model.summary()

    model.evaluate(test_ds)

if __name__ == '__main__':
    main()