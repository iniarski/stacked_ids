import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.regularizers import L2
import random

model_path = 'saved_models/multiclass_dnn.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def multiclass_DNN_model(n_features = 39):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        initializer = tf.keras.initializers.HeUniform()
        model = tf.keras.models.Sequential([
        Dense(60, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(40, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(24, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(12, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(3, activation='softmax', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
      ])

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 10 ** -2,   
            momentum = 0.9
        )

        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
                    )

        return model

def main():
    import data_utils
    random.seed(42)

    tfrecords_dir='dataset/AWID3_tfrecords_balanced'

    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio, shuffle=False)
    epochs = 15

    train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    batch_size = 100

    raw_train_ds = tf.data.TFRecordDataset(train_set)
    train_ds = raw_train_ds.map(data_utils.parse_multiclass_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    raw_test_ds = tf.data.TFRecordDataset(test_set)
    test_ds = raw_test_ds.map(data_utils.parse_multiclass_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    


    model = multiclass_DNN_model()
    model.fit(train_ds,
              validation_data = test_ds,
              validation_freq = 3,
              epochs=epochs,
              callbacks = [checkpoint_callback],)

    model.summary()


if __name__ == '__main__':
    main()