import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
import random

model_path = 'saved_models/multiclass_cnn1d.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='f1_score',
    mode='max',
    verbose=1
)

def multiclass_CNN1D_model(n_features = 39):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Reshape((1, n_features)),
        Conv1D(256, 1, activation='relu', padding='same', strides=1),
        Dropout(0.5),
        Conv1D(128, 1, activation='relu', padding='same', strides=1),
        Dropout(0.5),
        Conv1D(64, 1, activation='relu', padding='same', strides=1),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu', kernel_regularizer=L2(0.01)),
        Dropout(0.5),
        Dense(3, activation='softmax')
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -7,   
        )

        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
                    )

        return model

def main():
    import data_utils
    data_utils.awid3_attacks = [
    'Deauth',
    #'Disas',
    #'(Re)Assoc',
    #'RogueAP',
    #'Krack',
    #'Kr00k',
    #'Evil_Twin'
    ]
    random.seed(42)

    tfrecords_dir='dataset/AWID3_tfrecords_balanced'

    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio, shuffle=False)
    epochs = 15

    train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    batch_size = 200

    raw_train_ds = tf.data.TFRecordDataset(train_set)
    train_ds = raw_train_ds.map(data_utils.parse_multiclass_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    raw_test_ds = tf.data.TFRecordDataset(test_set)
    test_ds = raw_test_ds.map(data_utils.parse_multiclass_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    


    model = multiclass_CNN1D_model()
    model.fit(train_ds,
              validation_data=test_ds,
              epochs=epochs,
              callbacks = [checkpoint_callback],)

    model.summary()


if __name__ == '__main__':
    main()