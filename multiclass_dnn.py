import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape
import random

model_path = 'saved_models/multiclass_dnn.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='recall',
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
        Dense(3, activation='softmax', kernel_initializer=initializer),
      ])

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.01,   
            momentum = 0.9
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
    'Disas',
    #'(Re)Assoc',
    #'RogueAP',
    #'Krack',
    #'Kr00k',
    #'Evil_Twin'
    ]
    random.seed(42)

    tfrecords_dir='dataset/AWID3_tfrecords'

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
    


    model = multiclass_DNN_model()
    model.fit(train_ds,
              epochs=epochs,
              callbacks = [checkpoint_callback],)

    model.summary()

    model.evaluate(test_ds)

if __name__ == '__main__':
    main()