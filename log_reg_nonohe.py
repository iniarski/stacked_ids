import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
import random

model_path = 'saved_models/log_reg_nonohe.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuaracy',
    mode='max',
    verbose=1
)

def LogReg_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Dense(1, activation='sigmoid')
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -3,
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score']
                    )

        return model

def main():
    import data_utils

    tfrecords_dir='dataset/AWID3_tfrecords_balanced'
    train_ratio = 0.8
    val_ratio = 0.75
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    model = LogReg_model()
    dataset_lambda = lambda x : data_utils.create_binary_dataset_nonohe(x)

    if model.built:
        data_utils.per_attack_test(model, dataset_lambda)

    else :
        epochs = 10
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, val_files = data_utils.train_test_split(train_files, val_ratio, repeat_rare=False)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        val_files = [os.path.join(tfrecords_dir, f) for f in val_files]
        train_ds = dataset_lambda(train_files)
        val_ds = dataset_lambda(val_files)

        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = epochs,
            callbacks = [checkpoint_callback]
        )

        model.summary()
        print(history.history)
        data_utils.per_attack_test(model, dataset_lambda)

if __name__ == '__main__':
    main()
