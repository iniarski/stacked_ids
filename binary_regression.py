import tensorflow as tf
import os
import data_utils
from tensorflow.keras.layers import Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
import random

model_path = 'saved_models/binary_regression.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def binary_regression_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Dense(1, activation='sigmoid')
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4,   
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)
    import data_utils

    tfrecords_dir='dataset/AWID3_tfrecords_balanced'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    model = binary_regression_model()
    dataset_lambda = lambda x : data_utils.create_binary_dataset(x)

    if not model.built:
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 10
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, val_files = data_utils.train_test_split(train_files, train_ratio)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        val_files = [os.path.join(tfrecords_dir, f) for f in val_files]
        train_files = [os.path.join('dataset/AWID2_tfrecords/train', f) for f in os.listdir('dataset/AWID2_tfrecords/train')]
        val_files = [os.path.join('dataset/AWID2_tfrecords/test', f) for f in os.listdir('dataset/AWID2_tfrecords/test')]
        for f in val_files:
            try:
                train_ds = dataset_lambda([f])
                
                history = model.fit(
                    train_ds,
                    epochs = 1,
                )
            except Exception:
                print(f)
        
        model.summary()
        print(history.history)
        data_utils.per_attack_test(model, dataset_lambda)

if __name__ == '__main__':
    main()