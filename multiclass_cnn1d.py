import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import Dense, TimeDistributed, Conv1D, Dropout, Reshape, Input, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
import random

model_path = 'saved_models/multiclass_cnn1d.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
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
            learning_rate = 10 ** -6,
        )

        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
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

    model = multiclass_CNN1D_model()
    dataset_lambda = lambda x : data_utils.create_multiclass_dataset(x)

    if model.built:
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 20
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, validation_files = data_utils.train_test_split(train_files, train_ratio)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        val_files = [os.path.join(tfrecords_dir, f) for f in validation_files]
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
        data_utils.per_attack_test(test_files, dataset_lambda)

if __name__ == '__main__':
    main()
