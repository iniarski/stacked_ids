import tensorflow as tf
import os
import data_utils
from keras.layers import Dense, Dropout, BatchNormalization, Reshape
from keras.regularizers import L2
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
        Dropout(0.3),
        Dense(40, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(24, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(12, activation='relu', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax', kernel_initializer=initializer, kernel_regularizer=L2(0.04)),
      ])

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = 10 ** -2,   
            momentum = 0.9
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

    model = multiclass_DNN_model()
    dataset_lambda = lambda x : data_utils.create_multiclass_dataset(x)

    if model.built:
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 20
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, validation_files = data_utils.train_test_split(train_files, train_ratio, repeat_rare=True)
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
        data_utils.per_attack_test(model, dataset_lambda)

if __name__ == '__main__':
    main()
