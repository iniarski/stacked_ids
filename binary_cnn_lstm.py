import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout, MaxPooling1D, Bidirectional, Reshape, BatchNormalization
from keras.regularizers import L2
import random


model_path = 'saved_models/binary_cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def binary_CNN_LSTM_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(128, 1, activation='relu', padding='same'),
        Dropout(0.2),
        LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=L2(0.01)),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(
            Dense(32, activation='relu', kernel_regularizer=L2(0.01)),
            name='td_dense'),
        Dropout(0.2),
        TimeDistributed(
            Dense(1, activation='sigmoid', kernel_regularizer=L2(0.01)),
            name='td_output'),
      ])
        
        
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate = 10 ** -4
        )

        loss = tf.keras.losses.BinaryCrossentropy(
            #apply_class_balancing = True,
            #alpha = 0.85,
            #gamma = 2
        )

        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)
    import data_utils

    tfrecords_dir='dataset/AWID3_tfrecords'
    balanced_tfrecords_dir='dataset/AWID3_tfrecords_balanced'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    model = binary_CNN_LSTM_model()
    dataset_lambda = data_utils.create_binary_sequential_dataset

    if model.built:
        dataset_lambda = lambda x : data_utils.create_binary_sequential_dataset(x, shuffle=False, filter_out_normal=False)
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 3
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, validation_files = data_utils.train_test_split(train_files, train_ratio, repeat_rare=True)
        balanced_train_files = [os.path.join(balanced_tfrecords_dir, f) for f in train_files]
        balanced_validation_files = [os.path.join(balanced_tfrecords_dir, f) for f in validation_files]
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        validation_files = [os.path.join(tfrecords_dir, f) for f in validation_files]
        
        balanced_train_ds = dataset_lambda(balanced_train_files, seq_length=16, seq_shift=12)
        balanced_val_ds = dataset_lambda(balanced_validation_files)

        model.fit(
            balanced_train_ds,
            validation_data = balanced_val_ds,
            epochs=epochs
        )

        histories = data_utils.step_training(
            train_files, 
            validation_files, 
            model, 
            dataset_lambda, 
            training_callbacks=[checkpoint_callback],
            epochs_per_step=2,
            n_initial_files=6,
            val_freq=2,
            increment=0.25,
            )
        model.summary()
        print(histories)
        data_utils.per_attack_test(model, test_files, dataset_lambda)

if __name__ == '__main__':
    main()
