import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Dropout, Reshape
import random


model_path = 'saved_models/binary_cnn_gru.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def binary_CNN_GRU_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(48, 1, activation='relu', padding='same'),
        Dropout(0.2),
        Conv1D(32, 1, activation='relu', padding='same'),
        Dropout(0.2),
        GRU(24, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(12 , activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])
        
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -3
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

    model = binary_CNN_GRU_model()
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
        
        balanced_train_ds = dataset_lambda(balanced_train_files)
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
            epochs_per_step=5,
            n_initial_files=7,
            val_freq=5,
            increment=0.2,
            )
        model.summary()
        print(histories)
        data_utils.per_attack_test(model, test_files, dataset_lambda)

if __name__ == '__main__':
    main()
