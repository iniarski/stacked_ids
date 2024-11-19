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
        Conv1D(64, 1, activation='relu', padding='same'),
        Dropout(0.2),
        Conv1D(32, 1, activation='relu', padding='same'),
        Dropout(0.2),
        GRU(48, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])
        
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.8,
            gamma=1.8
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4
        )


        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    random.seed(42)
    import data_utils


def main():
    random.seed(42)
    import data_utils

    tfrecords_dir='dataset/AWID3_tfrecords'
    train_ratio = 0.8


    model = binary_CNN_GRU_model()
    dataset_lambda = lambda x : data_utils.create_binary_sequential_dataset(x, seq_length=64, seq_shift=60)

    if model.built:
        dataset_lambda = lambda x : data_utils.create_binary_sequential_dataset(x, shuffle=False, filter_out_normal=False)
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        dataset_lambda = lambda x : data_utils.create_binary_sequential_dataset(x, seq_length=64, seq_shift=60, filter_out_normal=False)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        test_files = [os.path.join(tfrecords_dir, f) for f in test_files]

        histories = data_utils.step_training(
            train_files=train_files,
            test_files=test_files,
            model=model,
            dataset_callback=dataset_lambda,
            training_callbacks=[checkpoint_callback],
        )

        print(histories)


if __name__ == '__main__':
    main()
