import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Dropout, Reshape
import random


model_path = 'saved_models/multiclass_cnn_gru.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def multiclass_CNN_GRU_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(96, 1, activation='relu', padding='same'),
        Dropout(0.15),
        Conv1D(64, 1, activation='relu', padding='same'),
        Dropout(0.15),
        GRU(48, activation='tanh', return_sequences=True),
        Dropout(0.15),
        TimeDistributed(
            Dense(32, activation='relu'),
            name='td_dense'),
        Dropout(0.15),
        TimeDistributed(
            Dense(3, activation='softmax'),
            name='td_output'),
      ])


        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=2
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -3,
        )


        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model

def main():
    import data_utils

    tfrecords_dir='dataset/AWID3_tfrecords'
    balanced_tfrecords_dir='dataset/AWID3_tfrecords_balanced'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    model = multiclass_CNN_GRU_model()
    dataset_lambda = lambda x: data_utils.create_multiclass_sequential_dataset(x, batch_size=4, shuffle_buffer=4096)

    if model.built:
        dataset_lambda = lambda x : data_utils.create_multiclass_sequential_dataset(x, shuffle=False, filter_out_normal=False)
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 7
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files1, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, validation_files = data_utils.train_test_split(train_files1, train_ratio)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        validation_files = [os.path.join(tfrecords_dir, f) for f in validation_files]
        
        train_files1, validation_files1 = data_utils.train_test_split(train_files1, train_ratio, repeat_rare=True)
        balanced_validation_files = [os.path.join(balanced_tfrecords_dir, f) for f in validation_files1]
        balanced_train_files = [os.path.join(balanced_tfrecords_dir, f) for f in train_files1]
        balanced_train_ds = dataset_lambda(balanced_train_files)
        balanced_val_ds = dataset_lambda(balanced_validation_files)

        model.fit(
            balanced_train_ds,
            validation_data = balanced_val_ds,
            epochs=epochs,
            validation_freq=epochs
        )

        histories = data_utils.step_training(
            train_files, 
            validation_files, 
            model, 
            dataset_lambda, 
            training_callbacks=[checkpoint_callback],
            epochs_per_step=1,
            n_initial_files=7,
            val_freq=1,
            increment=0.1,
            )
        model.summary()
        print(histories)
        data_utils.per_attack_test(model, test_files, dataset_lambda)

if __name__ == '__main__':
    main()
