import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv1D, Dropout, Reshape, BatchNormalization
import random


model_path = 'saved_models/multiclass_td_cnn.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def multiclass_time_domain_CNN_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(32, 1, activation='relu', padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(24, 3, activation='relu', padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(16, 5, activation='relu', padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        TimeDistributed(Dense(8, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'), name='td_dense'),
        Dropout(0.2),
        TimeDistributed(Dense(3, activation='softmax', kernel_regularizer='l2', bias_regularizer='l2'), name='td_output')
      ])

        loss = tf.keras.losses.CategoricalFocalCrossentropy(
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
    import data_utils
    #tf.config.run_functions_eagerly(True)

    tfrecords_dir='dataset/AWID3_tfrecords'
    balanced_tfrecords_dir='dataset/AWID3_tfrecords_balanced'
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    model = multiclass_time_domain_CNN_model()
    dataset_lambda = data_utils.create_multiclass_sequential_dataset

    if model.built:
        dataset_lambda = lambda x : data_utils.create_multiclass_sequential_dataset(x, shuffle=False, filter_out_normal=False)
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        epochs = 10
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
            epochs_per_step=2,
            n_initial_files=10,
            val_freq=2,
            increment=0.1,
            )
        model.summary()
        print(histories)
        data_utils.per_attack_test(model, test_files, dataset_lambda)

if __name__ == '__main__':
    main()
