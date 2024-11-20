import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout
import random


model_path = 'saved_models/multiclass_cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

def multiclass_CNN_LSTM_model():
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        Conv1D(128, 5, activation='relu', padding='same'),
        Dropout(0.2),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(3, activation='softmax')),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4
        )
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.2,
            gamma=2
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
    train_ratio = 0.01
    tfrecords_files = os.listdir(tfrecords_dir)
    full_train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)

    epochs = 6
    batch_size = 16
    histories = []
    model = multiclass_CNN_LSTM_model()
    
    if model.built:
        dataset_lambda = lambda x : data_utils.create_multiclass_sequential_dataset(x, shuffle=False, filter_out_normal=False)
        data_utils.per_attack_test(model, dataset_lambda)
    else :
        tfrecords_files = os.listdir(tfrecords_dir)
        train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
        train_files, validation_files = data_utils.train_test_split(train_files, train_ratio)
        train_files = [os.path.join(tfrecords_dir, f) for f in train_files]
        validation_files = [os.path.join(tfrecords_dir, f) for f in validation_files]
        histories = data_utils.step_training(
            train_files, 
            validation_files, 
            model, 
            dataset_lambda, 
            training_callbacks=[checkpoint_callback],
            epochs_per_step=5,
            n_initial_files=2,
            )
        model.summary()
        print(histories)
        data_utils.per_attack_test(test_files, dataset_lambda)

if __name__ == '__main__':
    main()
