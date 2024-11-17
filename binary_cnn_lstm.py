import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import data_utils
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, Dropout
import random


model_path = 'saved_models/binary_cnn_lstm.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=False,
    monitor='recall',
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
        Conv1D(60, 5, activation='relu', padding='same'),
        Dropout(0.2),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        LSTM(128, activation='tanh', return_sequences=True),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(240, activation='relu')),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Dense(1, activation='sigmoid')),
      ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -4
        )

        loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=0.07,
                gamma = 2
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
    train_ratio = 0.8
    tfrecords_files = os.listdir(tfrecords_dir)
    train_files, test_files, = data_utils.train_test_split(tfrecords_files, train_ratio)
    
    train_files = list(filter(lambda f : not f.startswith('Kr00k'), train_files))
    test_files = list(filter(lambda f : not f.startswith('Kr00k'), test_files))

    epochs = 35
    
    model = binary_CNN_LSTM_model()

    train_set = [os.path.join(tfrecords_dir, file) for file in train_files]
    test_set = [os.path.join(tfrecords_dir, file) for file in test_files]

    train_ds = data_utils.create_binary_sequential_dataset(train_set)
    test_ds = data_utils.create_binary_sequential_dataset(test_set, shuffle=False)
    
    model.fit(
    train_ds,              
    validation_data = test_ds,
    epochs=epochs,
    callbacks=[checkpoint_callback],
    class_weight = {
        0 : 0.2,
        1 : 0.8,
        },
    )
    
    model.summary()
    

if __name__ == '__main__':
    main()
