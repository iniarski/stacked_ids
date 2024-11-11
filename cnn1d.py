import tensorflow as tf
import os


model_path = 'saved_models/cnn1d.keras'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=True,
    monitor='accuracy',
    mode='max',
    verbose=1
)

def CNN1D():
    if os.path.exists(model_path):
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(64, 1, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv1D(32, 1, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = 10 ** -7
        )

        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                    )

        return model