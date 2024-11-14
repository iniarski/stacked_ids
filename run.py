import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random

random.seed(2024)
np.random.seed(2024)

tfrecords_dir='data/AWID3_tfrecords'

tfrecords = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')]

random.shuffle(tfrecords)
test_ratio = 0.8
train_size = int(len(tfrecords) * test_ratio)
train_data = tfrecords[:train_size]
test_data = tfrecords[train_size:]

class_weight = {
    0 : 0.1,
    1 : 1,
}

print("Training set:")
train_names = [t.split('/')[-1].split('.')[0] for t in train_data]
train_names.sort()
for t in train_names:
    print(t)

print("\nTest set:")
test_names = [t.split('/')[-1].split('.')[0] for t in test_data]
test_names.sort()
for t in test_names:
    print(t)

import dnn
import data_utils

batch_size = 32


model = dnn.DNN()
train_ds = tf.data.TFRecordDataset(train_data)
train_ds = train_ds.map(data_utils.parse_record)

model.fit(
        train_ds,
        epochs=15,
        batch_size=batch_size,
        callbacks = [dnn.checkpoint_callback],
        shuffle=True,
    )

