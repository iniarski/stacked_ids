import data_utils
import os
import tensorflow as tf

labels = [0, 0]

tfrecords_dir='data/AWID3_tfrecords'
tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecord')]
dataset = data_utils.create_sequential_dataset(tfrecords_files)
i = 0
for (X, y) in dataset:
    print(i)
    print(y.shape)
    anomalies = tf.reduce_sum(y)
    labels[0] += 64*32 - anomalies
    labels[1] += anomalies
    print(labels)
    i+=1