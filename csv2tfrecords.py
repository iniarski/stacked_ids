import tensorflow as tf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

tf.config.experimental.set_visible_devices([], 'GPU')

csv_dir = 'data/AWID3_preprocessed'
tfrecord_dir = 'data/AWID3_tfrecords'
os.makedirs(tfrecord_dir, exist_ok=True)

def create_example(row, column_names):
    feature = {
        col_name: tf.train.Feature(float_list=tf.train.FloatList(value=[val])) 
        for col_name, val in zip(column_names, row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_csv_to_tfrecord(csv_file, tfrecord_file):
    df = pd.read_csv(csv_file)
    column_names = list(map(lambda c : str(c).split("__")[-1], df.columns))
    print(csv_file, column_names)
    
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    
    # Open TFRecord writer
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        # Use `map` to convert each batch of rows to TFRecord Examples and write them
        for batch in dataset.batch(1024):  # Adjust batch size as needed
            for row in batch.numpy():
                example = create_example(row, column_names)
                writer.write(example.SerializeToString())

def process_file(file):
    if file.endswith('.csv'):
        csv_path = os.path.join(csv_dir, file)
        tfrecord_path = os.path.join(tfrecord_dir, f"{os.path.splitext(file)[0]}.tfrecord")
        convert_csv_to_tfrecord(csv_path, tfrecord_path)

with ProcessPoolExecutor() as executor:
    executor.map(process_file, os.listdir(csv_dir))