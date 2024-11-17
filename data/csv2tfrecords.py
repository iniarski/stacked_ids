import tensorflow as tf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

tf.config.experimental.set_visible_devices([], 'GPU')

csv_dir = 'dataset/AWID3_CSV_preprocessed'
tfrecord_dir = 'dataset/AWID3_tfrecords'
os.makedirs(tfrecord_dir, exist_ok=True)
column_names = []

def create_example(row, column_names):
    feature = {
        col_name: tf.train.Feature(float_list=tf.train.FloatList(value=[val])) 
        for col_name, val in zip(column_names, row)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_column_name(c):
    c = str(c).split("__")[-1]
    if c.endswith('.0'):
        return c[:-2]
    return c

def convert_csv_to_tfrecord(csv_file, tfrecord_file):
    df = pd.read_csv(csv_file)
    column_names = list(map(parse_column_name, df.columns))
    print(csv_file, column_names)
    
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for batch in dataset.batch(1024):
            for row in batch.numpy():
                example = create_example(row, column_names)
                writer.write(example.SerializeToString())

def process_file(file):
    if file.endswith('.csv'):
        csv_path = os.path.join(csv_dir, file)
        tfrecord_path = os.path.join(tfrecord_dir, f"{os.path.splitext(file)[0]}.tfrecords")
        if os.path.exists(tfrecord_path):
            return
        convert_csv_to_tfrecord(csv_path, tfrecord_path)


#for file in os.listdir(csv_dir):
#    process_file(file)
with ProcessPoolExecutor() as executor:
    executor.map(process_file, os.listdir(csv_dir))

csv_dir = 'dataset/AWID3_CSV_balanced'
tfrecord_dir = 'dataset/AWID3_tfrecords_balanced'

with ProcessPoolExecutor() as executor:
   executor.map(process_file, os.listdir(csv_dir))

#for file in os.listdir(csv_dir):
#    process_file(file)