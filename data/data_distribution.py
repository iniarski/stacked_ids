import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import data_utils  
import tensorflow as tf


tfrecords_dir='dataset/AWID3_tfrecords'
tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecords')]

seq_lens = [4, 4, 3, 3]
seq_shifts = [3, 2, 2, 1 ]

print('| Sequence length | Sequence shift | Total samples | Attack Samples | Attack % | Normal Samples | Normal % |')
print('|-----------------|----------------|---------------|----------------|----------|----------------|----------|')

for seq_len, seq_shift in zip(seq_lens, seq_shifts):
    labels = [0, 0]
    dataset = data_utils.create_binary_sequential_dataset(tfrecords_files, seq_length=seq_len, seq_shift=seq_shift, batch_size=100, shuffle=False)
    for (X, y) in dataset:
        n_labels = 1;
        for dim in y.shape:
            n_labels *= dim
        anomalies = tf.reduce_sum(y)
        labels[0] += n_labels - anomalies
        labels[1] += anomalies

    n_attack = labels[1].numpy()
    n_normal = labels[0].numpy()

    total = n_attack + n_normal
    attack_perc = n_attack / total * 100.0
    normal_perc = n_normal / total * 100.0

    print(f'| {seq_len}  | {seq_shift}  | {total}  | {n_attack}  | {attack_perc:.1f}%  | {n_normal}  | {normal_perc:.1f} % |')
