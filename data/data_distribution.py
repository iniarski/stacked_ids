import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import data_utils  
import tensorflow as tf


tfrecords_dir='dataset/AWID3_tfrecords'
tfrecords_files = [os.path.join(tfrecords_dir, file) for file in os.listdir(tfrecords_dir) if file.endswith('.tfrecords')]

seq_lens = [1024, 512, 256, 128, 64, 32, 16, 8, 6, 4]
seq_shifts = seq_lens

print('| Sequence length | Sequence shift | Total samples | Attack Samples | Attack % | Normal Samples | Normal % |')
print('|-----------------|----------------|---------------|----------------|----------|----------------|----------|')

for seq_len, seq_shift in zip(seq_lens, seq_shifts):
    labels = [0, 0]
    dataset = data_utils.create_binary_sequential_dataset(tfrecords_files, seq_length=seq_len, seq_shift=seq_shift, batch_size=16, shuffle=False)
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

'''
| Sequence length | Sequence shift | Total samples | Attack Samples | Attack % | Normal Samples | Normal % |
|-----------------|----------------|---------------|----------------|----------|----------------|----------|
| 1024  | 1024  | 5400576  | 433265  | 8.0%  | 4967311  | 92.0 % |
| 512  | 512  | 4848128  | 433265  | 8.9%  | 4414863  | 91.1 % |
| 256  | 256  | 4193024  | 433265  | 10.3%  | 3759759  | 89.7 % |
| 128  | 128  | 3554560  | 433265  | 12.2%  | 3121295  | 87.8 % |
| 64  | 64  | 3006592  | 433265  | 14.4%  | 2573327  | 85.6 % |
| 32  | 32  | 2575136  | 433265  | 16.8%  | 2141871  | 83.2 % |
| 16  | 16  | 2221856  | 433265  | 19.5%  | 1788591  | 80.5 % |
| 8  | 8  | 1835272  | 433265  | 23.6%  | 1402007  | 76.4 % |
| 6  | 6  | 1654464  | 433265  | 26.2%  | 1221199  | 73.8 % |
| 4  | 4  | 1318380  | 433265  | 32.9%  | 885115  | 67.1 % |
'''