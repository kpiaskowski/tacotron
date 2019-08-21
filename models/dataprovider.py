import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import glob

from tensorflow.python.framework.ops import EagerTensor
import multiprocessing as mp
import random
import time

import numpy as np
import tensorflow as tf


class DataProvider:
    def __init__(self, data_path, batch_size, ratio=0.9, seed=int(time.time()), parallel_calls=mp.cpu_count()):
        """Creates tf.data datasets.

        Args:
            data_path (str): Path to directory with .npz files.
            batch_size (int): Size of single batch.
            ratio (float): Real value (0...1) specifying the ratio between number of examples in training and validation dataset
            seed (int): Optional integer specifying seed for data shuffling.
            parallel_calls (int): Number of parallel processes to run, default: number of CPU cores.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.ratio = ratio
        self.seed = seed
        self.parallel_calls = parallel_calls

        self.train_files, self.val_files = self._prepare_train_val()
        self.lin_channels, self.mel_channels = self._get_constant_shapes()

    def _get_constant_shapes(self):
        """Evaluates the values of constant shapes, like number of channels in linear/mel spectrograms, based on example data point"""
        _, lin_spectrogram, mel_spectrogram = self._read_npy(self.train_files[0])
        lin_channels = lin_spectrogram.shape[-1]
        mel_channels = mel_spectrogram.shape[-1]
        return lin_channels, mel_channels

    def _prepare_train_val(self):
        """Splits paths to files to training and validation datasets"""
        rng = random.Random(1234)
        files = glob.glob(self.data_path + '/*.npz')
        rng.shuffle(files)
        num_files = len(files)

        train_files = files[:int(num_files * self.ratio)]
        val_files = files[int(num_files * self.ratio):]
        return train_files, val_files

    def _read_npy(self, file):
        """Reads single npz file"""
        content = np.load(file.numpy() if type(file) == EagerTensor else file)
        labels = content['labels']
        lin_spectrogram = content['lin_spectrogram']
        mel_spectrogram = content['mel_spectrogram']
        return labels, lin_spectrogram, mel_spectrogram

    def _create_tf_dataset(self, files):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.shuffle(500, self.seed)
        dataset = dataset.map(lambda file: tf.py_function(self._read_npy, [file], [tf.int32, tf.float32, tf.float32]), num_parallel_calls=self.parallel_calls)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=([-1], [-1, self.lin_channels], [-1, self.mel_channels]))
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.repeat()

        # iter allows to keep current index while interleaving training and validation datasets
        # please note that it must be manually deleted using `del`, otherwise will hang program forever
        return iter(dataset)

    def datasets(self):
        """
        Returns iterators for both train and validation tf.data datasets. IMPORTANT NOTE: iterators must be manually deleted,
        otherwise the program will hang forever
        """
        return self._create_tf_dataset(self.train_files), self._create_tf_dataset(self.val_files)
