import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 8
embedding_size = 256

provider = DataProvider(data_dir, batch_size)
train_dataset, val_dataset = provider.datasets()
vocabulary = load_alphabet(alphabet_path)
vocabulary_size = len(vocabulary)


class EncoderPrenet(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size):
        """Embeds input characters and passes them through bottleneck layers.

        Args:
            vocabulary_size (int): Number of symbols in vocabulary.
            embedding_size (int): Size of embedding output dimension.
        """
        super(EncoderPrenet, self).__init__(name='EncoderPrenet')
        self.dense_sizes = [256, 128]

        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_size,
            mask_zero=True)
        self.dense_1 = tf.keras.layers.Dense(self.dense_sizes[0], activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(self.dense_sizes[1], activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, is_training):
        """Computes forward pass of encoder prenet (embeddings + bottlenect layers).

        Args:
            inputs (tf.EagerTensor): Indices of input symbols (labels), shape: [batch, timesteps]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, bottleneck_size (128)]
        """
        x = self.embedding_layer(inputs)
        x = self.dense_1(x)
        x = self.dropout_1(x, training=is_training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=is_training)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.dense_sizes[-1])
        return tf.TensorShape(shape)


# Create an instance of the model
model = EncoderPrenet(vocabulary_size=vocabulary_size, embedding_size=embedding_size)

for i, (labels, lin_spectrograms, mel_spectrograms) in enumerate(train_dataset):
    output = model(labels, is_training=True)
    print(labels.shape)
    print(model.compute_output_shape(labels.shape))
    print(type(output), output.shape)
    exit()
