import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 8
embedding_size = 256
k_filters = 16
encoder_cbhg_units = 128

provider = DataProvider(data_dir, batch_size)
train_dataset, val_dataset = provider.datasets()
vocabulary = load_alphabet(alphabet_path)
vocabulary_size = len(vocabulary)


class HighwayDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu):
        """Implements Highway Networks (the networks inspired by gates from LSTMs)

        Args:
            units (int): Number of output units.
            activation (tf.nn.activation): Activation function, by default tf.nn.relu
        """
        super(HighwayDense, self).__init__(name='HighwayDense')
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w_t = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True,
                                   name='weight_transform')
        self.b_t = self.add_weight(shape=(self.units,),
                                   initializer='random_normal',
                                   trainable=True,
                                   name='bias_transform')
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='weight')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='bias')

    def call(self, inputs):
        x = inputs
        T = tf.sigmoid(tf.matmul(x, self.w_t) + self.b_t)
        H = self.activation(tf.matmul(x, self.w) + self.b)
        C = 1.0 - T
        return tf.multiply(H, T) + tf.multiply(x, C)


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
        self.dense_1 = tf.keras.layers.Dense(self.dense_sizes[0], activation='relu', input_shape=(None, None, embedding_size))
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(self.dense_sizes[1], activation='relu', input_shape=(None, None, embedding_size))
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, is_training):
        """
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


class EncoderCBHG(tf.keras.Model):
    def __init__(self, K, units):
        """Subnetwork containing bank of convolutional filters, followed by highway network and bidirectional GRU.

        Args:
            K (int): Number of sets of convolutional filters of size 1...K.
            units (int): Number of units in layers.
        """
        super(EncoderCBHG, self).__init__(name='EncoderCBHG')
        self.K = K
        self.units = units

        self.conv_filter_bank = [tf.keras.layers.Conv1D(units, k, padding='same') for k in range(1, self.K + 1)]
        self.batch_norm_0 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool1D(2, 1, 'same')

        self.conv_proj_1 = tf.keras.layers.Conv1D(self.units, 3, padding='same')
        self.conv_proj_2 = tf.keras.layers.Conv1D(self.units, 3, padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.highway_dense_1 = HighwayDense(self.units, tf.nn.relu)
        self.highway_dense_2 = HighwayDense(self.units, tf.nn.relu)
        self.highway_dense_3 = HighwayDense(self.units, tf.nn.relu)
        self.highway_dense_4 = HighwayDense(self.units, tf.nn.relu)

        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.units, return_sequences=True),
            input_shape=(None, None, self.units),
            merge_mode='sum'
        )

    def call(self, inputs, is_training):
        """
        Args:
            inputs (tf.EagerTensor): Outputs from PreNet, shape: [batch, timesteps, prenet_projection_size]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, projection_size (128)]
        """

        # convolution banks and max pooling
        x = [self.conv_filter_bank[i](inputs) for i in range(self.K)]
        x = tf.concat(x, -1)
        x = self.batch_norm_0(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        # convolutional projection
        x = self.conv_proj_1(x)
        x = self.batch_norm_1(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv_proj_2(x)
        x = self.batch_norm_2(x, training=is_training)

        # residual connection
        x = x + inputs

        # highway network
        x = self.highway_dense_1(x)
        x = self.highway_dense_2(x)
        x = self.highway_dense_3(x)
        x = self.highway_dense_4(x)

        # bidirectional, residual GRU
        x = x + self.gru(x)

        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.units)
        return tf.TensorShape(shape)


# Create an instance of the model
encoder_prenet = EncoderPrenet(vocabulary_size=vocabulary_size, embedding_size=embedding_size)
encoder_cbhg = EncoderCBHG(k_filters, encoder_cbhg_units)

for i, (labels, lin_spectrograms, mel_spectrograms) in enumerate(train_dataset):
    encoder_prenet_output = encoder_prenet(labels, is_training=True)
    print('encoder_prenet_output', encoder_prenet_output.shape)
    encoder_cbhg_output = encoder_cbhg(encoder_prenet_output, is_training=True)
    print('encoder_cbhg_output', encoder_cbhg_output.shape)

    exit()
