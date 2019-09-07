import warnings

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 2
r = 2
n_mels = 80

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


class PreNet(tf.keras.Model):
    def __init__(self, units):
        """Does a projection of inputs into another space.

        Args:
            units (list): Number of units per layer.
        """
        super(PreNet, self).__init__(name='PreNet')
        self.units = units

        self.dense_1 = tf.keras.layers.Dense(self.units[0], activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.5)

        self.dense_2 = tf.keras.layers.Dense(self.units[1], activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, is_training):
        x = self.dense_1(inputs)
        x = self.dropout_1(x, training=is_training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=is_training)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.units[-1])
        return tf.TensorShape(shape)


class CBHG(tf.keras.Model):
    def __init__(self, K, conv_bank_units, conv_projection_units, highway_units, gru_units):
        """Subnetwork containing bank of convolutional filters, followed by highway network and bidirectional GRU.

        Args:
            K (int): Number of sets of convolutional filters of size 1...K.
            conv_bank_units (int): Number of units per layer in convolutional bank.
            conv_projection_units (list): Number of units in convolutional projections layers.
            highway_units (int): Number of units in each layer in highway network.
            gru_units (int): Number of units in GRU RNN.
        """
        super(CBHG, self).__init__(name='CBHG')
        self.K = K
        self.out_shape = gru_units
        self.highway_units = highway_units

        self.conv_filter_bank = [tf.keras.layers.Conv1D(conv_bank_units, k, padding='same') for k in range(1, self.K + 1)]
        self.batch_norm_0 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool1D(2, 1, 'same')

        self.conv_proj_1 = tf.keras.layers.Conv1D(conv_projection_units[0], 3, padding='same')
        self.conv_proj_2 = tf.keras.layers.Conv1D(conv_projection_units[1], 3, padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        # projects output from projection convs to size of highway net if there is a mismatch (postprocessing cbhg)
        self.projection_highway_connector = tf.keras.layers.Dense(highway_units)
        self.output_projection = tf.keras.layers.Dense(highway_units)

        self.highway_dense_1 = HighwayDense(highway_units, tf.nn.relu)
        self.highway_dense_2 = HighwayDense(highway_units, tf.nn.relu)
        self.highway_dense_3 = HighwayDense(highway_units, tf.nn.relu)
        self.highway_dense_4 = HighwayDense(highway_units, tf.nn.relu)

        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_state=True, return_sequences=True),
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

        # add additional projection layer in postprocessin cbhg to connect outputs from projection convs to inputs of highway network
        if x.shape[-1] != self.highway_units:
            x = self.projection_highway_connector(x)

        # highway network
        x = self.highway_dense_1(x)
        x = self.highway_dense_2(x)
        x = self.highway_dense_3(x)
        x = self.highway_dense_4(x)

        # bidirectional, residual GRU
        rnn_out, forward_state, backward_state = self.gru(x)
        rnn_state = tf.concat([forward_state, backward_state], -1)
        return rnn_out, rnn_state

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.out_shape)
        return tf.TensorShape(shape)


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__(name='Encoder')

        self.vocabulary_size = vocabulary_size
        self.embedding_size = 256
        self.prenet_units = [256, 128]
        self.K = 16
        self.conv_bank_units = 128
        self.conv_proj_units = [128, 128]
        self.highway_units = 128
        self.gru_units = 128

        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            mask_zero=True
        )
        self.prenet = PreNet(
            units=self.prenet_units
        )
        self.cbhg = CBHG(
            K=self.K,
            conv_bank_units=self.conv_bank_units,
            conv_projection_units=self.conv_proj_units,
            highway_units=self.highway_units,
            gru_units=self.gru_units
        )

    def __call__(self, inputs, is_training):
        """
        Args:
            inputs (tf.EagerTensor): Indices of input symbols (labels), shape: [batch, timesteps]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, bottleneck_size (128)]
        """
        x = self.embedding_layer(inputs)
        x = self.prenet(x, is_training=is_training)
        x = self.cbhg(x, is_training=is_training)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.gru_units)
        return tf.TensorShape(shape)


class AttentionModule(tf.keras.Model):
    """
    Attention module computes context vectors and attention scores. Based on https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention#translate.
    """

    def __init__(self, units):
        super(AttentionModule, self).__init__(name='AttentionModule')
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # return attention weights for debugging purposes
        return context_vector, attention_weights


class DecoderRNN(tf.keras.Model):
    def __init__(self):
        super(DecoderRNN, self).__init__(name='DecoderRNN')
        self.prenet_units = [256, 128]
        self.attention_units = 256
        self.residual_units = 256
        self.mels = n_mels
        self.r = r

        self.prenet = PreNet(
            units=self.prenet_units
        )

        self.attention = AttentionModule(self.attention_units)
        self.attention_gru = tf.keras.layers.GRU(
            self.attention_units,
            return_sequences=True,
            return_state=True
        )

        self.gru_1 = tf.keras.layers.GRU(
            self.residual_units,
            return_sequences=True,
            return_state=True
        )
        self.gru_2 = tf.keras.layers.GRU(
            self.residual_units,
            return_sequences=True,
            return_state=True
        )

        self.projection_layer = tf.keras.layers.Dense(self.mels * self.r)

    def call(self, dec_inputs, hidden_states, enc_outputs, is_training):
        """

        :param dec_inputs: output of previous step (mel spectrograms)
        :param dec_hidden_states: previous hidden states of rnn layers (attention, 2x gru)
        :param enc_outputs: all outputs from encoder
        :param is_training: bool for prenet
        :return: linear mel projections,  n_mels * r -> remember to feed only last one of r during training
        """
        # attention rnn
        prenet_out = self.prenet(dec_inputs, is_training=is_training)

        context_vector, _ = self.attention(hidden_states[0], enc_outputs)  # context_vector 8, 128, att_weight = 8,135,1
        context_vector = tf.expand_dims(context_vector, 1)
        att_rnn_input = tf.concat([context_vector, prenet_out], axis=-1)  # (8, 1, 256)
        attention_rnn_out, attention_state = self.attention_gru(att_rnn_input)  # att_out = (8, 1, 256), att_state = 8,256
        attention_out = tf.concat([context_vector, attention_rnn_out], -1)  # 8,1,384

        res_out_1, res_state_1 = self.gru_1(attention_out, initial_state=hidden_states[1])
        res_out_2, res_state_2 = self.gru_2(res_out_1, initial_state=hidden_states[2])

        output = self.projection_layer(res_out_2)

        return output, [attention_state, res_state_1, res_state_2]


class PostProcessor(tf.keras.Model):
    def __init__(self):
        super(PostProcessor, self).__init__(name='PostProcessor')
        self.K = 8
        self.conv_bank_units = 128
        self.conv_proj_units = [256, 80]
        self.highway_units = 128
        self.gru_units = 128
        self.output_units = 1025

        self.cbhg = CBHG(
            K=self.K,
            conv_bank_units=self.conv_bank_units,
            conv_projection_units=self.conv_proj_units,
            highway_units=self.highway_units,
            gru_units=self.gru_units
        )
        self.output_projection = tf.keras.layers.Dense(self.output_units)

    def __call__(self, inputs, is_training):
        """
        Args:
            inputs (tf.EagerTensor): Outputs from decoder RNN, shaped [batch, timesteps, n_mels]
            is_training (bool): Boolean switch for dropout.

        Returns:
            tf.EagerTensor: Processed inputs, shaped [batch, timesteps, linear_bins]
        """
        rnn_out, *_ = self.cbhg(inputs, is_training=is_training)
        out = self.output_projection(rnn_out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape.append(self.output_units)
        return tf.TensorShape(shape)


# Create an instance of the model
encoder = Encoder()
decoder_rnn = DecoderRNN()
postprocessing_net = PostProcessor()
optimizer = tf.keras.optimizers.Adam(0.001)

file_writer = tf.summary.create_file_writer("logs")
file_writer.set_as_default()


for i, (labels, lin_spectrograms, mel_spectrograms) in enumerate(train_dataset):

    # inputs should be multiples of r
    if lin_spectrograms.shape[1] % r != 0:
        even_length = lin_spectrograms.shape[1] // r * r
        lin_spectrograms = lin_spectrograms[:, :even_length, :]
        mel_spectrograms = mel_spectrograms[:, :even_length, :]

    # todo lol OOM trololo
    lin_spectrograms = lin_spectrograms[:, :350, :]
    mel_spectrograms = mel_spectrograms[:, :350, :]

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(labels, is_training=True)

        # prepend zeroed <GO> frame
        batch, frames, _ = mel_spectrograms.shape
        dec_inputs = np.pad(mel_spectrograms, [[0, 0], [1, 0], [0, 0]])
        dec_hidden = [enc_hidden, enc_hidden, enc_hidden]

        loss_mels = 0
        decoder_outputs = []
        for t in range(int(frames / r)):
            dec_input = tf.expand_dims(dec_inputs[:, r * t, :], 1)
            dec_output, dec_hidden = decoder_rnn(dec_input, dec_hidden, enc_output, is_training=True)

            target_mels = mel_spectrograms[:, t * r: (t + 1) * r, :]
            target_mels = tf.reshape(target_mels, [-1, 1, r * n_mels])

            # L1 loss
            loss_mels += tf.reduce_mean(tf.abs(target_mels - dec_output))

            decoder_outputs.append(dec_output)

        # average loss across decoder outputs
        loss_mels = loss_mels / frames

        # stack all decoder outputs together and reshape them to match input mels
        decoder_outputs = tf.stack(decoder_outputs, 1)
        decoder_outputs = tf.reshape(
            decoder_outputs,
            [batch, frames, -1]
        )

        lin_predictions = postprocessing_net(decoder_outputs, is_training=True)
        loss_lin = tf.reduce_mean(tf.abs(lin_spectrograms - lin_predictions))

        total_loss = loss_mels + loss_lin

        tf.summary.scalar('mel loss', data=loss_mels, step=i)
        tf.summary.scalar('lin loss', data=loss_lin, step=i)
        tf.summary.scalar('total loss', data=total_loss, step=i)


        variables = encoder.trainable_variables + decoder_rnn.trainable_variables + postprocessing_net.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        print(total_loss)