import warnings

import numpy as np
from tensorflow.python.ops.summary_ops_v2 import create_file_writer

import config
from models.networks import HighwayDense, PreNet, CBHG, Encoder, AttentionModule, DecoderRNN

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 2

provider = DataProvider(data_dir, batch_size)
train_dataset, val_dataset = provider.datasets()
vocabulary = load_alphabet(alphabet_path)
vocabulary_size = len(vocabulary)





class PostProcessor(tf.keras.Model):
    def __init__(self):
        super(PostProcessor, self).__init__(name='PostProcessor')
        self.K = 8
        self.conv_bank_units = 128
        self.conv_proj_units = [256, 80]
        self.highway_units = [128, 128, 128, 128]
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
encoder = Encoder(
    vocabulary_size=vocabulary_size,
    embedding_size=config.enc_embedding_size,
    prenet_units=config.enc_prenet_units,
    k=config.enc_k,
    conv_bank_units=config.enc_conv_bank_units,
    conv_proj_units=config.enc_conv_proj_units,
    highway_units=config.enc_highway_units,
    gru_units=config.enc_gru_units
)

decoder_rnn = DecoderRNN(
    prenet_units=config.dec_prenet_units,
    attention_units=config.dec_attention_units,
    gru_units=config.dec_gru_units,
    n_mels=config.n_mels,
    r=config.r
)
postprocessing_net = PostProcessor()
optimizer = tf.keras.optimizers.Adam(0.001)

file_writer = create_file_writer("logs")
file_writer.set_as_default()

# todo
# todo saver
# todo log losses, attention, generated audio
# todo validation set
# todo librosa griffin lim
# todo model to separate module
# todo remove OOM trolololo
# todo clean up code of model
# todo clean up code of train
# todo train func / eval func
# todo config

# todo remove

r = config.r

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
        dec_inputs = np.pad(mel_spectrograms, [[0, 0], [1, 0], [0, 0]], mode='constant')
        dec_hidden = [enc_hidden, enc_hidden, enc_hidden]

        loss_mels = 0
        decoder_outputs = []
        for t in range(int(frames / r)):
            dec_input = tf.expand_dims(dec_inputs[:, r * t, :], 1)
            dec_output, dec_hidden, attention_score = decoder_rnn(
                inputs=dec_input,
                prev_hidden_states=dec_hidden,
                enc_outputs=enc_output,
                is_training=True
            )

            target_mels = mel_spectrograms[:, t * r: (t + 1) * r, :]

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
