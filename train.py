import warnings

import numpy as np
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from tensorflow.python.training.training_util import get_or_create_global_step

import config
import utils
from models.networks import Encoder, DecoderRNN, PostProcessor

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 2
max_eval_steps = 350 // config.r
loss_log_step = 1
predict_log_step = 10

provider = DataProvider(data_dir, batch_size)
train_dataset, val_dataset = provider.datasets()
vocabulary = load_alphabet(alphabet_path)
vocabulary_size = len(vocabulary)

# create network
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

postprocessing_net = PostProcessor(
    k=config.postproc_k,
    conv_bank_units=config.postproc_conv_bank_units,
    conv_proj_units=config.postproc_conv_proj_units,
    highway_units=config.postproc_highway_units,
    gru_units=config.postproc_gru_units,
    output_units=config.postproc_output_units
)

# training ops
global_step = get_or_create_global_step()
optimizer = tf.keras.optimizers.Adam(utils.get_lr(global_step.numpy()))

train_writer = create_file_writer("logs/train", flush_millis=100)
val_writer = create_file_writer("logs/val", flush_millis=100)


# todo check global step saving etc
# todo saver
# todo log losses, attention, generated audio
# todo validation set
# todo librosa griffin lim
# todo model to separate module
# todo remove OOM trolololo
# todo clean up code of train
# todo train func / eval func
# todo power 1.2 griffin lim


def pad_to_r(labels, mel_spectrograms, lin_spectrograms):
    """Trims input data to be multiplier of r"""
    if lin_spectrograms.shape[1] % config.r != 0:
        even_length = lin_spectrograms.shape[1] // config.r * config.r
        lin_spectrograms = lin_spectrograms[:, :even_length, :]
        mel_spectrograms = mel_spectrograms[:, :even_length, :]
    return labels, lin_spectrograms, mel_spectrograms


def compute_loss(labels, mel_spectrograms, lin_spectrograms):
    """Computes loss given inputs.

    Returns:
        loss_mel: Loss related to mel-spectrogram computation.
        loss_lin: Loss related to linear-spectrogram computation.
        total_loss: Sum of loss_mel and loss_lin.
        gradients: List of computed gradients.
    """
    labels, mel_spectrograms, lin_spectrograms = pad_to_r(labels, mel_spectrograms, lin_spectrograms)

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
        for t in range(int(frames / config.r)):
            dec_input = tf.expand_dims(dec_inputs[:, config.r * t, :], 1)
            dec_output, dec_hidden, attention_score = decoder_rnn(
                inputs=dec_input,
                prev_hidden_states=dec_hidden,
                enc_outputs=enc_output,
                is_training=True
            )

            target_mels = mel_spectrograms[:, t * config.r: (t + 1) * config.r, :]
            loss_mels += tf.reduce_mean(tf.abs(target_mels - dec_output))

            decoder_outputs.append(dec_output)

        # average mel loss across all decoder outputs
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

        variables = encoder.trainable_variables + decoder_rnn.trainable_variables + postprocessing_net.trainable_variables
        gradients = tape.gradient(total_loss, variables)

        return loss_mels, loss_lin, total_loss, gradients


def train_step(input_example):
    """Performs single training step"""
    labels, mel_spectrograms, lin_spectrograms = input_example
    loss_mels, loss_lin, total_loss, gradients = compute_loss(labels, mel_spectrograms, lin_spectrograms)

    variables = encoder.trainable_variables + decoder_rnn.trainable_variables + postprocessing_net.trainable_variables
    optimizer.learning_rate = utils.get_lr(global_step.numpy())
    optimizer.apply_gradients(zip(gradients, variables))

    global_step.assign_add(1)


def predict(labels, max_steps):
    """Generated mel and linear predictions given labels"""
    batch, _ = labels.shape

    # encoder
    enc_output, enc_hidden = encoder(labels, is_training=False)

    # prepare decoder
    step = 0
    start_frame = tf.zeros((batch, 1, config.n_mels))
    attention_scores, mel_predictions, lin_predictions = [], [], []

    # compute decoder output step by step
    dec_input = start_frame
    dec_hidden = [enc_hidden, enc_hidden, enc_hidden]
    while step < max_steps:
        dec_output, dec_hidden, attention_score = decoder_rnn(
            inputs=dec_input,
            prev_hidden_states=dec_hidden,
            enc_outputs=enc_output,
            is_training=False
        )

        attention_scores.append(attention_score)
        mel_predictions.append(dec_output)

        if np.allclose(dec_output.numpy(), 0.0):
            break

        step += 1
    attention_scores = tf.stack(attention_scores, 1)
    mel_predictions = tf.concat(mel_predictions, 1)
    lin_predictions = postprocessing_net(mel_predictions, is_training=False)

    return mel_predictions, lin_predictions, attention_scores


def log_losses(input_example, writer):
    labels, mel_spectrograms, lin_spectrograms = input_example
    loss_mels, loss_lin, total_loss, _ = compute_loss(
        labels,
        mel_spectrograms,
        lin_spectrograms
    )

    with writer.as_default():
        tf.summary.scalar('losses/mel', data=loss_mels, step=global_step)
        tf.summary.scalar('losses/linear', data=loss_lin, step=global_step)
        tf.summary.scalar('losses/total', data=total_loss, step=global_step)

    return total_loss


for _ in range(global_step.numpy(), config.iterations):

    # training step
    train_step(next(train_dataset))
    print(global_step)

    if global_step % loss_log_step == 0:
        train_loss = log_losses(
            input_example=next(train_dataset),
            writer=train_writer
        )
        val_loss = log_losses(
            input_example=next(val_dataset),
            writer=val_writer
        )
        print('Iteration {}, train loss: {:.6f}, val_loss: {:.6f}'.format(
            global_step.numpy(),
            train_loss,
            val_loss
        ))

    if global_step % predict_log_step == 0:
        print('Computing predicionts')

        labels, _, _ = next(train_dataset)
        text_labels = [utils.labels_to_chars(lab, vocabulary) for lab in labels]
        mel_predictions, lin_predictions, attention_scores = predict(labels, max_eval_steps)
        # train griffin lim
        # + log attention
        # + log restored autio

        labels, _, _ = next(train_dataset)
        text_labels = [utils.labels_to_chars(lab, vocabulary) for lab in labels]
        mel_predictions, lin_predictions, attention_scores = predict(labels, max_eval_steps)
        # val griffin lim
        # + log attention
        # + log restored autio
