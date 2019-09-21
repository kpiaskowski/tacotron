import os
import warnings
from datetime import datetime
import time
import librosa
import numpy as np
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.training.training_util import get_or_create_global_step

import tensorflow as tf
import config
import utils
from models.networks import Encoder, DecoderRNN, PostProcessor

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from models.dataprovider import DataProvider
from utils import load_alphabet

# todo move to config
data_dir = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech'
alphabet_path = '/media/kpiaskowski/1721eb42-bb52-4fe7-984f-19ec7ce1fcc0/datasets/converted_LJSpeech/alphabet.txt'
batch_size = 2
max_eval_steps = 350 // config.r
loss_log_step = 1
predict_log_step = 500
save_step = 2000
experiment_prefix = 'tacotron'
max_train_steps = 350


def trim_to_r(labels, mel_spectrograms, lin_spectrograms):
    """Trims input data to be multiplier of r"""
    if lin_spectrograms.shape[1] % config.r != 0:
        even_length = lin_spectrograms.shape[1] // config.r * config.r
        lin_spectrograms = lin_spectrograms[:, :even_length, :]
        mel_spectrograms = mel_spectrograms[:, :even_length, :]
    return labels, lin_spectrograms, mel_spectrograms


def compute_loss(labels, mel_spectrograms, lin_spectrograms, encoder, decoder_rnn, postprocessing_net):
    """Computes loss given inputs.

    Args:
        labels: Input labels to generate audio from.
        mel_spectrograms: Target mel-spectrograms.
        lin_spectrograms: Target lin-spectrograms.
        encoder (tf.keras.Model): Encoder network.
        decoder_rnn (tf.keras.Model): Recurrent decoder network.
        postprocessing_net (tf.keras.Model): Postprocessing network.

    Returns:
        loss_mel: Loss related to mel-spectrogram computation.
        loss_lin: Loss related to linear-spectrogram computation.
        total_loss: Sum of loss_mel and loss_lin.
        gradients: List of computed gradients.
    """
    labels, mel_spectrograms, lin_spectrograms = trim_to_r(labels, mel_spectrograms, lin_spectrograms)

    if max_train_steps is not None:
        lin_spectrograms = lin_spectrograms[:, :max_train_steps, :]
        mel_spectrograms = mel_spectrograms[:, :max_train_steps, :]

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


def train_step(input_example, encoder, decoder_rnn, postprocessing_net, optimizer, step):
    """Performs single training step"""
    labels, mel_spectrograms, lin_spectrograms = input_example
    loss_mels, loss_lin, total_loss, gradients = compute_loss(
        labels=labels,
        mel_spectrograms=mel_spectrograms,
        lin_spectrograms=lin_spectrograms,
        encoder=encoder,
        decoder_rnn=decoder_rnn,
        postprocessing_net=postprocessing_net
    )

    variables = encoder.trainable_variables + decoder_rnn.trainable_variables + postprocessing_net.trainable_variables
    optimizer.learning_rate = utils.get_lr(step.numpy())
    optimizer.apply_gradients(zip(gradients, variables))

    step.assign_add(1)


def predict(labels, max_steps, encoder, decoder_rnn, postprocessing_net):
    """Generates mel and linear predictions given labels

    Args:
        labels: Tensor containing integer labels.
        max_steps (int): Maximum number of steps to perform. If the network will produce <STOP> frame (i.e. all zero
                         outputs), then the actual number of steps might be smaller.

    Returns:
          mel_predictions
          linear_predictions
          attention_scores: Attention weights of shape (batch, decoder_steps // r, encoder_steps)
    """
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


def log_losses(input_example, writer, global_step, encoder, decoder_rnn, postprocessing_net):
    """Logs scalars, i.e. losses

    Args:
        input_example (tuple): Example containing: labels, mel-spectrogram, linear-spectrogram.
        writer (tf.summary.FileWriter): Summary writer, either for training or validation summaries.

    Returns:
        total_loss (float): Total loss for given input example. Returned for debugging purposes.
    """
    labels, mel_spectrograms, lin_spectrograms = input_example
    loss_mels, loss_lin, total_loss, _ = compute_loss(
        labels,
        mel_spectrograms,
        lin_spectrograms,
        encoder,
        decoder_rnn,
        postprocessing_net
    )

    with writer.as_default():
        tf.summary.scalar('losses/mel', data=loss_mels, step=global_step)
        tf.summary.scalar('losses/linear', data=loss_lin, step=global_step)
        tf.summary.scalar('losses/total', data=total_loss, step=global_step)
    return total_loss


def log_other(input_example, writer, vocabulary, encoder, decoder_rnn, postprocessing_net, global_step):
    """Logs data other than scalars, i.e. attention scores and synthesised audio with corresponding query text.

    Args:
        input_example (tuple): Example containing: labels, mel-spectrogram, linear-spectrogram.
        writer (tf.summary.FileWriter): Summary writer, either for training or validation summaries.
    """
    labels, _, _ = input_example

    _, lin_predictions, attention_scores = predict(labels, max_eval_steps, encoder, decoder_rnn, postprocessing_net)
    lin_predictions = tf.transpose(lin_predictions, [0, 2, 1])

    attention_scores = tf.expand_dims(tf.transpose(attention_scores, [0, 2, 1]), -1)
    # normalize scores
    attention_scores = (attention_scores - tf.reduce_min(attention_scores)) \
                       / (tf.reduce_max(attention_scores) - tf.reduce_min(attention_scores) + 1e-8)

    text_labels = np.array([utils.labels_to_chars(lab, vocabulary) for lab in labels])

    synthesised = [librosa.griffinlim(
        S=S,
        hop_length=int(config.hop_size * config.sampling_rate / 1000),
        win_length=int(config.window_size * config.sampling_rate / 1000)
    ) for S in lin_predictions.numpy()]
    synthesised = np.expand_dims(np.array(synthesised), -1)

    with writer.as_default():
        tf.summary.audio(
            name='synthesised audio',
            data=synthesised,
            sample_rate=config.sampling_rate,
            step=global_step,
        )

        tf.summary.image(
            name='attention',
            data=attention_scores,
            step=global_step
        )

        tf.summary.text(
            name='text',
            data=text_labels,
            step=global_step
        )


def run():
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

    # saving and logging ops
    prefix = experiment_prefix + '_' + datetime.now().strftime('%Y-%m-%d %H:%M')
    train_writer = create_file_writer(os.path.join('results/logs', prefix, 'train'), flush_millis=100)
    val_writer = create_file_writer(os.path.join('results/logs', prefix, 'val'), flush_millis=100)
    checkpoint = Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder_rnn=decoder_rnn,
        postprocessing_net=postprocessing_net
    )
    checkpoint_manager = CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join('results/saved', prefix),
        max_to_keep=5
    )

    # training starts here
    for _ in range(global_step.numpy(), config.iterations):
        # training step
        train_step(
            input_example=next(train_dataset),
            encoder=encoder,
            decoder_rnn=decoder_rnn,
            postprocessing_net=postprocessing_net,
            optimizer=optimizer,
            step=global_step
        )

        # log scalar losses
        if global_step % loss_log_step == 0:
            train_loss = log_losses(
                input_example=next(train_dataset),
                writer=train_writer,
                global_step=global_step,
                encoder=encoder,
                decoder_rnn=decoder_rnn,
                postprocessing_net=postprocessing_net,
            )
            val_loss = log_losses(
                input_example=next(val_dataset),
                writer=val_writer,
                global_step=global_step,
                encoder=encoder,
                decoder_rnn=decoder_rnn,
                postprocessing_net=postprocessing_net,
            )
            print('Iteration {}, train loss: {:.6f}, val_loss: {:.6f}'.format(
                global_step.numpy(),
                train_loss,
                val_loss
            ))

        # log non-scalar data
        if global_step % predict_log_step == 0:
            log_other(
                input_example=next(train_dataset),
                writer=train_writer,
                vocabulary=vocabulary,
                global_step=global_step,
                encoder=encoder,
                decoder_rnn=decoder_rnn,
                postprocessing_net=postprocessing_net,
            )
            log_other(
                input_example=next(val_dataset),
                writer=val_writer,
                vocabulary=vocabulary,
                global_step=global_step,
                encoder=encoder,
                decoder_rnn=decoder_rnn,
                postprocessing_net=postprocessing_net,
            )
            print('Logged synthesized audio and attention scores at step {}'.format(global_step.numpy()))

        # save network
        if global_step % save_step == 0 and global_step > 0:
            checkpoint_manager.save(global_step)
            print('Model saved at step {}'.format(global_step.numpy()))


if __name__ == '__main__':
    run()
