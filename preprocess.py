"""Does offline data preprocessing of LJdataset"""
import argparse
import functools
import multiprocessing as mp
import os
import pathlib

import librosa
import numpy as np
import tqdm

from utils import chars_to_ids


def unique_letters(transcriptions: list):
    """Finds unique characters in dataset"""
    unique = set()
    for trans in transcriptions:
        unique.update(list(trans))
    unique = sorted(unique)
    return unique


def process(input, alphabet, args):
    """Processes single audio file: converts it to linear and mel spectrogram, transforms transcription into numerical labels
    and saves all as .npz file.

    The elements of .npz file are:
        transcription (str): Normalized transcription of audio
        ids (list): Labels as integers, ending with <END> token (value=0)
        lin_spectrogram (np.array): Linear spectrogram of audio in form [timesteps, frequencies]
        mel_spectrogram (np.array): Mel-spectrogram of audio in form [timesteps, frequencies]
        restored_audio (np.array): Audio restored from linear spectrograms (for comparison purposes).

    Args:
        input (tuple): Tuple containing filename from meta (without extension) and corresponding string transcription
        alphabet (list): List of all symbols in alphabet.
        args (argparse.args) Additional args, like input and output directories, sample rate etc.
    """
    filename, transcription = input
    input_path = os.path.join(args.input_dir, filename + '.wav')
    output_path = os.path.join(args.output_dir, filename + '.npz')

    audio, sr = librosa.load(input_path, args.sampling_rate)
    window_length = int(args.frame_length * sr / 1000)
    hop_length = int(args.hop_length * sr / 1000)
    lin_spectrogram = np.abs(librosa.stft(audio,
                                          n_fft=args.n_fft,
                                          hop_length=hop_length,
                                          win_length=window_length))
    mel_spectrogram = librosa.feature.melspectrogram(S=lin_spectrogram ** 2, n_mels=args.n_mels)

    # generate audio with Griffin-Lim algorithm from ground truth spectrograms for comparison during training
    restored_audio = librosa.griffinlim(lin_spectrogram, args.griffin_lim_iter, hop_length, window_length)

    ids = chars_to_ids(transcription, alphabet)

    np.savez(output_path,
             transcription=transcription,
             ids=ids,
             lin_spectrogram=lin_spectrogram.T,
             mel_spectrogran=mel_spectrogram.T,
             restored_audio=restored_audio)


def run(args):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    transcriptions = []
    filenames = []
    with open(args.meta_path) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            filename = line.split('|')[0]
            transcription = line.split('|')[2]
            transcriptions.append(transcription)
            filenames.append(filename)

    alphabet = unique_letters(transcriptions)
    alphabet = ['END'] + alphabet  # allows for zero-padding inputs during training

    pool = mp.pool.Pool(args.cpu_count)
    print('Running script on {} CPUs'.format(args.cpu_count))
    list(tqdm.tqdm(pool.imap(functools.partial(process, alphabet=alphabet, args=args), zip(filenames, transcriptions)),
                   total=len(filenames),
                   desc="Converting audio to linear and mel spectrograms..."))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', type=str, help='Path to meta file')
    parser.add_argument('--input_dir', type=str, help='Path to directory with WAV files')
    parser.add_argument('--output_dir', type=str, help='Path to store resulting files')
    parser.add_argument('--sampling_rate', type=int, help='Target sampling rate', default=24000)
    parser.add_argument('--frame_length', type=float, help='Length of STFT frame (in miliseconds)', default=50)
    parser.add_argument('--hop_length', type=float, help='Length of STFT hop (in miliseconds)', default=12.5)
    parser.add_argument('--n_fft', type=int, help='Length of FFT window', default=2048)
    parser.add_argument('--n_mels', type=int, help='Number of mel bins', default=80)
    parser.add_argument('--griffin_lim_iter', type=int, help='Number of Griffin-Lim algorithm iterations', default=50)
    parser.add_argument('--cpu_count', type=int, help='Number of cpu cores to run script concurrently', default=mp.cpu_count())
    args = parser.parse_args()

    run(args)
