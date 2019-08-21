"""Does offline data preprocessing of LJdataset"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import functools
import glob
import multiprocessing as mp
import os
import pathlib

import librosa
import numpy as np
import tqdm

from utils import chars_to_labels, save_alphabet


def unique_letters(transcriptions: list):
    """Finds unique characters in dataset"""
    unique = set()
    for trans in transcriptions:
        unique.update(list(trans))
    unique = sorted(unique)
    return unique


def find_already_processed(output_directory, override):
    """Finds files that are already processed and ones that are possibly corrupt.

    Args:
        output_directory (str): Path to output directory, where processed files will be stored.
        override (bool): Specifies wheter the existing files should be overriden.

    Returns:
        already_processed (list): List of files that are already processed.
        filenames_to_remove (list): List of filenames to remove from previous runs of script.
    """
    current_filenames = os.listdir(output_directory)
    filenames_to_remove = glob.glob(output_directory + '/*.temp_npz')
    already_processed = [name.rstrip('.npz') for name in current_filenames if name.endswith('.npz')]

    if override:
        return [], filenames_to_remove
    else:
        return already_processed, filenames_to_remove


def remove_corrupted_files(filepaths):
    for file in filepaths:
        os.remove(file)


def process(input, alphabet, args):
    """Processes single audio file: converts it to linear and mel spectrogram, transforms transcription into numerical labels
    and saves all as .npz file.

    The elements of .npz file are:
        labels (list): Labels as integers, ending with <END> token (value=0)
        lin_spectrogram (np.array): Linear spectrogram of audio in form [timesteps, frequencies]
        mel_spectrogram (np.array): Mel-spectrogram of audio in form [timesteps, frequencies]

    Args:
        input (tuple): Tuple containing filename from meta (without extension) and corresponding string transcription
        alphabet (list): List of all symbols in alphabet.
        args (argparse.args) Additional args, like input and output directories, sample rate etc.
    """
    filename, transcription = input
    input_path = os.path.join(args.input_dir, filename + '.wav')

    audio, sr = librosa.load(input_path, args.sampling_rate)
    window_length = int(args.frame_length * sr / 1000)
    hop_length = int(args.hop_length * sr / 1000)
    lin_spectrogram = np.abs(librosa.stft(audio,
                                          n_fft=args.n_fft,
                                          hop_length=hop_length,
                                          win_length=window_length))
    mel_spectrogram = librosa.feature.melspectrogram(S=lin_spectrogram ** 2, n_mels=args.n_mels)

    labels = chars_to_labels(transcription, alphabet)

    # according to POSIX, only operation of file renaming is atomic
    temporary_path = os.path.join(args.output_dir, filename + '.temp.npz')
    output_path = os.path.join(args.output_dir, filename + '.npz')
    np.savez(temporary_path,
             labels=labels,
             lin_spectrogram=lin_spectrogram.T,
             mel_spectrogram=mel_spectrogram.T)
    # atomic operation
    os.rename(temporary_path, output_path)


def run(args):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    already_processed_filenames, filenames_to_remove = find_already_processed(args.output_dir, args.override)
    remove_corrupted_files(filenames_to_remove)

    transcriptions = []
    filenames = []
    with open(args.meta_path) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            filename = line.split('|')[0]
            if filename in already_processed_filenames:
                continue
            transcription = line.split('|')[2]
            transcriptions.append(transcription)
            filenames.append(filename)

    alphabet = unique_letters(transcriptions)
    alphabet = ['END'] + alphabet  # allows for zero-padding inputs during training
    save_alphabet(alphabet, os.path.join(args.output_dir, 'alphabet.txt'))

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
    parser.add_argument('--cpu_count', type=int, help='Number of cpu cores to run script concurrently', default=mp.cpu_count()),
    parser.add_argument('--override', help='Boolean flag specifying wheter to override already processed files', action='store_true')
    args = parser.parse_args()

    run(args)
