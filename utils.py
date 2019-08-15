import numpy as np


def chars_to_ids(chars, alphabet):
    """Converts string or list/tuple of letters to list of their indices, based on alphabet.

    Args:
        chars (string/list/tuple): Characters to be converted to indices from alphabet.
        alphabet (list): List of characters in the alphabet.

    Returns:
        list: List of indices of input characters.
    """
    return [alphabet.index(c) for c in chars] + [alphabet.index('END')]


def preemphasis(signal, coeff=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized_signal
