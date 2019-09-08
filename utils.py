import numpy as np
from tensorflow.python.framework.ops import EagerTensor


def chars_to_labels(chars, alphabet):
    """Converts string or list/tuple of letters to list of labels, based on alphabet.

    Args:
        chars (string/list/tuple): Characters to be converted to indices from alphabet.
        alphabet (list): List of characters in the alphabet.

    Returns:
        list: List of labels
    """
    return [alphabet.index(c) for c in chars] + [alphabet.index('END')]


def labels_to_chars(labels, alphabet):
    """Converts list/tuple of integer labels to string, based on given alphabet.

    Args:
        labels (list/tuple/tf.EagerTensor): List of labels.
        alphabet (list): List of characters in the alphabet.

    Returns:
        list: Decoded string.
    """
    if isinstance(labels, EagerTensor):
        labels = labels.numpy()

    end_id = alphabet.index('END')
    chars = [alphabet[l] if l != end_id else "" for l in labels]
    return ''.join(chars)


def save_alphabet(alphabet, path):
    """Saves alphabet in file.

    First line of file contains valid alphabet, next lines are for debugging purposes only - they
    are not used during loading.

    Args:
        alphabet (list): List of characters in the alphabet
        path (str): full target path of alphabet file
    """
    with open(path, 'w') as f:
        lines = [' , '.join(alphabet) + '\n']
        lines.append('\n###### Only for debugging purposes #######\n')
        lines.extend(['{} : {}\n'.format(i, s) for i, s in enumerate(alphabet)])
        f.writelines(lines)


def load_alphabet(path):
    """Loads alphabet from file.

    Args:
        path (str): full target path of alphabet file

    Returns:
        alphabet (list): List of characters in the alphabet.
    """
    with open(path, 'r') as f:
        line = f.readline()
        line = line.rstrip('\n')
        alphabet = line.split(' , ')
        return alphabet


def preemphasis(signal, coeff=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized_signal


def get_lr(step):
    """Returns learning rate for current training step (values based on paper)"""
    if step < 500000:
        return 0.001
    if step < 1000000:
        return 0.0005
    if step < 2000000:
        return 0.0003
    else:
        return 0.0001