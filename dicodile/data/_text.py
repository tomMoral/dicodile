import os
import string
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from dicodile.config import DATA_HOME
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import prox_d


TMP = pathlib.Path('/tmp')
if not TMP.exists():
    TMP = pathlib.Path('.')

TEXT_DATA_DIR = DATA_HOME / 'images' / 'text'
HEADER_FILE = os.path.join(os.path.dirname(__file__), 'header.tex')


##############################################################################
# Command line to generate the image from the text using pandoc, pdfcrop
# and ImageMagik. One should make sure these utilities are available on
# the pass to use the function in this module.
#

PANDOC_CMD = f"""pandoc \
    -f gfm --include-in-header {HEADER_FILE} \
    -V geometry:a4paper -V geometry:margin=2cm \
    -V fontsize=8pt -V mainfont='typewriter' \
    -V monofont='typewriter' \
    {{name}}.md -o {{name}}.pdf
"""

PDFCROP_CMD = """
    pdfcrop --margin {margin} \
    {name}.pdf {name}.pdf > /dev/null
"""

CONVERT_CMD = """convert \
    -quality 100 -density 300 \
    -alpha off -negate -strip \
    {name}.pdf {name}.png
"""


def convert_str_to_png(text, margin=12):
    """Returns the image associated to a string of characters.

    Parameters
    ----------
    text: str
        Text to encode as an image.
    margin: int (default: 12)
        Margin to add around the text. To generate a dictionary element, one
        should use 0 as a margin.

    Returns
    -------
    im : ndarray, shape (height, width)
        image associated to `text`.

    """
    filename = str(TMP / 'sample')
    with open(f"{filename}.md", 'w') as f:
        f.write(text)
    assert os.system(PANDOC_CMD.format(name=filename)) == 0
    assert os.system(PDFCROP_CMD.format(
        name=filename, margin=margin)) == 0
    assert os.system(CONVERT_CMD.format(name=filename)) == 0
    im = plt.imread(f"{filename}.png")
    return im


def get_centered_padding(shape, expected_shape):
    """Compute a padding to have an array centered in the expected_shape.

    Parameters
    ----------
    shape: tuple
      Original array dimensions.
    expected_shape: ndarray, tuple
      Expected array dimensions.

    Returns
    -------
    padding: list
        padding necessary for original array to have the `expected_shape`.
    """

    padding = []
    for s, es in zip(shape, expected_shape):
        pad = es - s
        padding.append((pad // 2, (pad + 1) // 2))
    return padding


def generate_text(n_atoms=5, text_length=3000, n_spaces=3, random_state=None):
    """Generate a text image with text_length leters chosen among n_atoms.

    Parameters
    ----------
    n_atoms: int (default: 5)
        Number of letters used to generate the text. This should not be above
        26 as only lower-case ascii letters are used here.
    text_length: int (default: 3000)
        Number of character that compose the text image. This also account for
        white space characters.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    Returns
    -------
    X: ndarray, shape (height, width)
        Image composed of a text of `text_length` characters drawn uniformly
        among `n_atoms` letters and 2 whitespaces.
    D: ndarray, shape (n_atoms, *atom_support)
        Images of the characters used to generate the image `X`.
    """

    if random_state == 'PAMI':
        rng = check_random_state(0)
        D_char = np.array(list('PAMI' + ' ' * n_spaces))
    else:
        rng = check_random_state(random_state)
        chars = list(string.ascii_lowercase)
        D_char = np.r_[rng.choice(chars, replace=False, size=n_atoms),
                       [' '] * n_spaces]
    text_char_idx = rng.choice(len(D_char), replace=True, size=text_length)

    text = ''.join([D_char[i] for i in text_char_idx])

    X = convert_str_to_png(text, margin=0)
    D = [convert_str_to_png(D_k, margin=0) for D_k in D_char[:-n_spaces]]

    # Reshape all atoms to the same shape
    D_reshaped = []
    atom_shape = np.array([dk.shape for dk in D]).max(axis=0)
    for dk in D:
        padding = get_centered_padding(dk.shape, atom_shape)
        D_reshaped.append(np.pad(dk, padding))
    D = np.array(D_reshaped)
    D = prox_d(D)

    print(f"{text_length} - image shape: {X.shape}, pattern shape: {D.shape}")

    return X, D


def generate_text_npy(n_atoms=5, text_length=3000, random_state=None):
    """Generate a file with image and patterns from func:`generate_text`.

    Parameters
    ----------
    n_atoms: int (default: 5)
        Number of letters used to generate the text. This should not be above
        26 as only lower-case ascii letters are used here.
    text_length: int (default: 3000)
        Number of character that compose the text image. This also account for
        white space characters.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    Returns
    -------
    filename: str
        Name of the generated file.
    """
    X, D = generate_text(n_atoms=n_atoms, text_length=text_length,
                         random_state=random_state)
    tag = f"{n_atoms}_{text_length}"
    if isinstance(random_state, (int, str)):
        tag = f"{tag}_{random_state}"
    filename = f'text_{tag}.npz'
    np.savez(TEXT_DATA_DIR / filename, X=X, D=D, text_length=text_length)
    return filename


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Generate data for the text experiment for Dicodile')
    parser.add_argument('--max-length', '-l', type=int, default=5000,
                        help='Maximal length of the generate image.')
    parser.add_argument('--n-rep', '-r', type=int, default=5,
                        help='Number of repetition that will be required.')
    parser.add_argument('--n-atoms', '-k', type=int, default=5,
                        help='Number of letters used to generate the image.')
    parser.add_argument('--PAMI', action='store_true',
                        help='Generate an data with PAMI letters.')
    args = parser.parse_args()

    if args.PAMI:
        print(generate_text_npy(n_atoms=4, text_length=5000,
                                random_state='PAMI'))
        raise SystemExit(0)

    files = []
    for l in np.logspace(np.log10(150 + .1),  # noqa: E741
                         np.log10(args.max_length + .1),
                         num=5, dtype=int):
        for seed in range(args.n_rep):
            files.append(generate_text_npy(
                n_atoms=args.n_atoms, text_length=l, random_state=seed
            ))
    print(' '.join(files))
    print(files)
