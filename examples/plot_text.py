"""DiCoDiLe on text images
==============================

This example illlustrates pattern recovery on a noisy text image using
DiCoDiLe algorithm.

"""  # noqa
import matplotlib.pyplot as plt
import numpy as np

from dicodile.data.text import generate_text
from dicodile.update_d.update_d import tukey_window
from dicodile.utils import check_random_state
from dicodile.utils.dictionary import init_dictionary, prox_d
from dicodile.utils.viz import display_dictionaries


###############################################################################
# We will first generate a text image `X` from a text of **5000**
# characters drawn uniformly from the **4** letters **P** **A** **M**
# **I** and 2 whitespaces.
#
# We also generate the images of the characters used to generate the
# image `X` and assign it to variable `D`.


# number of letters used to generate the text
n_atoms = 4
# number of characters that compose the text image
text_length = 5000

X, D = generate_text(n_atoms=n_atoms, text_length=text_length,
                     random_state='PAMI')


###############################################################################
# We need to reshape image data `X` and `D` to fit to expected signal
# shape of `dicodile`:
#
# (n_channels, *sig_support)

X = X.reshape(1, *X.shape)
D = D[:, None]

# pad `D`
D = np.pad(D, [(0, 0), (0, 0), (4, 4), (4, 4)])


###############################################################################
# Let's display an extract of the generated text image `X` and all the images
# of characters from `D`.

extract_x = X[0][190:490, 250:750]
plt.axis('off')
plt.imshow(extract_x)

display_dictionaries(D)

###############################################################################
# We add a Gaussian white noise with standard deviation σ std (X) and σ = 3
# to `X`.

std = 3
rng = check_random_state(None)

X += std * X.std() * rng.randn(*X.shape)

###############################################################################
# We will create a random dictionary of **K = 4** patches from the noisy image.

# set individual atom (patch) size
atom_support = np.array(D.shape[-2:])

D_init = init_dictionary(X, n_atoms=n_atoms, atom_support=atom_support,
                         random_state=rng)

# normalize the atoms
D_init = prox_d(D_init)

# window the dictionary, this helps make sure that the border values are 0
atom_support = D_init.shape[-2:]
tw = tukey_window(atom_support)[None, None]
D_init *= tw

###############################################################################
# Let's display noisy `X` and random dictionary `D_init` generated from `X`.

zoom_x = X[0][190:490, 250:750]
plt.axis('off')
plt.imshow(zoom_x)

display_dictionaries(D_init)
