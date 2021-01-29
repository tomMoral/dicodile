"""
===================================================
Reconstruction of the image Mandrill using dicod
===================================================
This example illlustrates reconstruction of `Mandrill image <http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03>`._

"""

###############################################################################
# We will first download the Mandril image.


import numpy as np
from dicodile.utils.dictionary import init_dictionary
import matplotlib.pyplot as plt
from dicodile.data.images import get_mandril

from dicodile.utils.viz import display_dictionaries

X = get_mandril()

###############################################################################
# Plot the image.

plt.axis('off')
plt.imshow(X.swapaxes(0, 2))

###############################################################################
# Let's create a dictionary of K = 25 patches of size 8 x 8 from the original Mandrill image.

# set dictionary size
n_atoms = 25

# set individual atom (patch) size
atom_support = (8, 8)

#
rng = np.random.RandomState(60)

D_init = init_dictionary(X, n_atoms, atom_support, random_state=rng)

###############################################################################
# Let's see the dictionary patches.

list_D = np.array([D_init])
display_dictionaries(*list_D)
