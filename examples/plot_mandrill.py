"""
DiCoDiLe on the Mandrill image
==============================

This example illlustrates reconstruction of `Mandrill image
<http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03>`_
using DiCoDiLe algorithm with soft_lock "corner" and 16 workers.

"""  # noqa

import numpy as np
import matplotlib.pyplot as plt

from dicodile.data.images import fetch_mandrill

from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.viz import display_dictionaries
from dicodile.utils.csc import reconstruct

from dicodile import dicodile


###############################################################################
# We will first download the Mandril image.

X = fetch_mandrill()

plt.axis('off')
plt.imshow(X.swapaxes(0, 2))


###############################################################################
# We will create a random dictionary of K = 25 patches of size 8 x 8 from the
# original Mandrill image to be used for sparse coding.

# set dictionary size
n_atoms = 25

# set individual atom (patch) size
atom_support = (4, 4)

# random state to seed the random number generator
rng = np.random.RandomState(60)

D_init = init_dictionary(X, n_atoms, atom_support, random_state=rng)

###############################################################################
# Set parameters.

w_world = 3
n_workers = w_world * w_world

###############################################################################
# Run DiCoDiLe

pobj, times, D_hat, z_hat = dicodile(X, D_init, n_iter=3,
                                     n_workers=n_workers,
                                     strategy='greedy',
                                     dicod_kwargs={"max_iter": 10000},
                                     verbose=20)


z_hat = np.clip(z_hat, -1e3, 1e3)
print("[DICOD] final cost : {}".format(pobj))

###############################################################################
# Plot the dictionary patches

list_D = np.array([D_hat])
display_dictionaries(*list_D)


###############################################################################
# Reconstruct the image from z_hat and D_init

X_hat = reconstruct(z_hat, D_hat)
X_hat = np.clip(X_hat, 0, 1)


###############################################################################
# Plot the reconstructed image.

fig = plt.figure("recovery")

ax = plt.subplot()
ax.imshow(X_hat.swapaxes(0, 2))
ax.axis('off')
plt.tight_layout()
