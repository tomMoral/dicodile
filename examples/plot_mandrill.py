"""DiCoDiLe on the Mandrill image
==============================

This example illlustrates reconstruction of `Mandrill image
<http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03>`_
using DiCoDiLe algorithm with default soft_lock value "border" and 9
workers.

"""  # noqa

import numpy as np
import matplotlib.pyplot as plt

from dicodile.data.images import fetch_mandrill

from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.viz import display_dictionaries
from dicodile.utils.csc import reconstruct

from dicodile import dicodile


###############################################################################
# We will first download the Mandrill image.

X = fetch_mandrill()

plt.axis('off')
plt.imshow(X.swapaxes(0, 2))


###############################################################################
# We will create a random dictionary of **K = 25** patches of size **8x8**
# from the original Mandrill image to be used for sparse coding.

# set dictionary size
n_atoms = 25

# set individual atom (patch) size
atom_support = (8, 8)

D_init = init_dictionary(X, n_atoms, atom_support, random_state=60)

###############################################################################
# We are going to run `dicodile` with **9** workers on **3x3** grids.

# number of iterations for dicodile
n_iter = 3

# number of iterations for csc (dicodile_z)
max_iter = 10000

# number of splits along each dimension
w_world = 3

# number of workers
n_workers = w_world * w_world

###############################################################################
# Run `dicodile`.

D_hat, z_hat, pobj, times = dicodile(X, D_init, n_iter=n_iter,
                                     n_workers=n_workers,
                                     dicod_kwargs={"max_iter": max_iter},
                                     verbose=6)


print("[DICOD] final cost : {}".format(pobj))

###############################################################################
# Plot and compare the initial dictionary `D_init` with the
# dictionary `D_hat` improved by `dicodile`.

# normalize dictionaries
normalized_D_init = D_init / D_init.max()
normalized_D_hat = D_hat / D_hat.max()

display_dictionaries(normalized_D_init, normalized_D_hat)


###############################################################################
# Reconstruct the image from `z_hat` and `D_hat`.

X_hat = reconstruct(z_hat, D_hat)
X_hat = np.clip(X_hat, 0, 1)


###############################################################################
# Plot the reconstructed image.

fig = plt.figure("recovery")

ax = plt.subplot()
ax.imshow(X_hat.swapaxes(0, 2))
ax.axis('off')
plt.tight_layout()
