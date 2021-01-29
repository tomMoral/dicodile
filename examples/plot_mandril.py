"""
===================================================
Reconstruction of the image Mandrill using dicod
===================================================
This example illlustrates reconstruction of `Mandrill image <http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03>`._

"""

###############################################################################
# We will first download the Mandril image.


import numpy as np
import matplotlib.pyplot as plt

from dicodile.utils.dictionary import init_dictionary
from dicodile.data.images import get_mandril

from dicodile.utils.viz import display_dictionaries
from dicodile.utils.dictionary import get_lambda_max

from dicodile.update_z.dicod import dicod

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

###############################################################################
# Set parameters
#
reg = .01

#
tol = 5e-2

#
w_world = 7
n_workers = w_world * w_world

lmbd_max = get_lambda_max(X, D_init).max()
reg_ = reg * lmbd_max

###############################################################################
# Run dicod

# z_hat, *_ = dicod(X, D_init, reg_, max_iter=1000000, n_workers=n_workers,
#                   tol=tol, strategy='greedy', verbose=1, soft_lock='none',
#                   z_positive=False, timing=False)
