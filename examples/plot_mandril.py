"""
===================================================
Reconstruction of the image Mandrill using dicod
===================================================
This example illlustrates reconstruction of `Mandrill image <http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03>`._

"""

###############################################################################
# We will first download the Mandril image.


from dicodile.utils.shape_helpers import get_valid_support
from dicodile.utils.segmentation import Segmentation
import numpy as np
import matplotlib.pyplot as plt

from dicodile.utils.dictionary import init_dictionary
from dicodile.data.images import get_mandril

from dicodile.utils.viz import display_dictionaries
from dicodile.utils.dictionary import get_lambda_max

from dicodile.update_z.dicod import dicod
from dicodile.utils.csc import compute_objective, reconstruct

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
w_world = 4
n_workers = w_world * w_world

lmbd_max = get_lambda_max(X, D_init).max()
reg_ = reg * lmbd_max

###############################################################################
# Run dicod

z_hat, *_ = dicod(X, D_init, reg_, max_iter=1000000, n_workers=n_workers,
                  tol=tol, strategy='greedy', verbose=1, soft_lock='corner',
                  z_positive=False, timing=False)

pobj = compute_objective(X, z_hat, D_init, reg_)
z_hat = np.clip(z_hat, -1e3, 1e3)
print("[DICOD] final cost : {}".format(pobj))

###############################################################################
# Reconstruct the image from z_hat and D_init


X_hat = reconstruct(z_hat, D_init)
X_hat = np.clip(X_hat, 0, 1)

# Compute the worker segmentation for the image,
n_channels, *sig_support = X_hat.shape
valid_support = get_valid_support(sig_support, atom_support)
workers_segments = Segmentation(n_seg=(w_world, w_world),
                                signal_support=valid_support,
                                overlap=0)

###############################################################################
# Plot reconstructed image

fig = plt.figure("recovery")
fig.patch.set_alpha(0)

ax = plt.subplot()
ax.imshow(X_hat.swapaxes(0, 2))
for i_seg in range(workers_segments.effective_n_seg):
    seg_bounds = np.array(workers_segments.get_seg_bounds(i_seg))
    seg_bounds = seg_bounds + np.array(atom_support) / 2
    ax.vlines(seg_bounds[1], *seg_bounds[0], linestyle='--')
    ax.hlines(seg_bounds[0], *seg_bounds[1], linestyle='--')
ax.axis('off')
plt.tight_layout()
