"""
DiCoDiLe on text images
=======================

This example illustrates pattern recovery on a noisy text image using
DiCoDiLe algorithm.

"""  # noqa
import matplotlib.pyplot as plt
import numpy as np

from dicodile import dicodile
from dicodile.data.images import fetch_letters_pami
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.csc import reconstruct
from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.viz import display_dictionaries


###############################################################################
# We will first load PAMI image generated from a text of **5000**
# characters drawn uniformly from the **4** letters **P** **A** **M**
# **I** and 2 whitespaces and assign it to `X`.
#
# We will also load the images of the four characters used to generate
# `X` and assign it to variable `D`.

X_original, D = fetch_letters_pami()


###############################################################################
# We will work on the copy `X` of the original image and we need to reshape
# image data `X` to fit to the expected signal shape of `dicodile`:
#
# `(n_channels, *sig_support)`

X = X_original.copy()
X = X.reshape(1, *X.shape)
print(X.shape)

###############################################################################
# Reshape `D` to fit to dictionary format:
#
# `(n_atoms, n_channels, *atom_support)`

D = D.reshape(4, 1, *D.shape[-2:])

# We pad `D` in order to have atoms with 0 on the border,
# compatible with windowed dictionary.
D = np.pad(D, [(0, 0), (0, 0), (4, 4), (4, 4)])
print(D.shape)

###############################################################################
# Let's display an extract of the original text image `X_original` and
# all the images of characters from `D`.

zoom_x = X_original[190:490, 250:750]
plt.axis('off')
plt.imshow(zoom_x, cmap='gray')

display_dictionaries(D)

###############################################################################
# We add some Gaussian white noise with standard deviation `std` 3 times
# larger than `X.std` to `X`.

std = 3
rng = np.random.default_rng(None)

X += std * X.std() * rng.standard_normal(X.shape)

###############################################################################
# We will create a random dictionary of **K = 10** patches from the
# noisy image.

# set number of patches
n_atoms = 10
# set individual atom (patch) size
atom_support = np.array(D.shape[-2:])

D_init = init_dictionary(X, n_atoms=n_atoms, atom_support=atom_support,
                         random_state=60)

# window the dictionary, this helps make sure that the border values are 0
atom_support = D_init.shape[-2:]
tw = tukey_window(atom_support)[None, None]
D_init *= tw

print(D_init.shape)

###############################################################################
# Let's display an extract of noisy `X` and random dictionary `D_init`
# generated from `X`.

zoom_x = X[0][190:490, 250:750]
plt.axis('off')
plt.imshow(zoom_x, cmap='gray')

display_dictionaries(D_init)

###############################################################################
# Set model parameters.

# regularization parameter
reg = .2
# maximum number of iterations
n_iter = 100
# when True, makes sure that the borders of the atoms are 0
window = True
# when True, requires all activations Z to be positive
z_positive = True
# number of workers to be used for computations
n_workers = 10
# number of jobs per row
w_world = 'auto'
# tolerance for minimal update size
tol = 1e-3

###############################################################################
# Fit the dictionary with `dicodile`.
D_hat, z_hat, pobj, times = dicodile(X, D_init, reg=reg, n_iter=n_iter,
                                     window=window, z_positive=z_positive,
                                     n_workers=n_workers,
                                     dicod_kwargs={"max_iter": 10000},
                                     w_world=w_world, tol=tol, verbose=6)

print("[DICOD] final cost : {}".format(pobj))

###############################################################################
# Let's compare the initially generated random patches in `D_init`
# with the atoms in `D_hat` recovered with `dicodile`.

display_dictionaries(D_init, D_hat)

###############################################################################
# Now we will reconstruct the image from `z_hat` and `D_hat`.

X_hat = reconstruct(z_hat, D_hat)
X_hat = np.clip(X_hat, 0, 1)

###############################################################################
# Let's plot the reconstructed image `X_hat` together with the
# original image `X_original` and the noisy image `X` that was input
# to `dicodile`.

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[6.4, 8])

ax1.imshow(X_original[190:490, 250:750], cmap='gray')
ax1.set_title('Original image')
ax1.axis('off')

ax2.imshow(X[0][190:490, 250:750], cmap='gray')
ax2.set_title('Noisy image')
ax2.axis('off')

ax3.imshow(X_hat[0][190:490, 250:750], cmap='gray')
ax3.set_title('Recovered image')
ax3.axis('off')
plt.tight_layout()
