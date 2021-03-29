"""
Gait (steps) example
====================

In this example, we use DiCoDiLe on an open `dataset`_ of gait (steps)
IMU time-series to discover patterns in the data.
We will then use those to attempt to detect steps and compare our findings
with the ground truth.

.. _dataset: https://github.com/deepcharles/gait-data
"""

import matplotlib.pyplot as plt
import numpy as np

from dicodile.data.gait import get_gait_data
from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.viz import display_dictionaries
from dicodile.utils.csc import reconstruct
from dicodile import dicodile

###############################################################################
# Retrieve trial data
# -------------------

trial = get_gait_data(subject=6, trial=1)

###############################################################################
# Let's have a look at the data for one trial.

trial.keys()

###############################################################################
# We get a dictionary whose keys are metadata items, plus a 'data' key that
# contains a numpy array with the trial time series for each sensor axis,
# at 100 Hz resolution.

# right foot acceleration (vertical)
plt.plot(trial['data']['RAV'])

###############################################################################
# Let's look at a small portion of the series for both feet,
# overlaid on the same plot

fig, ax = plt.subplots()
ax.plot(trial['data']['LAV'][5000:5800],
        label='left foot vertical acceleration')
ax.plot(trial['data']['RAV'][5000:5800],
        label='right foot vertical acceleration')
ax.set_xlabel('time (x10ms)')
ax.set_ylabel('acceleration ($m.s^{-2}$)')
ax.legend()

###############################################################################
# We can see the alternating left and right foot movements.
#
# In the rest of this example, we will only use the right foot
# vertical acceleration.

###############################################################################
# Convolutional Dictionary Learning
# ---------------------------------
#
# Now, let's use DiCoDiLe to learn patterns from the data and reconstruct
# the signal from a sparse representation.
#
# First, we initialize a dictionary from parts of the signal:

X = trial['data']['RAV'].to_numpy()
X = X.reshape(1, *X.shape)

print(X.shape)

D_init = init_dictionary(X, n_atoms=8, atom_support=(200,), random_state=60)

###############################################################################
# Note the use of ``reshape`` to shape the signal as per ``dicodile``
# requirements: the shape of the signal should be
# ``(n_channels, *sig_support)``.
# Here, we have a single-channel time series so it is ``(1, n_times)``.

###############################################################################
# Then, we run DiCoDiLe!

D_hat, z_hat, pobj, times = dicodile(
    X, D_init, n_iter=3, n_workers=4, window=True,
    dicod_kwargs={"max_iter": 10000}, verbose=6
)


print("[DiCoDiLe] final cost : {}".format(pobj))

###############################################################################
# We can order the dictionary patches by decreasing sum of the activations'
# absolute values in the activations ``z_hat``, which, intuitively, gives
# a measure of how they contribute to the reconstruction.

sum_abs_val = np.sum(np.abs(z_hat), axis=-1)

# we negate sum_abs_val to sort in decreasing order
patch_indices = np.argsort(-sum_abs_val)

fig_reordered = display_dictionaries(D_init[patch_indices],
                                     D_hat[patch_indices])

###############################################################################
# Signal reconstruction
# ^^^^^^^^^^^^^^^^^^^^^
#
# Now, let's reconstruct the original signal

X_hat = reconstruct(z_hat, D_hat)

###############################################################################
# Plot a small part of the original and reconstructed signals

fig_hat, ax_hat = plt.subplots()
ax_hat.plot(X[0][5000:5800],
            label='right foot vertical acceleration (ORIGINAL)')
ax_hat.plot(X_hat[0][5000:5800],
            label='right foot vertical acceleration (RECONSTRUCTED)')
ax_hat.set_xlabel('time (x10ms)')
ax_hat.set_ylabel('acceleration ($m.s^{-2}$)')
ax_hat.legend()

###############################################################################
# Check that our representation is indeed sparse:

np.count_nonzero(z_hat)

###############################################################################
# Besides our visual check, a measure of how closely we're reconstructing the
# original signal is the (normalized) cross-correlation. Let's compute this:

np.correlate(X[0], X_hat[0]) / (
    np.sqrt(np.correlate(X[0], X[0]) * np.correlate(X_hat[0], X_hat[0])))

###############################################################################
# Multichannel signals
# --------------------
#
# DiCoDiLe works just as well with multi-channel signals. The gait dataset
# contains 16 signals (8 for each foot), in the rest of this tutorial,
# we'll use three of those.

# Left foot Vertical acceleration, Y rotation and X acceleration
channels = ['LAV', 'LRY', 'LAX']

###############################################################################
# Let's look at a small portion of multi-channel data

colors = plt.rcParams["axes.prop_cycle"]()
mc_fig, mc_ax = plt.subplots(len(channels), sharex=True)

for ax, chan in zip(mc_ax, channels):
    ax.plot(trial['data'][chan][5000:5800],
            label=chan, color=next(colors)["color"])
mc_fig.legend(loc="upper center")


###############################################################################
# Let's put the data in shape for DiCoDiLe: (n_channels, n_times)

X_mc_subset = trial['data'][channels].to_numpy().T
print(X_mc_subset.shape)

###############################################################################
# Initialize the dictionary (note that the call is identical
# to the single-channel version)

D_init_mc = init_dictionary(X_mc_subset,
                            n_atoms=8,
                            atom_support=(200,),
                            random_state=60)

###############################################################################
# And run DiCoDiLe (note that the call is identical to the single-channel
# version here as well)

D_hat_mc, z_hat_mc, pobj_mc, times_mc = dicodile(X_mc_subset,
                                                 D_init_mc,
                                                 n_iter=3,
                                                 n_workers=4,
                                                 dicod_kwargs={"max_iter": 10000},  # noqa: E501
                                                 verbose=6,
                                                 window=True)


print("[DiCoDiLe] final cost : {}".format(pobj_mc))

###############################################################################
# Signal reconstruction (multichannel)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, let's reconstruct the original signal

X_hat_mc = reconstruct(z_hat_mc, D_hat_mc)
X_hat_mc.shape

###############################################################################
# Let's visually compare a small part of the original and reconstructed signals
# along with the activations.

viz_start_idx = 4000
viz_end_idx = 5800
viz_chan = 2

max_abs = np.max(np.abs(z_hat_mc), axis=-1)
max_abs = max_abs.reshape(z_hat_mc.shape[0], 1)
z_hat_normalized = z_hat_mc / max_abs
fig_hat_mc, ax_hat_mc = plt.subplots(2, figsize=(12, 8))
ax_hat_mc[0].plot(X_mc_subset[viz_chan][viz_start_idx:viz_end_idx],
                  label='ORIGINAL')
ax_hat_mc[0].plot(X_hat_mc[viz_chan][viz_start_idx:viz_end_idx],
                  label='RECONSTRUCTED')
for idx in range(z_hat_normalized.shape[0]):
    ax_hat_mc[1].stem(z_hat_normalized[idx][viz_start_idx:viz_end_idx],
                      linefmt=f"C{idx}-",
                      markerfmt=f"C{idx}o")
ax_hat_mc[0].set_xlabel('time (x10ms)')
ax_hat_mc[0].legend()
