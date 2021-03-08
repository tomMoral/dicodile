# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: .spx.py:sphinx
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
# Gait (steps) example
In this example, we use DiCoDiLe on an open `dataset`_ of gait (steps)
IMU time-series to discover patterns in the data.
We will then use those to attempt to detect steps and compare our findings
with the ground truth.

.. _dataset: https://github.com/deepcharles/gait-data
"""

from dicodile.data.gait import get_gait_data
import matplotlib.pyplot as plt
import numpy as np

from dicodile.utils.dictionary import init_dictionary
from dicodile.utils.viz import display_dictionaries
from dicodile.utils.csc import reconstruct

from dicodile import dicodile

###############################################################################
# ## Retrieve trial data

trial = get_gait_data(subject=6, trial=1)

###############################################################################
# Let's have a look at the data for one trial.

trial.keys()

###############################################################################
# We get a dictionary whose keys are metadata items, plus a 'data' key that
# contains a numpy array with the trial time series for each sensor axis,
# at 100 Hz resolution.

# right foot acceleration (vertical)
trial['data']['RAV']

""
# left foot acceleration (vertical)
trial['data']['LAV']

""
plt.plot(trial['data']['LAV'])

""
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
# ## Convolutional Dictionary Learning
# Now, let's use DiCoDiLe to learn patterns from the data and reconstruct
# the signal from a sparse representation.
#
# First, we initialize a dictionary from parts of the signal:

X = trial['data']['RAV']
X = X.reshape(1, *X.shape)

print(X.shape)

D_init = init_dictionary(X, n_atoms=8, atom_support=(300,), random_state=60)

###############################################################################
# Note the use of ``reshape`` to shape the signal as per ``dicodile``
# requirements: the shape of the signal should be
# ``(n_channels, *sig_support)``.
# Here, we have a single-channel time series.

###############################################################################
# Then, we run DiCoDiLe!

D_hat, z_hat, pobj, times = dicodile(X, D_init, n_iter=3,
                                     n_workers=4,
                                     dicod_kwargs={"max_iter": 10000},
                                     verbose=6,
                                     window=True)


print("[DiCoDiLe] final cost : {}".format(pobj))

###############################################################################
# We'll now display the initial and final dictionary side by side

# normalize dictionaries
normalized_D_init = D_init / D_init.max()
normalized_D_hat = D_hat / D_hat.max()

display_dictionaries(normalized_D_init, normalized_D_hat)

###############################################################################
# We can order the dictionary patches by decreasing sum of the activations'
# absolute values in the reconstruction ``z_hat``, which, intuitively, gives
# a measure of how they contribute to the reconstruction.
#
# **TODO: legend**
#
# We now reconstruct a sparse version of the input signal
# and plot it together with the original

sum_abs_val = np.sum(np.abs(z_hat), axis=-1)
sum_abs_val

""
patch_indices = np.argsort(-sum_abs_val)
patch_indices

""
display_dictionaries(normalized_D_init[patch_indices],
                     normalized_D_hat[patch_indices])

###############################################################################
# ### Signal reconstruction
# Now, let's reconstruct the original signal

X_hat = reconstruct(z_hat, D_hat)

""
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
# ## Detecting steps

###############################################################################
# ## Multi-channel signals

# Left foot Vertical acceleration, Y rotation and X acceleration
channels = 'LAV', 'LRY', 'LAX'

###############################################################################
# Let's look at a small portion of multi-channel data

colors = plt.rcParams["axes.prop_cycle"]()
mc_fig, mc_ax = plt.subplots(len(channels), sharex=True)

for ax, chan in zip(mc_ax, channels):
    ax.plot(trial['data'][chan][5000:5800],
            label=chan, color=next(colors)["color"])
mc_fig.legend(loc="upper center")


""
