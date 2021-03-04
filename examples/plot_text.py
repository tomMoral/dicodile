"""DiCoDiLe on text image
==============================

This example illlustrates pattern recovery on text images using
DiCoDiLe algorithm.

"""  # noqa
import matplotlib.pyplot as plt

from dicodile.data.text import generate_text
from dicodile.utils.viz import display_dictionaries


###############################################################################
# We will first generate text image `X` composed of a text of **5000**
# characters drawn uniformly from the **4** letters **P** **A** **M**
# **I** and 2 whitespaces.


# number of letters used to generate the text
n_atoms = 4
# number of characters that compose the text image
text_length = 5000

random_state = 'PAMI'

X, D = generate_text(n_atoms=n_atoms, text_length=text_length,
                     random_state=random_state)

###############################################################################
# Display the text image `X`.

plt.axis('off')
plt.imshow(X)

###############################################################################
# Zoom `X`.

zoom_x = X[190:490, 250:750]
plt.axis('off')
plt.imshow(zoom_x)


###############################################################################
# `D` contains the images of the characters used to generate the image `X`.

display_dictionaries(D)
