import os
import PIL
import matplotlib.pyplot as plt


DATA_DIR = os.environ.get("DATA_DIR", "../../data")


def get_mandril():

    mandril = os.path.join(DATA_DIR,
                           "images/standard_images/mandril_color.tif")
    X = plt.imread(mandril) / 255
    return X.swapaxes(0, 2)


def get_hubble(size="Medium"):

    image_path = f"images/hubble/STScI-H-2016-39-a-{size}.jpg"

    image_path = os.path.join(DATA_DIR, image_path)

    PIL.Image.MAX_IMAGE_PIXELS = 617967525
    X = plt.imread(image_path)
    X = X / 255
    return X.swapaxes(0, 2)
