import os
import pathlib
import PIL
import matplotlib.pyplot as plt
from download import download

from .home import DATA_HOME

def get_mandril():

    mandril_dir = pathlib.Path(DATA_HOME) / "images" / "standard_images"
    mandril_dir.mkdir(parents=True, exist_ok=True)
    mandril = download("http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03", mandril_dir / "mandril_color.tif")

    X = plt.imread(mandril) / 255
    return X.swapaxes(0, 2)


def get_hubble(size="Medium"):

    image_path = f"images/hubble/STScI-H-2016-39-a-{size}.jpg"

    image_path = os.path.join(DATA_HOME, image_path)

    PIL.Image.MAX_IMAGE_PIXELS = 617967525
    X = plt.imread(image_path)
    X = X / 255
    return X.swapaxes(0, 2)
