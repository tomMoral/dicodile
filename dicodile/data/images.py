import PIL
from download import download
import matplotlib.pyplot as plt
import numpy as np

from dicodile.config import DATA_HOME


def fetch_mandrill():

    mandrill_dir = DATA_HOME / "images" / "standard_images"
    mandrill_dir.mkdir(parents=True, exist_ok=True)
    mandrill = download(
        "http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03",
        mandrill_dir / "mandrill_color.tif"
    )

    X = plt.imread(mandrill) / 255
    return X.swapaxes(0, 2)


def fetch_letters_pami():
    """Loads text image `X` and dictionary `D` of the images of the
    letters `P`, `A`, `M`, `I` used to generate `X`.

    Returns
    -------
    X : ndarray, shape (2321, 2004)
        The text image generated from a text of 5000 characters drawn uniformly
        from the letters `P`, `A`, `M`, `I` and 3 whitespaces.
    D : ndarray, shape (4, 29, 25)
        A dictionary of images of the 4 letters `P`, `A`, `M`, `I`.
    """

    pami_dir = DATA_HOME / "images" / "text"
    pami_dir.mkdir(parents=True, exist_ok=True)

    pami_path = download(
        "https://ndownloader.figshare.com/files/26750168", pami_dir /
        "text_4_5000_PAMI.npz")

    data = np.load(pami_path)

    X = data.get('X')
    D = data.get('D')

    return X, D


def get_hubble(size="Medium"):

    image_path = f"images/hubble/STScI-H-2016-39-a-{size}.jpg"

    image_path = DATA_HOME / image_path

    PIL.Image.MAX_IMAGE_PIXELS = 617967525
    X = plt.imread(image_path)
    X = X / 255
    return X.swapaxes(0, 2)
