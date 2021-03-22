import PIL
import matplotlib.pyplot as plt
from download import download

from dicodile._config import DATA_HOME

def fetch_mandrill():

    mandrill_dir = DATA_HOME / "images" / "standard_images"
    mandrill_dir.mkdir(parents=True, exist_ok=True)
    mandrill = download(
        "http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03",
        mandrill_dir / "mandrill_color.tif"
    )

    X = plt.imread(mandrill) / 255
    return X.swapaxes(0, 2)


def get_hubble(size="Medium"):

    image_path = f"images/hubble/STScI-H-2016-39-a-{size}.jpg"

    image_path = DATA_HOME / image_path

    PIL.Image.MAX_IMAGE_PIXELS = 617967525
    X = plt.imread(image_path)
    X = X / 255
    return X.swapaxes(0, 2)
