import json
from zipfile import ZipFile

import numpy as np
from download import download

from .home import DATA_HOME


def get_gait_data(subject, trial):
    """
    Retrieve gait data from this `dataset`_.

    Parameters
    ----------
    subject: int
        Subject identifier.
        Valid subject-trial pairs can be found in this `list`_.
    trial: int
        Trial number.
        Valid subject-trial pairs can be found in this `list`_.

    Returns
    -------
    dict
        A dictionary containing metadata and data relative
        to a trial.


    .. _dataset: https://github.com/deepcharles/gait-data
    .. _list:
       https://github.com/deepcharles/gait-data/blob/master/code_list.json
    """
    # coerce subject and trial
    subject = int(subject)
    trial = int(trial)

    gait_dir = DATA_HOME / "gait"
    gait_dir.mkdir(parents=True, exist_ok=True)
    gait_zip = download(
        "http://dev.ipol.im/~truong/GaitData.zip",
        gait_dir / "GaitData.zip"
    )

    with ZipFile(gait_zip) as zf:
        with zf.open(f"GaitData/{subject}-{trial}.json") as meta_file, \
             zf.open(f"GaitData/{subject}-{trial}.csv") as data_file:
            meta = json.load(meta_file)
            data = np.genfromtxt(data_file, delimiter=',', names=True)
            meta['data'] = data
            return meta
