import json
from zipfile import ZipFile

import pandas as pd
from download import download

from dicodile.config import DATA_HOME


def get_gait_data(subject=1, trial=1):
    """
    Retrieve gait data from this `dataset`_.

    Parameters
    ----------
    subject: int, defaults to 1
        Subject identifier.
        Valid subject-trial pairs can be found in this `list`_.
    trial: int, defaults to 1
        Trial number.
        Valid subject-trial pairs can be found in this `list`_.

    Returns
    -------
    dict
        A dictionary containing metadata and data relative
        to a trial. The 'data' attribute contains time
        series for the trial, as a Pandas dataframe.


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
            data = pd.read_csv(data_file, sep=',', header=0)
            meta['data'] = data
            return meta
