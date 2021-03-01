from download import download
from zipfile import ZipFile
import json
import numpy as np

from .home import DATA_HOME


def get_gait_data(subject, trial):
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
