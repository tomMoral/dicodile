import re
import json
from tqdm import tqdm
from zipfile import ZipFile

import pandas as pd
from download import download

from dicodile.config import DATA_HOME

GAIT_RECORD_ID_LIST_FNAME = DATA_HOME / "gait" / "gait_record_id_list.json"
GAIT_PARTICIPANTS_FNAME = DATA_HOME / "gait" / "gait_participants.tsv"


def download_gait(verbose=True):
    gait_dir = DATA_HOME / "gait"
    gait_dir.mkdir(parents=True, exist_ok=True)
    gait_zip = download(
        "http://dev.ipol.im/~truong/GaitData.zip",
        gait_dir / "GaitData.zip",
        replace=False,
        verbose=verbose
    )

    return gait_zip


def get_gait_data(subject=1, trial=1, only_meta=False, verbose=True):
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
    only_meta: bool, default to False
        If True, only returns the subject metadata
    verbose : bool, default to True
        Whether to print download status to the screen.

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

    gait_zip = download_gait(verbose=verbose)

    with ZipFile(gait_zip) as zf:
        with zf.open(f"GaitData/{subject}-{trial}.json") as meta_file, \
                zf.open(f"GaitData/{subject}-{trial}.csv") as data_file:
            meta = json.load(meta_file)
            if not only_meta:
                data = pd.read_csv(data_file, sep=',', header=0)
                meta['data'] = data
            return meta


def get_gait_record_id_list():
    """Returns the list of ids for all available records.

    Returns
    -------
    record_id_list: list
        List of record's id, formed as [subject_id]-[trial].
    """
    if GAIT_RECORD_ID_LIST_FNAME.exists():
        with open(GAIT_RECORD_ID_LIST_FNAME, "r") as f:
            record_id_list = json.load(f)

    else:
        gait_zip = download_gait(verbose=False)

        with ZipFile(gait_zip) as zf:
            all_files = zf.namelist()

        record_id_list = []
        for file in all_files:
            record_id_list.extend(re.findall(r"\d+-\d+", file))

        record_id_list = list(set(record_id_list))  # remove duplicates
        # sort by subject id then by trial number
        record_id_list.sort(key=lambda x: (
            int(x.split('-')[0]), int(x.split('-')[1])))

        # save as JSON
        with open(GAIT_RECORD_ID_LIST_FNAME, 'w') as f:
            json.dump(record_id_list, f, indent=2)

    return record_id_list


def get_participants():
    """Get the information relatives to all individual subjects, such as age,
    gender, number of available trials, etc.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the informations of each individual subjects
    """

    if GAIT_PARTICIPANTS_FNAME.exists():
        participants = pd.read_csv(GAIT_PARTICIPANTS_FNAME, sep='\t')

    else:
        all_records_id = get_gait_record_id_list()
        n_subjects = int(all_records_id[-1].split('-')[0])

        subject_rows = []
        for subject in tqdm(range(1, n_subjects+1)):

            meta = get_gait_data(
                subject, trial=1, only_meta=True, verbose=False
            )

            key_to_remove = [
                'Trial', 'Code', 'LeftFootActivity', 'RightFootActivity'
            ]
            for key in key_to_remove:
                del meta[key]

            subject_trials = [
                idx for idx in all_records_id
                if idx.split('-')[0] == str(subject)
            ]
            meta.update(n_trials=len(subject_trials))
            subject_rows.append(meta)

        participants = pd.DataFrame(subject_rows)
        participants.to_csv(GAIT_PARTICIPANTS_FNAME, sep='\t', index=False)

    return participants


if __name__ == '__main__':
    get_gait_record_id_list()
    get_participants()
