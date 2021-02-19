import os
from pathlib import Path


def get_data_home():
    """
    DATA_HOME is determined using environment variables.
    The top priority is the environment variable $DICODILE_DATA_HOME which is
    specific to this package.
    Else, it falls back on XDG_DATA_HOME if it is set.
    Finally, it defaults to $HOME/data.
    The data will be put in a subfolder 'dicodile'
    """
    data_home = os.environ.get(
        'DICODILE_DATA_HOME', os.environ.get('XDG_DATA_HOME', None)
    )
    if data_home is None:
        data_home = Path.home() / 'data'

    return Path(data_home) / 'dicodile'


DATA_HOME = get_data_home()
