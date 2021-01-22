import os
from pathlib import Path

# DATA_HOME is determined using environment variables.
# The top priority is the environment variable $DICODILE_HOME which is
# specific to this package.
# Else, it falls back on XDG_DATA_HOME if it is set.
# Finally, it defaults to $HOME/data.
# The data will be put in a subfolder 'dicodile'
def get_data_home():
    data_home = os.environ.get(
        'DICODILE_HOME', os.environ.get('XDG_DATA_HOME', None)
    )
    if data_home is None:
        data_home = Path.home() / 'data'

    return Path(data_home) / 'dicodile'


DATA_HOME = get_data_home()
