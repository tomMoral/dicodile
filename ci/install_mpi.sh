#!/bin/bash
set -euo pipefail

case "$MPI_INSTALL" in
    "conda")
        conda install -y openmpi;;
    "system")
        sudo apt-get install -qy openmpi-bin;;
    *)
        false;;
esac