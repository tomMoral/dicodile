#!/bin/bash
set -euo pipefail

case "$MPI_INSTALL" in
    "conda")
        conda install -y openmpi openmpi-mpicc;;
    "system")
        sudo apt-get install -qy libopenmpi-dev openmpi-bin;;
    *)
        false;;
esac