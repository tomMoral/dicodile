#!/bin/bash
set -euo pipefail

case "$MPI_INSTALL" in
    "conda")
        conda install -y "$MPI_IMPL" mpi4py
	;;
    "system")
        sudo apt-get update
	case "$MPI_IMPL" in
	    "openmpi")
		sudo apt-get install -qy libopenmpi-dev openmpi-bin
		;;
	    "mpich")
		sudo apt-get install -qy mpich
		;;
	    *)
		false
		;;
	esac
	;;
    *)
	false;;
esac

