"""Main script for MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""

from dicodile.utils import constants
from dicodile.workers.dicod_worker import DICODWorker
from dicodile.workers.dicodile_worker import dicodile_worker
from dicodile.utils.mpi import wait_message, sync_workers, shutdown_mpi


from threadpoolctl import threadpool_limits
threadpool_limits(1)


def main():
    sync_workers()
    tag = wait_message()
    while tag != constants.TAG_WORKER_STOP:
        if tag == constants.TAG_WORKER_RUN_DICOD:
            dicod = DICODWorker(backend='mpi')
            dicod.run()
        if tag == constants.TAG_WORKER_RUN_DICODILE:
            dicodile_worker()
        tag = wait_message()

    # We should never reach here but to be on the safe side...
    shutdown_mpi()


if __name__ == "__main__":
    main()
