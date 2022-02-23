from dicodile.utils import constants
from dicodile.workers.dicod_worker import DICODWorker
from dicodile.utils.mpi import wait_message


def dicodile_worker():
    dicod_worker = DICODWorker(backend='mpi')

    tag = wait_message()
    while tag != constants.TAG_DICODILE_STOP:
        if tag == constants.TAG_DICODILE_COMPUTE_Z_HAT:
            dicod_worker.compute_z_hat()
        if tag == constants.TAG_DICODILE_GET_COST:
            dicod_worker.return_cost()
        if tag == constants.TAG_DICODILE_GET_Z_HAT:
            dicod_worker.return_z_hat()
        if tag == constants.TAG_DICODILE_GET_Z_NNZ:
            dicod_worker.return_z_nnz()
        if tag == constants.TAG_DICODILE_GET_SUFFICIENT_STAT:
            dicod_worker.return_sufficient_statistics()
        if tag == constants.TAG_DICODILE_SET_D:
            dicod_worker.recv_D()
        if tag == constants.TAG_DICODILE_SET_PARAMS:
            dicod_worker.recv_params()
        if tag == constants.TAG_DICODILE_SET_SIGNAL:
            dicod_worker.recv_signal()
        if tag == constants.TAG_DICODILE_SET_TASK:
            dicod_worker.recv_task()
        if tag == constants.TAG_DICODILE_GET_MAX_ERROR_PATCH:
            dicod_worker.compute_and_return_max_error_patch()
        tag = wait_message()
