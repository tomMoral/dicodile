from dicodile.utils import constants
from dicodile.workers.dicod_worker import DICODWorker
from dicodile.utils.mpi import wait_message


def dicodile_worker():
    dicod = DICODWorker(backend='mpi')
    dicod.recv_task()

    tag = wait_message()
    while tag != constants.TAG_DICODILE_STOP:
        if tag == constants.TAG_DICODILE_COMPUTE_Z_HAT:
            dicod.compute_z_hat()
        if tag == constants.TAG_DICODILE_GET_COST:
            dicod.return_cost()
        if tag == constants.TAG_DICODILE_GET_Z_HAT:
            dicod.return_z_hat()
        if tag == constants.TAG_DICODILE_GET_Z_NNZ:
            dicod.return_z_nnz()
        if tag == constants.TAG_DICODILE_GET_SUFFICIENT_STAT:
            dicod.return_sufficient_statistics()
        if tag == constants.TAG_DICODILE_SET_D:
            dicod.recv_D()
        if tag == constants.TAG_DICODILE_SET_PARAMS:
            dicod.recv_params()
        if tag == constants.TAG_DICODILE_SET_SIGNAL:
            dicod.recv_signal()
        tag = wait_message()
