"""Start and shutdown MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
import os
import sys
import time
import warnings
import numpy as np
from mpi4py import MPI
from multiprocessing import util

from ..utils import constants
from ..utils import debug_flags as flags

# global worker communicator
_n_workers = None
_worker_comm = None


SYSTEM_HOSTFILE = os.environ.get("MPI_HOSTFILE", None)


# Constants to start interactive workers
INTERACTIVE_EXEC = "xterm"
INTERACTIVE_ARGS = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


def get_reusable_workers(n_jobs=4, hostfile=None):

    global _worker_comm, _n_workers
    if _worker_comm is None:
        _n_workers = n_jobs
        _worker_comm = _spawn_workers(n_jobs, hostfile)
        util.Finalize(None, shutdown_reusable_workers, exitpriority=20)
    else:
        if _n_workers != n_jobs:
            warnings.warn("You should not require different size")
            shutdown_reusable_workers()
            return get_reusable_workers(n_jobs=n_jobs, hostfile=hostfile)

    return _worker_comm


def send_command_to_reusable_workers(tag, verbose=0):
    global _worker_comm, _n_workers

    t_start = time.time()
    msg = np.empty(1, dtype='i')
    msg[0] = tag
    requests = []
    for i_worker in range(_n_workers):
        requests.append(_worker_comm.Issend([msg, MPI.INT], dest=i_worker,
                                            tag=tag))
    while requests:
        if requests[0].Test():
            requests.pop(0)
        time.sleep(.001)
    if verbose > 5:
        print("Sent message {} in {:.3f}s".format(tag, time.time() - t_start))


def shutdown_reusable_workers():
    global _worker_comm, _n_workers
    if _worker_comm is not None:
        send_command_to_reusable_workers(constants.TAG_WORKER_STOP)
        _worker_comm.Barrier()
        _worker_comm.Disconnect()
        _worker_comm.Free()
        _n_workers = None
        _worker_comm = None


def _spawn_workers(n_jobs, hostfile=None):
    t_start = time.time()
    info = MPI.Info.Create()
    if hostfile is None:
        hostfile = SYSTEM_HOSTFILE
    if hostfile and os.path.exists(hostfile):
        info.Set("hostfile", hostfile)

    # Pass some environment variable to the child process
    envstr = ''
    for key in ['TESTING_DICOD']:
        if key in os.environ:
            envstr += f"{key}={os.environ[key]}\n"
    if envstr != '':
        info.Set("env", envstr)

    # Spawn the workers
    script_name = os.path.join(os.path.dirname(__file__),
                               "main_worker.py")
    if flags.INTERACTIVE_PROCESSES:
        comm = MPI.COMM_SELF.Spawn(
            INTERACTIVE_EXEC, args=INTERACTIVE_ARGS + [script_name],
            maxprocs=n_jobs, info=info)

    else:
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_name],
                                   maxprocs=n_jobs, info=info)
    comm.Barrier()
    print("Started {} workers in {:.3}s".format(n_jobs, time.time() - t_start))
    return comm
