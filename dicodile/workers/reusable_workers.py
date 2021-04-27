"""Start and shutdown MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
import os
import sys
import time
import warnings
import numpy as np
from mpi4py import MPI

from ..utils import constants
from ..utils import debug_flags as flags

# global worker communicator
workers = None


SYSTEM_HOSTFILE = os.environ.get("MPI_HOSTFILE", None)


# Constants to start interactive workers
INTERACTIVE_EXEC = "xterm"
INTERACTIVE_ARGS = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


class Workers:
    def __init__(self, n_workers, hostfile):
        self.comm = _spawn_workers(n_workers, hostfile)
        self.n_workers = n_workers
        self.hostfile = hostfile
        self.shutdown = False

    def __del__(self):
        if not self.shutdown:
            shutdown_reusable_workers(_workers=self)


def get_reusable_workers(n_workers=4, hostfile=None):

    global workers
    if workers is None:
        workers = Workers(n_workers, hostfile)
    else:
        if workers.n_workers != n_workers:
            warnings.warn("You should not require different size")
            shutdown_reusable_workers()
            workers = None
            time.sleep(.5)
            return get_reusable_workers(n_workers=n_workers, hostfile=hostfile)

    return workers.comm


def send_command_to_reusable_workers(tag, _workers=None, verbose=0):
    if _workers is None:
        global workers
        _workers = workers

    msg = np.empty(1, dtype='i')
    msg[0] = tag
    t_start = time.time()
    for i_worker in range(_workers.n_workers):
        _workers.comm.Send([msg, MPI.INT], dest=i_worker, tag=tag)
    if verbose > 5:
        print("Sent message {} in {:.3f}s".format(tag, time.time() - t_start))


def shutdown_reusable_workers(_workers=None):
    if _workers is None:
        global workers
        _workers = workers

    if _workers is not None:
        send_command_to_reusable_workers(constants.TAG_WORKER_STOP,
                                         _workers=_workers)
        _workers.comm.Barrier()
        _workers.comm.Disconnect()
        MPI.COMM_SELF.Barrier()
        _workers.shutdown = True


def _spawn_workers(n_workers, hostfile=None):
    t_start = time.time()
    info = MPI.Info.Create()
    if hostfile is None:
        hostfile = SYSTEM_HOSTFILE
    if hostfile and os.path.exists(hostfile):
        info.Set("hostfile", hostfile)

    # Pass some environment variable to the child process
    env_str = ''
    for key in ['TESTING_DICOD']:
        if key in os.environ:
            env_str += f"{key}={os.environ[key]}\n"
    if env_str != '':
        info.Set("env", env_str)

    # Spawn the workers
    script_name = os.path.join(os.path.dirname(__file__),
                               "main_worker.py")
    exception = None

    MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_RETURN)
    for i in range(10):
        MPI.COMM_SELF.Barrier()
        try:
            if flags.INTERACTIVE_PROCESSES:
                comm = MPI.COMM_SELF.Spawn(
                    INTERACTIVE_EXEC, args=INTERACTIVE_ARGS + [script_name],
                    maxprocs=n_workers, info=info
                )

            else:
                comm = MPI.COMM_SELF.Spawn(
                    sys.executable,
                    args=["-W", "error::RuntimeWarning", script_name],
                    maxprocs=n_workers, info=info
                )
            break
        except Exception as e:
            print(i, "Exception")
            if e.error_code == MPI.ERR_SPAWN:
                time.sleep(10)
                exception = e
                continue
            raise
    else:
        raise exception
    comm.Barrier()
    duration = time.time() - t_start
    print("Started {} workers in {:.3}s".format(n_workers, duration))
    return comm
