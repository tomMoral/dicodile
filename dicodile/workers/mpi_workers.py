"""Start and shutdown MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
import os
import sys
import time
import numpy as np
from mpi4py import MPI

from ..utils import constants
from ..utils import debug_flags as flags


SYSTEM_HOSTFILE = os.environ.get("MPI_HOSTFILE", None)


# Constants to start interactive workers
INTERACTIVE_EXEC = "xterm"
INTERACTIVE_ARGS = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


class MPIWorkers:
    def __init__(self, n_workers, hostfile):
        self.comm = _spawn_workers(n_workers, hostfile)
        self.n_workers = n_workers
        self.hostfile = hostfile
        self.shutdown = False

    def __del__(self):
        if not self.shutdown:
            self.shutdown_workers()

    def send_command(self, tag, verbose=0):
        """Send a command (tag) to the workers.

        Parameters
        ----------
        tag : int
            Command tag to send.
        verbose : int
            If > 5, print a trace message.

        See Also
        --------
        dicodile.constants : tag constant definitions
        """
        msg = np.empty(1, dtype='i')
        msg[0] = tag
        t_start = time.time()
        for i_worker in range(self.n_workers):
            self.comm.Send([msg, MPI.INT], dest=i_worker, tag=tag)
        if verbose > 5:
            print("Sent message {} in {:.3f}s".format(
                tag, time.time() - t_start))

    def shutdown_workers(self):
        """Shut down workers.
        """
        if not self.shutdown:
            self.send_command(constants.TAG_WORKER_STOP)
            self.comm.Barrier()
            self.comm.Disconnect()
            MPI.COMM_SELF.Barrier()
            self.shutdown = True


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
