import weakref
import numpy as np
from mpi4py import MPI

from ..utils import constants
from ..utils.csc import compute_objective
from ..workers.mpi_workers import MPIWorkers

from ..utils import debug_flags as flags
from ..utils.debugs import main_check_beta
from ..utils.shape_helpers import get_valid_support

from .dicod import recv_z_hat, recv_z_nnz
from .dicod import _gather_run_statistics
from .dicod import _send_task, _send_D, _send_signal
from .dicod import recv_cost, recv_sufficient_statistics


class DistributedSparseEncoder:
    def __init__(self, n_workers, w_world='auto', hostfile=None, verbose=0):
        # check the parameters
        if w_world != 'auto':
            assert n_workers % w_world == 0, (
                "`w_world={}` should divide the number of jobs `n_workers={}` "
                "used.".format(w_world, n_workers))

        # Store the parameters
        self.n_workers = n_workers
        self.w_world = w_world
        self.hostfile = hostfile
        self.verbose = verbose

    def init_workers(self, X, D_hat, reg, params, z0=None, DtD=None):

        # compute the partition for the signals
        assert D_hat.ndim - 1 == X.ndim, (D_hat.shape, X.shape)
        n_channels, *sig_support = X.shape
        n_atoms, n_channels, *atom_support = self.D_shape = D_hat.shape

        # compute effective n_workers to not have smaller worker support than
        # 4 times the atom_support
        valid_support = get_valid_support(sig_support, atom_support)
        max_n_workers = np.prod(np.maximum(
            1, np.array(valid_support) // (2 * np.array(atom_support))
        ))
        effective_n_workers = min(max_n_workers, self.n_workers)
        self.effective_n_workers = effective_n_workers

        # Create the workers with MPI
        self.workers = MPIWorkers(effective_n_workers, hostfile=self.hostfile)
        self.workers.send_command(constants.TAG_WORKER_RUN_DICODILE,
                                  verbose=self.verbose)

        w_world = self.w_world
        if self.w_world != 'auto' and self.w_world > effective_n_workers:
            w_world = effective_n_workers

        self.params = params.copy()
        self.params['reg'] = reg
        self.params['precomputed_DtD'] = DtD is not None
        self.params['verbose'] = self.verbose

        self.workers.send_command(constants.TAG_DICODILE_SET_TASK,
                                  verbose=self.verbose)
        self.t_init, self.workers_segments = _send_task(
            self.workers, X, D_hat, z0, DtD, w_world, self.params
        )

    def set_worker_D(self, D, DtD=None):
        msg = "The support of the dictionary cannot be changed on an encoder."
        assert D.shape[1:] == self.D_shape[1:], msg
        self.D_shape = D.shape

        if self.params['precomputed_DtD'] and DtD is None:
            raise ValueError("The pre-computed value DtD need to be passed "
                             "each time D is updated.")

        self.workers.send_command(constants.TAG_DICODILE_SET_D,
                                  verbose=self.verbose)
        _send_D(self.workers, D, DtD)

    def set_worker_params(self, params=None, **kwargs):
        if params is None:
            assert kwargs is not {}
            params = kwargs
        self.params.update(params)

        self.workers.send_command(constants.TAG_DICODILE_SET_PARAMS,
                                  verbose=self.verbose)
        self.workers.comm.bcast(self.params, root=MPI.ROOT)

    def set_worker_signal(self, X, z0=None):

        n_atoms, n_channels, *atom_support = self.D_shape
        if self.is_same_signal(X):
            return

        self.workers.send_command(constants.TAG_DICODILE_SET_SIGNAL,
                                  verbose=self.verbose)
        self.workers_segments = _send_signal(self.workers, self.w_world,
                                             atom_support, X, z0)
        self._ref_X = weakref.ref(X)

    def process_z_hat(self):
        self.workers.send_command(constants.TAG_DICODILE_COMPUTE_Z_HAT,
                                  verbose=self.verbose)

        if flags.CHECK_WARM_BETA:
            main_check_beta(self.workers.comm, self.workers_segments)

        # Then wait for the end of the computation
        self.workers.comm.Barrier()
        return _gather_run_statistics(self.workers.comm, self.workers_segments,
                                      verbose=self.verbose)

    def get_cost(self):
        self.workers.send_command(constants.TAG_DICODILE_GET_COST,
                                  verbose=self.verbose)
        return recv_cost(self.workers.comm)

    def get_z_hat(self):
        self.workers.send_command(constants.TAG_DICODILE_GET_Z_HAT,
                                  verbose=self.verbose)
        return recv_z_hat(self.workers.comm,
                          self.D_shape[0],
                          self.workers_segments)

    def get_z_nnz(self):
        self.workers.send_command(constants.TAG_DICODILE_GET_Z_NNZ,
                                  verbose=self.verbose)
        return recv_z_nnz(self.workers.comm, self.D_shape[0])

    def get_sufficient_statistics(self):
        self.workers.send_command(
            constants.TAG_DICODILE_GET_SUFFICIENT_STAT,
            verbose=self.verbose)
        return recv_sufficient_statistics(self.workers.comm, self.D_shape)

    def release_workers(self):
        self.workers.send_command(
            constants.TAG_DICODILE_STOP)

    def shutdown_workers(self):
        self.workers.shutdown_workers()

    def check_cost(self, X, D_hat, reg):
        cost = self.get_cost()
        z_hat = self.get_z_hat()
        cost_2 = compute_objective(X, z_hat, D_hat, reg)
        assert np.isclose(cost, cost_2), (cost, cost_2)
        print("check cost ok", cost, cost_2)

    def is_same_signal(self, X):
        if not hasattr(self, '_ref_X') or self._ref_X() is not X:
            return False
        return True
