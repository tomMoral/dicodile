"""Worker for the distributed algorithm DICOD

Author : tommoral <thomas.moreau@inria.fr>
"""

import time
import numpy as np
from mpi4py import MPI

from dicodile.utils.csc import reconstruct
from dicodile.utils import check_random_state
from dicodile.utils import debug_flags as flags
from dicodile.utils import constants as constants
from dicodile.utils.segmentation import Segmentation
from dicodile.utils.mpi import recv_broadcasted_array
from dicodile.utils.csc import compute_ztz, compute_ztX
from dicodile.utils.shape_helpers import get_full_shape
from dicodile.utils.dictionary import compute_DtD, compute_norm_atoms

from dicodile.coordinate_descent import _select_coordinate
from dicodile.coordinate_descent import _check_convergence
from dicodile.coordinate_descent import _init_beta, coordinate_update


class DICODWorker:
    """Worker for DICOD, running LGCD locally and using MPI for communications

    Parameters
    ----------
    backend: str
        Backend used to communicate between workers. Available backends are
        { 'mpi' }.
    """

    def __init__(self, backend):
        self._backend = backend

    def recv_task(self):
        # Retrieve different constants from the base communicator and store
        # then in the class.
        params = self.get_params()

        if self.timeout:
            self.timeout *= 3

        self.random_state = params['random_state']
        if isinstance(self.random_state, int):
            self.random_state += self.rank

        self.size_msg = len(params['valid_shape']) + 2

        # Compute the shape of the worker segment.
        self.D = self.get_D()
        n_atoms, n_channels, *atom_support = self.D.shape
        self.overlap = np.array(atom_support) - 1
        self.workers_segments = Segmentation(
            n_seg=params['workers_topology'],
            signal_shape=params['valid_shape'],
            overlap=self.overlap)

        # Receive X and z from the master node.
        worker_shape = self.workers_segments.get_seg_shape(self.rank)
        X_shape = (n_channels,) + get_full_shape(worker_shape, atom_support)
        if params['has_z0']:
            z0_shape = (n_atoms,) + worker_shape
            self.z0 = self.get_signal(z0_shape, params['debug'])
        else:
            self.z0 = None
        self.X_worker = self.get_signal(X_shape, params['debug'])

        # Compute the local segmentation for LGCD algorithm

        # If n_seg is not specified, compute the shape of the local segments
        # as the size of an interfering zone.
        n_seg = self.n_seg
        local_seg_shape = None
        if self.n_seg == 'auto':
            n_seg = None
            local_seg_shape = 2 * np.array(atom_support) - 1

        # Get local inner bounds. First, compute the seg_bound without overlap
        # in local coordinates and then convert the bounds in the local
        # coordinate system.
        inner_bounds = self.workers_segments.get_seg_bounds(
            self.rank, inner=True)
        inner_bounds = np.transpose([
            self.workers_segments.get_local_coordinate(self.rank, bound)
            for bound in np.transpose(inner_bounds)])

        self.local_segments = Segmentation(
            n_seg=n_seg, seg_shape=local_seg_shape, inner_bounds=inner_bounds,
            full_shape=worker_shape)

        # Initialize the solution
        n_atoms = self.D.shape[0]
        seg_shape = self.workers_segments.get_seg_shape(self.rank)
        if self.z0 is None:
            self.z_hat = np.zeros((n_atoms,) + seg_shape)
        else:
            self.z_hat = self.z0

        self.info("Start DICOD with {} workers, strategy '{}', soft_lock"
                  "={} and n_seg={}({})", self.n_jobs, self.strategy,
                  self.soft_lock, self.n_seg,
                  self.local_segments.effective_n_seg, global_msg=True)

        self.synchronize_workers()

    def compute_z_hat(self):

        # Initialization of the algorithm variables
        random_state = check_random_state(self.random_state)
        i_seg = -1
        n_coordinate_updates = 0
        accumulator = 0
        k0, pt0 = 0, None
        self.n_paused_worker = 0

        # compute the number of coordinates
        n_atoms, *_ = self.D.shape
        seg_in_shape = self.workers_segments.get_seg_shape(
            self.rank, inner=True)
        n_coordinates = n_atoms * np.prod(seg_in_shape)

        self.init_cd_variables()

        diverging = False
        if flags.INTERACTIVE_PROCESSES and self.n_jobs == 1:
            import ipdb; ipdb.set_trace()  # noqa: E702
        self.t_start = t_start = time.time()
        if self.timeout is not None:
            deadline = t_start + self.timeout
        else:
            deadline = None
        for ii in range(self.max_iter):
            # Display the progress of the algorithm
            self.progress(ii, max_ii=self.max_iter, unit="iterations")

            # Process incoming messages
            self.process_messages()

            # Increment the segment and select the coordinate to update
            try:
                i_seg = self.local_segments.increment_seg(i_seg)
            except ZeroDivisionError:
                print(self.local_segments.signal_shape,
                      self.local_segments.n_seg_per_axis)
                raise
            if self.local_segments.is_active_segment(i_seg):
                k0, pt0, dz = _select_coordinate(
                    self.dz_opt, self.dE, self.local_segments, i_seg,
                    strategy=self.strategy, random_state=random_state)

                assert self.workers_segments.is_contained_coordinate(
                    self.rank, pt0, inner=True), pt0
            else:
                k0, pt0, dz = None, None, 0

            # update the accumulator for 'random' strategy
            accumulator = max(abs(dz), accumulator)

            # If requested, check that the update chosen only have an impact on
            # the segment and its overlap area.
            if flags.CHECK_UPDATE_CONTAINED and pt0 is not None:
                self.workers_segments.check_area_contained(self.rank,
                                                           pt0, self.overlap)

            # Check if the coordinate is soft-locked or not.
            soft_locked = False
            if (pt0 is not None and abs(dz) > self.tol and
                    self.soft_lock != 'none'):
                n_lock = 1 if self.soft_lock == "corner" else 0
                lock_slices = self.workers_segments.get_touched_overlap_slices(
                    self.rank, pt0, np.array(self.overlap) + 1)
                # Only soft lock in the corners
                if len(lock_slices) > n_lock:
                    max_on_lock = 0
                    for u_slice in lock_slices:
                        max_on_lock = max(abs(self.dz_opt[u_slice]).max(),
                                          max_on_lock)
                    soft_locked = max_on_lock > abs(dz)

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance and is contained in the
            # worker. If the update is not in the worker, this will
            # effectively work has a soft lock to prevent interferences.
            if abs(dz) > self.tol and not soft_locked:
                n_coordinate_updates += 1

                # update the selected coordinate and beta
                self.coordinate_update(k0, pt0, dz)

                # Notify neighboring workers of the update if needed.
                pt_global = self.workers_segments.get_global_coordinate(
                    self.rank, pt0)
                workers = self.workers_segments.get_touched_segments(
                    pt=pt_global, radius=np.array(self.overlap) + 1
                )
                msg = np.array([k0, *pt_global, dz], 'd')

                self.notify_neighbors(msg, workers)

                if self.timing:
                    t_update = time.time() - t_start
                    self._log_updates.append((t_update, ii, self.rank,
                                              k0, pt_global, dz))

            # Inactivate the current segment if the magnitude of the update is
            # too small. This only work when using LGCD.
            if abs(dz) <= self.tol and self.strategy == "greedy":
                self.local_segments.set_inactive_segments(i_seg)

            # When workers are diverging, finish the worker to avoid having to
            # wait until max_iter for stopping the algorithm.
            if abs(dz) >= 1e3:
                self.info("diverging worker")
                self.wait_status_changed(status=constants.STATUS_FINISHED)
                diverging = True
                break

            # Check the stopping criterion and if we have locally converged,
            # wait either for an incoming message or for full convergence.
            if _check_convergence(self.local_segments, self.tol, ii,
                                  self.dz_opt, n_coordinates, self.strategy,
                                  accumulator=accumulator):

                if flags.CHECK_ACTIVE_SEGMENTS:
                    inner_slice = (Ellipsis,) + tuple([
                        slice(start, end)
                        for start, end in self.local_segments.inner_bounds
                    ])
                    assert np.all(abs(self.dz_opt[inner_slice]) <= self.tol)
                if self.check_no_transitting_message():
                    status = self.wait_status_changed()
                    if status == constants.STATUS_STOP:
                        self.debug("LGCD converged with {} coordinate "
                                   "updates", n_coordinate_updates)
                        break

            # Check is we reach the timeout
            if deadline is not None and time.time() >= deadline:
                self.stop_before_convergence("Reached timeout",
                                             n_coordinate_updates)
                break
        else:
            self.stop_before_convergence("Reached max_iter",
                                         n_coordinate_updates)

        self.synchronize_workers()
        assert diverging or self.check_no_transitting_message()
        runtime = time.time() - t_start

        comm = MPI.Comm.Get_parent()
        comm.gather([n_coordinate_updates, runtime], root=0)

        return n_coordinate_updates, runtime

    def run(self):
        self.recv_task()
        n_coordinate_updates, runtime = self.compute_z_hat()
        self.send_result(n_coordinate_updates, runtime)

    def stop_before_convergence(self, msg, n_coordinate_updates):
        self.info("{}. Done {} coordinate updates. Max of |dz|={}.",
                  msg, n_coordinate_updates, abs(self.dz_opt).max())
        self.wait_status_changed(status=constants.STATUS_FINISHED)

    def init_cd_variables(self):
        t_start = time.time()

        # Pre-compute some quantities
        constants = {}
        constants['norm_atoms'] = compute_norm_atoms(self.D)
        constants['DtD'] = compute_DtD(self.D)
        self.constants = constants

        # List of all pending messages sent
        self.messages = []

        # Log all updates for logging purpose
        self._log_updates = []

        # Avoid printing progress too often
        self._last_progress = 0

        # Initialization of the auxillary variable for LGCD
        return_dE = self.strategy == "gs-q"
        self.beta, self.dz_opt, self.dE = _init_beta(
            self.X_worker, self.D, self.reg, z_i=self.z0, constants=constants,
            z_positive=self.z_positive, return_dE=return_dE)

        # Make sure all segments are activated
        self.local_segments.reset()

        if self.z0 is not None:
            self.freezed_support = None
            self.correct_beta_z0()

        if flags.CHECK_WARM_BETA:
            pt_global = self.workers_segments.get_seg_shape(0, inner=True)
            pt = self.workers_segments.get_local_coordinate(self.rank,
                                                            pt_global)
            if self.workers_segments.is_contained_coordinate(self.rank, pt):
                _, _, *atom_support = self.D.shape
                beta_slice = (Ellipsis,) + tuple([
                    slice(v - size_ax + 1, v + size_ax - 1)
                    for v, size_ax in zip(pt, atom_support)
                ])
                sum_beta = np.array(self.beta[beta_slice].sum(), dtype='d')
                self.return_array(sum_beta)

        if self.freeze_support:
            assert self.z0 is not None
            self.freezed_support = self.z0 == 0
            self.dz_opt[self.freezed_support] = 0
        else:
            self.freezed_support = None

        self.synchronize_workers()

        t_local_init = time.time() - t_start
        self.info("End local initialization in {:.2f}s", t_local_init,
                  global_msg=True)

    def coordinate_update(self, k0, pt0, dz, coordinate_exist=True):
        self.beta, self.dz_opt, self.dE = coordinate_update(
            k0, pt0, dz, beta=self.beta, dz_opt=self.dz_opt, dE=self.dE,
            z_hat=self.z_hat, D=self.D, reg=self.reg, constants=self.constants,
            z_positive=self.z_positive, freezed_support=self.freezed_support,
            coordinate_exist=coordinate_exist)

        # Re-activate the segments where beta have been updated to ensure
        # convergence.
        touched_segments = self.local_segments.get_touched_segments(
            pt=pt0, radius=self.overlap)
        n_changed_status = self.local_segments.set_active_segments(
            touched_segments)

        # If requested, check that all inactive segments have no coefficients
        # to update over the tolerance.
        if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
            self.local_segments.test_active_segments(
                self.dz_opt, self.tol)

    def process_messages(self, worker_status=constants.STATUS_RUNNING):
        mpi_status = MPI.Status()
        while MPI.COMM_WORLD.Iprobe(status=mpi_status):
            src = mpi_status.source
            tag = mpi_status.tag
            if tag == constants.TAG_DICOD_UPDATE_BETA:
                if worker_status == constants.STATUS_PAUSED:
                    self.notify_worker_status(
                        constants.TAG_DICOD_RUNNING_WORKER, wait=True)
                    worker_status = constants.STATUS_RUNNING
            elif tag == constants.TAG_DICOD_STOP:
                worker_status = constants.STATUS_STOP
            elif tag == constants.TAG_DICOD_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == constants.TAG_DICOD_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0

            msg = np.empty(self.size_msg, 'd')
            MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src, tag=tag)

            if tag == constants.TAG_DICOD_UPDATE_BETA:
                self.message_update_beta(msg)

        if self.n_paused_worker == self.n_jobs:
            worker_status = constants.STATUS_STOP
        return worker_status

    def message_update_beta(self, msg):
        k0, *pt_global, dz = msg

        k0 = int(k0)
        pt_global = tuple([int(v) for v in pt_global])
        pt0 = self.workers_segments.get_local_coordinate(self.rank, pt_global)
        assert not self.workers_segments.is_contained_coordinate(
            self.rank, pt0, inner=True), (pt_global, pt0)
        coordinate_exist = self.workers_segments.is_contained_coordinate(
            self.rank, pt0, inner=False)
        self.coordinate_update(k0, pt0, dz, coordinate_exist=coordinate_exist)

        if flags.CHECK_BETA and np.random.rand() > 0.99:
            # Only check beta 1% of the time to avoid the check being too long
            inner_slice = (Ellipsis,) + tuple([
                slice(start, end)
                for start, end in self.local_segments.inner_bounds
            ])
            beta, *_ = _init_beta(
                self.X_worker, self.D, self.reg, z_i=self.z_hat,
                constants=self.constants, z_positive=self.z_positive)
            assert np.allclose(beta[inner_slice], self.beta[inner_slice])

    def notify_neighbors(self, msg, neighbors):
        assert self.rank in neighbors
        for i_neighbor in neighbors:
            if i_neighbor != self.rank:
                req = self.send_message(msg, constants.TAG_DICOD_UPDATE_BETA,
                                        i_neighbor, wait=False)
                self.messages.append(req)

    def notify_worker_status(self, tag, i_worker=0, wait=False):
        # handle the messages from Worker0 to himself.
        if self.rank == 0 and i_worker == 0:
            if tag == constants.TAG_DICOD_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == constants.TAG_DICOD_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0
            elif tag == constants.TAG_DICOD_INIT_DONE:
                pass
            else:
                raise ValueError("Got tag {}".format(tag))
            return

        # Else send the message to the required destination
        msg = np.empty(self.size_msg, 'd')
        self.send_message(msg, tag, i_worker, wait=wait)

    def wait_status_changed(self, status=constants.STATUS_PAUSED):
        if status == constants.STATUS_FINISHED:
            # Make sure to flush the messages
            while not self.check_no_transitting_message():
                self.process_messages(worker_status=status)
                time.sleep(0.001)

        self.notify_worker_status(constants.TAG_DICOD_PAUSED_WORKER)
        self.debug("paused worker")

        # Wait for all sent message to be processed
        count = 0
        while status not in [constants.STATUS_RUNNING, constants.STATUS_STOP]:
            time.sleep(.005)
            status = self.process_messages(worker_status=status)
            if (count % 500) == 0:
                self.progress(self.n_paused_worker, max_ii=self.n_jobs,
                              unit="done workers")

        if self.rank == 0 and status == constants.STATUS_STOP:
            for i_worker in range(1, self.n_jobs):
                self.notify_worker_status(constants.TAG_DICOD_STOP, i_worker,
                                          wait=True)
        elif status == constants.STATUS_RUNNING:
            self.debug("wake up")
        else:
            assert status == constants.STATUS_STOP
        return status

    def compute_sufficient_statistics(self):
        _, _, *atom_support = self.D.shape
        z_slice = (Ellipsis,) + tuple([
            slice(start, end)
            for start, end in self.local_segments.inner_bounds
        ])
        X_slice = (Ellipsis,) + tuple([
            slice(start, end + size_atom_ax - 1)
            for (start, end), size_atom_ax in zip(
                self.local_segments.inner_bounds, atom_support)
        ])

        ztX = compute_ztX(self.z_hat[z_slice], self.X_worker[X_slice])

        padding_shape = self.workers_segments.get_padding_to_overlap(self.rank)
        ztz = compute_ztz(self.z_hat, atom_support,
                          padding_shape=padding_shape)
        return np.array(ztz, dtype='d'), np.array(ztX, dtype='d')

    def correct_beta_z0(self):
        # Send coordinate updates to neighbors for all nonzero coordinates in
        # z0
        msg_send, msg_recv = [0] * self.n_jobs, [0] * self.n_jobs
        for k0, *pt0 in zip(*self.z0.nonzero()):
            # Notify neighboring workers of the update if needed.
            pt_global = self.workers_segments.get_global_coordinate(
                self.rank, pt0)
            workers = self.workers_segments.get_touched_segments(
                pt=pt_global, radius=np.array(self.overlap) + 1
            )
            msg = np.array([k0, *pt_global, self.z0[(k0, *pt0)]], 'd')
            self.notify_neighbors(msg, workers)
            for i in workers:
                msg_send[i] += 1

        n_init_done = 0
        done_pt = set()
        no_msg, init_done = False, False
        mpi_status = MPI.Status()
        while not init_done:
            if n_init_done == self.n_jobs:
                for i_worker in range(1, self.n_jobs):
                    self.notify_worker_status(constants.TAG_DICOD_INIT_DONE,
                                              i_worker=i_worker)
                init_done = True
            if not no_msg:
                if self.check_no_transitting_message(check_incoming=False):
                    self.notify_worker_status(constants.TAG_DICOD_INIT_DONE)
                    if self.rank == 0:
                        n_init_done += 1
                    assert len(self.messages) == 0
                    no_msg = True

            if MPI.COMM_WORLD.Iprobe(status=mpi_status):
                tag = mpi_status.tag
                src = mpi_status.source
                if tag == constants.TAG_DICOD_INIT_DONE:
                    if self.rank == 0:
                        n_init_done += 1
                    else:
                        init_done = True

                msg = np.empty(self.size_msg, 'd')
                MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src, tag=tag)

                if tag == constants.TAG_DICOD_UPDATE_BETA:
                    msg_recv[src] += 1
                    k0, *pt_global, dz = msg
                    k0 = int(k0)
                    pt_global = tuple([int(v) for v in pt_global])
                    pt0 = self.workers_segments.get_local_coordinate(self.rank,
                                                                     pt_global)
                    pt_exist = self.workers_segments.is_contained_coordinate(
                        self.rank, pt0, inner=False)
                    if not pt_exist and (k0, *pt0) not in done_pt:
                        done_pt.add((k0, *pt0))
                        self.coordinate_update(k0, pt0, dz,
                                               coordinate_exist=False)

            else:
                time.sleep(.001)

    def compute_cost(self):
        X_hat_worker = reconstruct(self.z_hat, self.D)
        inner_bounds = self.local_segments.inner_bounds
        inner_slice = tuple([Ellipsis] + [
            slice(start_ax, end_ax) for start_ax, end_ax in inner_bounds])
        X_hat_slice = list(inner_slice)
        i_seg = self.rank
        ax_rank_offset = self.workers_segments.effective_n_seg
        for ax, n_seg_ax in enumerate(self.workers_segments.n_seg_per_axis):
            ax_rank_offset //= n_seg_ax
            ax_i_seg = i_seg // ax_rank_offset
            i_seg % ax_rank_offset
            if (ax_i_seg + 1) % n_seg_ax == 0:
                s = inner_slice[ax + 1]
                X_hat_slice[ax + 1] = slice(s.start, None)
        X_hat_slice = tuple(X_hat_slice)
        diff = (X_hat_worker[X_hat_slice] - self.X_worker[X_hat_slice]).ravel()
        cost = .5 * np.dot(diff, diff)
        return cost + self.reg * abs(self.z_hat[inner_slice]).sum()

    def return_z_hat(self):
        if flags.GET_OVERLAP_Z_HAT:
            res_slice = (Ellipsis,)
        else:
            res_slice = (Ellipsis,) + tuple([
                slice(start, end)
                for start, end in self.local_segments.inner_bounds
            ])
        z_worker = self.z_hat[res_slice].ravel()
        self.return_array(z_worker)

    def return_z_nnz(self):
        res_slice = (Ellipsis,) + tuple([
            slice(start, end)
            for start, end in self.local_segments.inner_bounds
        ])
        z_nnz = self.z_hat[res_slice] != 0
        z_nnz = np.sum(z_nnz, axis=tuple(range(1, z_nnz.ndim)))
        self.reduce_sum_array(z_nnz)

    def return_sufficient_statistics(self):
        ztz, ztX = self.compute_sufficient_statistics()
        self.reduce_sum_array(ztz)
        self.reduce_sum_array(ztX)

    def return_cost(self):
        cost = self.compute_cost()
        cost = np.array(cost, dtype='d')
        self.reduce_sum_array(cost)

    ###########################################################################
    #     Display utilities
    ###########################################################################

    def progress(self, ii, max_ii, unit):
        t_progress = time.time()
        if t_progress - self._last_progress < 1:
            return
        self._last_progress = t_progress
        self._log("{:.0f}s - progress : {:7.2%} {}",
                  time.time() - self.t_start,
                  ii / max_ii, unit, level=1, level_name="PROGRESS",
                  global_msg=True, endline=False)

    def info(self, msg, *fmt_args, global_msg=False, **fmt_kwargs):
        self._log(msg, *fmt_args, level=1, level_name="INFO",
                  global_msg=global_msg, **fmt_kwargs)

    def debug(self, msg, *fmt_args, global_msg=False, **fmt_kwargs):
        self._log(msg, *fmt_args, level=5, level_name="DEBUG",
                  global_msg=global_msg, **fmt_kwargs)

    def _log(self, msg, *fmt_args, level=0, level_name="None",
             global_msg=False, endline=True, **fmt_kwargs):
        if self.verbose >= level:
            if global_msg:
                if self.rank != 0:
                    return
                msg_fmt = constants.GLOBAL_OUTPUT_TAG + msg
                identity = self.n_jobs
            else:
                msg_fmt = constants.WORKER_OUTPUT_TAG + msg
                identity = self.rank
            if endline:
                kwargs = {}
            else:
                kwargs = {'end': '', 'flush': True}
            msg_fmt = msg_fmt.ljust(80)
            print(msg_fmt.format(identity, level_name, *fmt_args,
                                 **fmt_kwargs,), **kwargs)

    ###########################################################################
    #     Communication primitives
    ###########################################################################

    def synchronize_workers(self):
        if self._backend == "mpi":
            self._synchronize_workers_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def get_params(self):
        """Receive the parameter of the algorithm from the master node."""
        if self._backend == "mpi":
            self.rank, self.n_jobs, params = self._get_params_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

        self.tol = params['tol']
        self.reg = params['reg']
        self.n_seg = params['n_seg']
        self.timing = params['timing']
        self.timeout = params['timeout']
        self.verbose = params['verbose']
        self.strategy = params['strategy']
        self.max_iter = params['max_iter']
        self.soft_lock = params['soft_lock']
        self.z_positive = params['z_positive']
        self.return_ztz = params['return_ztz']
        self.freeze_support = params['freeze_support']

        self.info("tol updated to {:.2e}", self.tol, global_msg=True)
        return params

    def get_signal(self, X_shape, debug=False):
        """Receive the part of the signal to encode from the master node."""
        if self._backend == "mpi":
            return self._get_signal_mpi(X_shape, debug=debug)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def send_message(self, msg, tag, i_worker, wait=False):
        """Send a message to a specified worker."""
        assert self.rank != i_worker
        if self._backend == "mpi":
            return self._send_message_mpi(msg, tag, i_worker, wait=wait)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def send_result(self, iterations, runtime):
        if self._backend == "mpi":
            self._send_result_mpi(iterations, runtime)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def get_D(self):
        """Receive a dictionary D"""
        if self._backend == "mpi":
            comm = MPI.Comm.Get_parent()
            self.D = recv_broadcasted_array(comm)
            return self.D
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def return_array(self, sig):
        if self._backend == "mpi":
            self._return_array_mpi(sig)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def reduce_sum_array(self, arr):
        if self._backend == "mpi":
            self._reduce_sum_array_mpi(arr)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def shutdown(self):
        if self._backend == "mpi":
            self._shutdown_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    ###########################################################################
    #     mpi4py implementation
    ###########################################################################

    def _synchronize_workers_mpi(self):
        comm = MPI.Comm.Get_parent()
        comm.Barrier()

    def check_no_transitting_message(self, check_incoming=True):
        """Check no message is in waiting to complete to or from this worker"""
        if check_incoming and MPI.COMM_WORLD.Iprobe():
            return False
        while self.messages:
            if not self.messages[0].Test() or (
                    check_incoming and MPI.COMM_WORLD.Iprobe()):
                return False
            self.messages.pop(0)
        assert len(self.messages) == 0, len(self.messages)
        return True

    def _get_params_mpi(self):
        comm = MPI.Comm.Get_parent()

        rank = comm.Get_rank()
        n_jobs = comm.Get_size()
        params = comm.bcast(None, root=0)
        return rank, n_jobs, params

    def _get_signal_mpi(self, sig_shape, debug):
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()

        sig_worker = np.empty(sig_shape, dtype='d')
        comm.Recv([sig_worker.ravel(), MPI.DOUBLE], source=0,
                  tag=constants.TAG_ROOT + rank)

        if debug:
            X_alpha = 0.25 * np.ones(sig_shape)
            self.return_array(X_alpha)

        return sig_worker

    def _send_message_mpi(self, msg, tag, i_worker, wait=False):
        if wait:
            return MPI.COMM_WORLD.Ssend([msg, MPI.DOUBLE], i_worker, tag=tag)
        else:
            return MPI.COMM_WORLD.Issend([msg, MPI.DOUBLE], i_worker, tag=tag)

    def _send_result_mpi(self, iterations, runtime):
        comm = MPI.Comm.Get_parent()
        self.info("Reducing the distributed results", global_msg=True)

        self.return_z_hat()

        if self.return_ztz:
            self.return_sufficient_statistics()

        self.return_cost()

        if self.timing:
            comm.send(self._log_updates, dest=0)

        comm.Barrier()

    def _return_array_mpi(self, arr):
        comm = MPI.Comm.Get_parent()
        arr.astype('d')
        comm.Send([arr, MPI.DOUBLE], dest=0,
                  tag=constants.TAG_ROOT + self.rank)

    def _reduce_sum_array_mpi(self, arr):
        comm = MPI.Comm.Get_parent()
        arr = np.array(arr, dtype='d')
        comm.Reduce([arr, MPI.DOUBLE], None, op=MPI.SUM, root=0)

    def _shutdown_mpi(self):
        comm = MPI.Comm.Get_parent()
        self.debug("clean shutdown")
        comm.Barrier()
        comm.Disconnect()


if __name__ == "__main__":
    dicod = DICODWorker(backend='mpi')
    dicod.run()
    dicod.shutdown()
