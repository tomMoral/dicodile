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
from dicodile.utils.debugs import worker_check_beta
from dicodile.utils.segmentation import Segmentation
from dicodile.utils.mpi import recv_broadcasted_array
from dicodile.utils.csc import compute_ztz, compute_ztX
from dicodile.utils.shape_helpers import get_full_support
from dicodile.utils.order_iterator import get_order_iterator
from dicodile.utils.dictionary import D_shape, compute_DtD
from dicodile.utils.dictionary import get_max_error_patch
from dicodile.utils.dictionary import norm_atoms_from_DtD_reshaped

from dicodile.update_z.coordinate_descent import _select_coordinate
from dicodile.update_z.coordinate_descent import _check_convergence
from dicodile.update_z.coordinate_descent import _init_beta, coordinate_update


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
        self.D = None

    def run(self):
        self.recv_task()
        self.compute_z_hat()
        self.send_result()

    def recv_task(self):
        # Retrieve the parameter of the algorithm
        self.recv_params()

        # Retrieve the dictionary used for coding
        self.D = self.recv_D()

        # Retrieve the signal to encode
        self.X_worker, self.z0 = self.recv_signal()

    def compute_z_hat(self):

        # compute the number of coordinates
        n_atoms, *_ = D_shape(self.D)
        seg_in_support = self.workers_segments.get_seg_support(
            self.rank, inner=True
        )
        n_coordinates = n_atoms * np.prod(seg_in_support)

        # Initialization of the algorithm variables
        rng = check_random_state(self.random_state)
        order = None
        if self.strategy in ['cyclic', 'cyclic-r', 'random']:
            offset = np.r_[0, self.local_segments.inner_bounds[:, 0]]
            order = get_order_iterator(
                (n_atoms, *seg_in_support), strategy=self.strategy,
                random_state=rng, offset=offset
            )

        i_seg = -1
        dz = 1
        n_coordinate_updates = 0
        accumulator = 0
        k0, pt0 = 0, None
        self.n_paused_worker = 0
        t_local_init = self.init_cd_variables()

        diverging = False
        if flags.INTERACTIVE_PROCESSES and self.n_workers == 1:
            import ipdb; ipdb.set_trace()  # noqa: E702

        self.t_start = t_start = time.time()
        t_run = 0
        t_select_coord, t_update_coord = [], []
        if self.timeout is not None:
            deadline = t_start + self.timeout
        else:
            deadline = None

        for ii in range(self.max_iter):
            # Display the progress of the algorithm
            self.progress(ii, max_ii=self.max_iter, unit="iterations",
                          extra_msg=abs(dz))

            # Process incoming messages
            self.process_messages()

            # Increment the segment and select the coordinate to update
            i_seg = self.local_segments.increment_seg(i_seg)
            if self.local_segments.is_active_segment(i_seg):
                t_start_selection = time.time()
                k0, pt0, dz = _select_coordinate(
                    self.dz_opt, self.dE, self.local_segments, i_seg,
                    strategy=self.strategy, order=order)
                selection_duration = time.time() - t_start_selection
                t_select_coord.append(selection_duration)
                t_run += selection_duration
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
                    self.rank, pt0, np.array(self.overlap) + 1
                )
                # Only soft lock in the corners
                if len(lock_slices) > n_lock:
                    max_on_lock = max([
                        abs(self.dz_opt[u_slice]).max()
                        for u_slice in lock_slices
                    ])
                    soft_locked = max_on_lock > abs(dz)

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance and is contained in the
            # worker. If the update is not in the worker, this will
            # effectively work has a soft lock to prevent interferences.
            if abs(dz) > self.tol and not soft_locked:
                t_start_update = time.time()

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

                # Logging of the time and the cost function if necessary
                update_duration = time.time() - t_start_update
                n_coordinate_updates += 1
                t_run += update_duration
                t_update_coord.append(update_duration)

                if self.timing:
                    self._log_updates.append((t_run, ii, self.rank,
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
                        self.debug("LGCD converged with {} iterations ({} "
                                   "updates)", ii + 1, n_coordinate_updates)
                        break
                # else:
                #     time.sleep(.001)

            # Check if we reach the timeout
            if deadline is not None and time.time() >= deadline:
                self.stop_before_convergence(
                    "Reached timeout", ii + 1, n_coordinate_updates
                )
                break
        else:
            self.stop_before_convergence(
                "Reached max_iter", ii + 1, n_coordinate_updates
            )

        self.synchronize_workers(with_main=True)
        assert diverging or self.check_no_transitting_message()
        runtime = time.time() - t_start

        if flags.CHECK_FINAL_BETA:
            worker_check_beta(
                self.rank, self.workers_segments, self.beta, D_shape(self.D)
            )

        t_select_coord = np.mean(t_select_coord)
        t_update_coord = (np.mean(t_update_coord) if len(t_update_coord) > 0
                          else None)
        self.return_run_statistics(
            ii=ii, t_run=t_run, n_coordinate_updates=n_coordinate_updates,
            runtime=runtime, t_local_init=t_local_init,
            t_select_coord=t_select_coord, t_update_coord=t_update_coord
        )

    def stop_before_convergence(self, msg, ii, n_coordinate_updates):
        self.info("{}. Done {} iterations ({} updates). Max of |dz|={}.",
                  msg, ii, n_coordinate_updates, abs(self.dz_opt).max())
        self.wait_status_changed(status=constants.STATUS_FINISHED)

    def init_cd_variables(self):
        t_start = time.time()

        # Pre-compute some quantities
        constants = {}
        if self.precomputed_DtD:
            constants['DtD'] = self.DtD
        else:
            constants['DtD'] = compute_DtD(self.D)

        n_atoms, _, *atom_support = D_shape(self.D)
        constants['norm_atoms'] = norm_atoms_from_DtD_reshaped(
            constants['DtD'],
            n_atoms,
            atom_support
        )
        self.constants = constants

        # List of all pending messages sent
        self.messages = []

        # Log all updates for logging purpose
        self._log_updates = []

        # Avoid printing progress too often
        self._last_progress = 0

        if self.warm_start and hasattr(self, 'z_hat'):
            self.z0 = self.z_hat.copy()

        # Initialization of the auxillary variable for LGCD
        self.beta, self.dz_opt, self.dE = _init_beta(
            self.X_worker, self.D, self.reg, z_i=self.z0, constants=constants,
            z_positive=self.z_positive, return_dE=self.strategy == "gs-q"
        )

        # Make sure all segments are activated
        self.local_segments.reset()

        if self.z0 is not None:
            self.freezed_support = None
            self.z_hat = self.z0.copy()
            self.correct_beta_z0()
        else:
            self.z_hat = np.zeros(self.beta.shape)

        if flags.CHECK_WARM_BETA:
            worker_check_beta(self.rank, self.workers_segments, self.beta,
                              D_shape(self.D))

        if self.freeze_support:
            assert self.z0 is not None
            self.freezed_support = self.z0 == 0
            self.dz_opt[self.freezed_support] = 0
        else:
            self.freezed_support = None

        self.synchronize_workers(with_main=False)

        t_local_init = time.time() - t_start
        self.debug("End local initialization in {:.2f}s", t_local_init,
                   global_msg=True)

        self.info("Start DICOD with {} workers, strategy '{}', soft_lock"
                  "={} and n_seg={}({})", self.n_workers, self.strategy,
                  self.soft_lock, self.n_seg,
                  self.local_segments.effective_n_seg, global_msg=True)
        return t_local_init

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
                assert self.n_paused_worker <= self.n_workers
            elif tag == constants.TAG_DICOD_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0

            msg = np.empty(self.size_msg, 'd')
            MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src, tag=tag)

            if tag == constants.TAG_DICOD_UPDATE_BETA:
                self.message_update_beta(msg)

        if self.n_paused_worker == self.n_workers:
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
                assert self.n_paused_worker <= self.n_workers
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
                self.progress(self.n_paused_worker, max_ii=self.n_workers,
                              unit="done workers")

        if self.rank == 0 and status == constants.STATUS_STOP:
            for i_worker in range(1, self.n_workers):
                self.notify_worker_status(constants.TAG_DICOD_STOP, i_worker,
                                          wait=True)
        elif status == constants.STATUS_RUNNING:
            self.debug("wake up")
        else:
            assert status == constants.STATUS_STOP
        return status

    def compute_sufficient_statistics(self):
        _, _, *atom_support = D_shape(self.D)
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

        padding_support = self.workers_segments.get_padding_to_overlap(
            self.rank)
        ztz = compute_ztz(self.z_hat, atom_support,
                          padding_support=padding_support)
        return np.array(ztz, dtype='d'), np.array(ztX, dtype='d')

    def correct_beta_z0(self):
        # Send coordinate updates to neighbors for all nonzero coordinates in
        # z0
        msg_send, msg_recv = [0] * self.n_workers, [0] * self.n_workers
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
            if n_init_done == self.n_workers:
                for i_worker in range(1, self.n_workers):
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

        if not hasattr(self, 'z_hat'):
            v = self.X_worker[X_hat_slice]
            return .5 * np.dot(v.ravel(), v.ravel())

        X_hat_worker = reconstruct(self.z_hat, self.D)
        diff = (X_hat_worker[X_hat_slice] - self.X_worker[X_hat_slice]).ravel()
        cost = .5 * np.dot(diff, diff)
        return cost + self.reg * abs(self.z_hat[inner_slice]).sum()

    def _get_z_hat(self):
        if flags.GET_OVERLAP_Z_HAT:
            res_slice = (Ellipsis,)
        else:
            res_slice = (Ellipsis,) + tuple([
                slice(start, end)
                for start, end in self.local_segments.inner_bounds
            ])
        return self.z_hat[res_slice].ravel()

    def return_z_hat(self):
        self.return_array(self._get_z_hat())

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

    def return_run_statistics(self, ii, n_coordinate_updates, runtime,
                              t_local_init, t_run, t_select_coord,
                              t_update_coord):
        """Return the # of iteration, the init and the run time for this worker
        """
        arr = [ii, n_coordinate_updates, runtime, t_local_init, t_run,
               t_select_coord, t_update_coord]
        self.gather_array(arr)

    def compute_and_return_max_error_patch(self):
        # receive window param
        # cutting through abstractions here, refactor if needed
        assert self._backend == "mpi"
        comm = MPI.Comm.Get_parent()
        params = comm.bcast(None, root=0)
        assert 'window' in params

        _, _, *atom_support = self.D.shape

        max_error_patch, max_error = get_max_error_patch(
            self.X_worker, self.z_hat, self.D, window=params['window'],
            local_segments=self.local_segments
        )
        self.gather_array([max_error_patch, max_error])

    ###########################################################################
    #     Display utilities
    ###########################################################################

    def progress(self, ii, max_ii, unit, extra_msg=None):
        t_progress = time.time()
        if t_progress - self._last_progress < 1:
            return
        if extra_msg is None:
            extra_msg = ''
        else:
            extra_msg = f"({extra_msg})"
        self._last_progress = t_progress
        self._log("{:.0f}s - progress : {:7.2%} {} {}",
                  t_progress - self.t_start, ii / max_ii, unit,
                  extra_msg, level=1, level_name="PROGRESS",
                  global_msg=True, endline=False)

    def info(self, msg, *fmt_args, global_msg=False, **fmt_kwargs):
        self._log(msg, *fmt_args, level=2, level_name="INFO",
                  global_msg=global_msg, **fmt_kwargs)

    def debug(self, msg, *fmt_args, global_msg=False, **fmt_kwargs):
        self._log(msg, *fmt_args, level=10, level_name="DEBUG",
                  global_msg=global_msg, **fmt_kwargs)

    def _log(self, msg, *fmt_args, level=0, level_name="None",
             global_msg=False, endline=True, **fmt_kwargs):
        if self.verbose >= level:
            if global_msg:
                if self.rank != 0:
                    return
                msg_fmt = constants.GLOBAL_OUTPUT_TAG + msg
                identity = self.n_workers
            else:
                msg_fmt = constants.WORKER_OUTPUT_TAG + msg
                identity = self.rank
            if endline:
                kwargs = {}
            else:
                kwargs = dict(end='', flush=True)
            msg_fmt = msg_fmt.ljust(80)
            print(msg_fmt.format(*fmt_args, identity=identity,
                                 level_name=level_name, **fmt_kwargs,),
                  **kwargs)

    ###########################################################################
    #     Communication primitives
    ###########################################################################

    def synchronize_workers(self, with_main=True):
        """Wait for all the workers to reach this point before continuing

        If main is True, this synchronization must also be called in the main
        program.
        """
        if self._backend == "mpi":
            self._synchronize_workers_mpi(with_main=with_main)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def recv_params(self):
        """Receive the parameter of the algorithm from the master node."""
        if self._backend == "mpi":
            self.rank, self.n_workers, params = self._recv_params_mpi()
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
        self.warm_start = params['warm_start']
        self.freeze_support = params['freeze_support']
        self.precomputed_DtD = params['precomputed_DtD']
        self.rank1 = params['rank1']

        # Set the random_state and add salt to avoid collapse between workers
        if not hasattr(self, 'random_state'):
            self.random_state = params['random_state']
            if isinstance(self.random_state, int):
                self.random_state += self.rank

        self.debug("tol updated to {:.2e}", self.tol, global_msg=True)
        return params

    def recv_D(self):
        """Receive a dictionary D"""
        comm = MPI.Comm.Get_parent()

        previous_D_shape = D_shape(self.D) if self.D is not None else None
        if self.rank1:
            self.u = recv_broadcasted_array(comm)
            self.v = recv_broadcasted_array(comm)
            self.D = (self.u, self.v)
        else:
            self.D = recv_broadcasted_array(comm)

        if self.precomputed_DtD:
            self.DtD = recv_broadcasted_array(comm)

        # update z if the shape of D changed (when adding new atoms)
        if (previous_D_shape is not None and
                previous_D_shape != D_shape(self.D)):
            self._extend_z()

        # update overlap if necessary
        _, _, *atom_support = D_shape(self.D)
        self.overlap = np.array(atom_support) - 1

        return self.D

    def _extend_z(self):
        """
        When adding new atoms in D, add the corresponding
        number of (zero-valued) rows in z
        """
        # Only extend z_hat if it has already been created.
        if not hasattr(self, "z_hat"):
            return

        if self.rank1:
            d_shape = D_shape(self.D)
        else:
            d_shape = self.D.shape
        n_new_atoms = d_shape[0] - self.z_hat.shape[0]
        assert n_new_atoms > 0, "cannot decrease the number of atoms"

        self.z_hat = np.concatenate([
            self.z_hat,
            np.zeros((n_new_atoms, *self.z_hat.shape[1:]))
        ], axis=0)

    def recv_signal(self):

        n_atoms, n_channels, *atom_support = D_shape(self.D)

        comm = MPI.Comm.Get_parent()
        X_info = comm.bcast(None, root=0)
        self.has_z0 = X_info['has_z0']
        self.valid_support = X_info['valid_support']
        self.workers_topology = X_info['workers_topology']
        self.size_msg = len(self.workers_topology) + 2

        self.workers_segments = Segmentation(
            n_seg=self.workers_topology,
            signal_support=self.valid_support,
            overlap=self.overlap
        )

        # Receive X and z from the master node.
        worker_support = self.workers_segments.get_seg_support(self.rank)
        X_shape = (n_channels,) + get_full_support(worker_support,
                                                   atom_support)
        z0_shape = (n_atoms,) + worker_support
        if self.has_z0:
            z0 = self.recv_array(z0_shape)
        else:
            z0 = None
        X_worker = self.recv_array(X_shape)

        # Compute the local segmentation for LGCD algorithm

        # If n_seg is not specified, compute the shape of the local segments
        # as the size of an interfering zone.
        n_atoms, _, *atom_support = D_shape(self.D)
        n_seg = self.n_seg
        local_seg_support = None
        if self.n_seg == 'auto':
            n_seg = None
            local_seg_support = 2 * np.array(atom_support) - 1

        # Get local inner bounds. First, compute the seg_bound without overlap
        # in local coordinates and then convert the bounds in the local
        # coordinate system.
        inner_bounds = self.workers_segments.get_seg_bounds(
            self.rank, inner=True)
        inner_bounds = np.transpose([
            self.workers_segments.get_local_coordinate(self.rank, bound)
            for bound in np.transpose(inner_bounds)])

        worker_support = self.workers_segments.get_seg_support(self.rank)
        self.local_segments = Segmentation(
            n_seg=n_seg, seg_support=local_seg_support,
            inner_bounds=inner_bounds,
            full_support=worker_support)

        self.max_iter *= self.local_segments.effective_n_seg
        self.synchronize_workers(with_main=True)

        return X_worker, z0

    def recv_array(self, shape):
        """Receive the part of the signal to encode from the master node."""
        if self._backend == "mpi":
            return self._recv_array_mpi(shape=shape)
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

    def send_result(self):
        if self._backend == "mpi":
            self._send_result_mpi()
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

    def gather_array(self, arr):
        if self._backend == "mpi":
            self._gather_array_mpi(arr)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def shutdown(self):
        if self._backend == "mpi":
            from ..utils.mpi import shutdown_mpi
            shutdown_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    ###########################################################################
    #     mpi4py implementation
    ###########################################################################

    def _synchronize_workers_mpi(self, with_main=True):
        if with_main:
            comm = MPI.Comm.Get_parent()
        else:
            comm = MPI.COMM_WORLD
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

    def _recv_params_mpi(self):
        comm = MPI.Comm.Get_parent()

        rank = comm.Get_rank()
        n_workers = comm.Get_size()
        params = comm.bcast(None, root=0)
        return rank, n_workers, params

    def _send_message_mpi(self, msg, tag, i_worker, wait=False):
        if wait:
            return MPI.COMM_WORLD.Ssend([msg, MPI.DOUBLE], i_worker, tag=tag)
        else:
            return MPI.COMM_WORLD.Issend([msg, MPI.DOUBLE], i_worker, tag=tag)

    def _send_result_mpi(self):
        comm = MPI.Comm.Get_parent()
        self.info("Reducing the distributed results", global_msg=True)

        self.return_z_hat()

        if self.return_ztz:
            self.return_sufficient_statistics()

        self.return_cost()

        if self.timing:
            comm.send(self._log_updates, dest=0)

        comm.Barrier()

    def _recv_array_mpi(self, shape):
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()

        arr = np.empty(shape, dtype='d')
        comm.Recv([arr.ravel(), MPI.DOUBLE], source=0,
                  tag=constants.TAG_ROOT + rank)
        return arr

    def _return_array_mpi(self, arr):
        comm = MPI.Comm.Get_parent()
        arr.astype('d')
        comm.Send([arr, MPI.DOUBLE], dest=0,
                  tag=constants.TAG_ROOT + self.rank)

    def _reduce_sum_array_mpi(self, arr):
        comm = MPI.Comm.Get_parent()
        arr = np.array(arr, dtype='d')
        comm.Reduce([arr, MPI.DOUBLE], None, op=MPI.SUM, root=0)

    def _gather_array_mpi(self, arr):
        comm = MPI.Comm.Get_parent()
        comm.gather(arr, root=0)


if __name__ == "__main__":
    dicod = DICODWorker(backend='mpi')
    dicod.run()
    dicod.shutdown()
