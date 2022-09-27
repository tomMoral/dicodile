import numpy as np


class Segmentation:
    """Segmentation of a multi-dimensional signal and utilities to navigate it.

    Parameters
    ----------
    n_seg : int or list of int
        Number of segments to use for each dimension. If only one int is
        given, use this same number for all axis.
    signal_support : list of int or None
        Size of the considered signal.
    inner_bounds : list of (int, int) or None
        Outer boundaries of the full signal in case of nested segmentation.
    full_support : list of int or None
        Full shape of the underlying signals
    """

    def __init__(self, n_seg=None, seg_support=None, signal_support=None,
                 inner_bounds=None, full_support=None, overlap=None):

        # Get the shape of the signal from signal_support or inner_bounds
        if inner_bounds is not None:
            signal_support_ = [v[0] for v in np.diff(inner_bounds, axis=1)]
            if signal_support is not None:
                assert signal_support == signal_support_, (
                    "Incoherent shape for inner_bounds and signal_support. Got"
                    " signal_support={} and inner_bounds={}".format(
                        signal_support, inner_bounds
                    ))
            signal_support = signal_support_
        else:
            assert signal_support is not None, (
                "either signal_support or inner_bounds should be provided")
            if isinstance(signal_support, int):
                signal_support = [signal_support]
            inner_bounds = [[0, s] for s in signal_support]
        self.signal_support = signal_support
        self.inner_bounds = inner_bounds
        self.n_axis = len(signal_support)

        if full_support is None:
            full_support = [end for _, end in self.inner_bounds]
        self.full_support = full_support
        assert np.all([size_full_ax >= end
                       for size_full_ax, (_, end) in zip(self.full_support,
                                                         self.inner_bounds)])

        # compute the size of each segment and the number of segments
        if seg_support is not None:
            if isinstance(seg_support, int):
                seg_support = [seg_support] * self.n_axis
            self.seg_support = tuple(seg_support)
            self.compute_n_seg()
        elif n_seg is not None:
            if isinstance(n_seg, int):
                n_seg = [n_seg] * self.n_axis
            self.n_seg_per_axis = tuple(n_seg)
            self.compute_seg_support()

        # Validate the overlap
        if overlap is None:
            self.overlap = [0] * self.n_axis
        elif isinstance(overlap, int):
            self.overlap = [overlap] * self.n_axis
        else:
            assert np.iterable(overlap)
            self.overlap = overlap

        # Initializes variable to keep track of active segments
        self._n_active_segments = self.effective_n_seg
        self._active_segments = [True] * self.effective_n_seg

        # Validate the Segmentation
        if n_seg is not None:
            assert tuple(n_seg) == self.n_seg_per_axis
        if seg_support is not None:
            assert tuple(seg_support) == self.seg_support

    def compute_n_seg(self):
        """Compute the number of segment for each axis based on their shapes.
        """
        self.effective_n_seg = 1
        self.n_seg_per_axis = []
        for size_ax, size_seg_ax in zip(self.signal_support, self.seg_support):
            # Make sure that n_seg_ax is of type int (and not np.int*)
            n_seg_ax = max(1, int(size_ax // size_seg_ax) +
                           ((size_ax % size_seg_ax) != 0))
            self.n_seg_per_axis.append(n_seg_ax)
            self.effective_n_seg *= n_seg_ax

    def compute_seg_support(self):
        """Compute the number of segment for each axis based on their shapes.
        """
        self.effective_n_seg = 1
        self.seg_support = []
        for size_ax, n_seg_ax in zip(self.signal_support, self.n_seg_per_axis):
            # Make sure that n_seg_ax is of type int (and not np.int*)
            size_seg_ax = size_ax // n_seg_ax
            size_seg_ax += (size_ax % n_seg_ax >= n_seg_ax // 2)
            self.seg_support.append(size_seg_ax)
            self.effective_n_seg *= n_seg_ax

    def get_seg_bounds(self, i_seg, inner=False):
        """Return a segment's boundaries."""

        seg_bounds = []
        ax_offset = self.effective_n_seg
        for (n_seg_ax, size_seg_ax, size_full_ax,
             (start_in_ax, end_in_ax), overlap_ax) in zip(
                self.n_seg_per_axis, self.seg_support, self.full_support,
                self.inner_bounds, self.overlap):
            ax_offset //= n_seg_ax
            ax_i_seg = i_seg // ax_offset
            ax_bound_start = start_in_ax + ax_i_seg * size_seg_ax
            ax_bound_end = ax_bound_start + size_seg_ax
            if (ax_i_seg + 1) % n_seg_ax == 0:
                ax_bound_end = end_in_ax
            if not inner:
                ax_bound_end = min(ax_bound_end + overlap_ax, size_full_ax)
                ax_bound_start = max(ax_bound_start - overlap_ax, 0)
            seg_bounds.append([ax_bound_start, ax_bound_end])
            i_seg %= ax_offset
        return seg_bounds

    def get_seg_slice(self, i_seg, inner=False):
        """Return a segment's slice"""
        seg_bounds = self.get_seg_bounds(i_seg, inner=inner)
        return (Ellipsis,) + tuple([slice(s, e) for s, e in seg_bounds])

    def get_seg_support(self, i_seg, inner=False):
        """Return a segment's shape"""
        seg_bounds = self.get_seg_bounds(i_seg, inner=inner)
        return tuple(np.diff(seg_bounds, axis=1).squeeze(axis=1))

    def find_segment(self, pt):
        """Find the indice of the segment containing the given point.

        If the point is not contained in the segmentation boundaries, return
        the indice of the closest segment in manhattan distance.

        Parameter
        ---------
        pt : list of int
            Coordinate of the given update.

        Return
        ------
        i_seg : int
            Indices of the segment containing pt or the closest one in
            manhattan distance if pt is out of range.
        """
        assert len(pt) == self.n_axis
        i_seg = 0
        axis_offset = self.effective_n_seg
        for x, n_seg_axis, size_seg_axis, (axis_start, axis_end) in zip(
                pt, self.n_seg_per_axis, self.seg_support, self.inner_bounds):
            axis_offset //= n_seg_axis
            axis_i_seg = max(min((x - axis_start) // size_seg_axis,
                                 n_seg_axis - 1), 0)
            i_seg += axis_i_seg * axis_offset

        return i_seg

    def increment_seg(self, i_seg):
        """Return the next segment indice in a cyclic way."""
        return (i_seg + 1) % self.effective_n_seg

    def get_touched_segments(self, pt, radius):
        """Return all segments touched by an update in pt with a given radius.

        Parameter
        ---------
        pt : list of int
            Coordinate of the given update.
        radius: int or list of int
            Radius of the update. If an integer is given, use the same integer
            for all axis.

        Return
        ------
        segments : list of int
            Indices of all segments touched by this update, including the one
            in which the update took place.
        """
        assert len(pt) == self.n_axis
        if isinstance(radius, int):
            radius = [radius] * self.n_axis

        for r, size_axis in zip(radius, self.seg_support):
            if r >= size_axis:
                raise ValueError("Interference radius is too large compared "
                                 "to the segmentation size.")

        i_seg = self.find_segment(pt)
        seg_bounds = self.get_seg_bounds(i_seg, inner=True)

        segments = [i_seg]
        axis_offset = self.effective_n_seg
        for x, r, n_seg_axis, (axis_start, axis_end), overlap_ax in zip(
                pt, radius, self.n_seg_per_axis, seg_bounds, self.overlap):
            axis_offset //= n_seg_axis
            axis_i_seg = i_seg // axis_offset
            i_seg %= axis_offset
            new_segments = []
            if x - r < axis_start + overlap_ax and axis_i_seg > 0:
                new_segments.extend([n - axis_offset for n in segments])
            if (x + r >= axis_start - overlap_ax or
                    x - r < axis_end + overlap_ax):
                new_segments.extend([n for n in segments])
            if x + r >= axis_end - overlap_ax and axis_i_seg < n_seg_axis - 1:
                new_segments.extend([n + axis_offset for n in segments])
            segments = new_segments

        for ii_seg in segments:
            msg = ("Segment indice out of bound. Got {} for effective n_seg {}"
                   .format(ii_seg, self.effective_n_seg))
            assert ii_seg < self.effective_n_seg, msg

        return segments

    def is_active_segment(self, i_seg):
        """Return True if segment i_seg is active"""
        return self._active_segments[i_seg]

    def set_active_segments(self, indices):
        """Activate segments indices and return the number of changed status.
        """
        if isinstance(indices, int):
            indices = [indices]

        n_changed_status = 0
        for i_seg in indices:
            n_changed_status += not self._active_segments[i_seg]
            self._active_segments[i_seg] = True

        self._n_active_segments += n_changed_status
        assert self._n_active_segments <= self.effective_n_seg

        return n_changed_status

    def set_inactive_segments(self, indices):
        """Deactivate segments indices and return the number of changed status.
        """
        if not np.iterable(indices):
            indices = [indices]

        n_changed_status = 0
        for i_seg in indices:
            n_changed_status += self._active_segments[i_seg]
            self._active_segments[i_seg] = False

        self._n_active_segments -= n_changed_status
        return self._n_active_segments >= 0

        return n_changed_status

    def exist_active_segment(self):
        """Return True if at least one segment is active."""
        return self._n_active_segments > 0

    def test_active_segments(self, dz, tol):
        """Test the state of active segments is coherent with dz and tol
        """
        for i in range(self.effective_n_seg):
            if not self.is_active_segment(i):
                seg_slice = self.get_seg_slice(i, inner=True)
                assert np.all(abs(dz[seg_slice]) <= tol)

    def get_global_coordinate(self, i_seg, pt):
        """Convert a point from local coordinate to global coordinate

        Parameters
        ----------
        pt: (int, int)
            Coordinate to convert, from the local coordinate system.

        Return
        ------
        pt : (int, int)
            Coordinate converted in the global coordinate system.
        """
        seg_bounds = self.get_seg_bounds(i_seg)
        res = []
        for v, (offset, _) in zip(pt, seg_bounds):
            res += [v + offset]
        return tuple(res)

    def get_local_coordinate(self, i_seg, pt):
        """Convert a point from global coordinate to local coordinate

        Parameters
        ----------
        pt: (int, int)
            Coordinate to convert, from the global coordinate system.

        Return
        ------
        pt : (int, int)
            Coordinate converted in the local coordinate system.
        """
        seg_bounds = self.get_seg_bounds(i_seg)
        res = []
        for v, (offset, _) in zip(pt, seg_bounds):
            res += [v - offset]
        return tuple(res)

    def is_contained_coordinate(self, i_seg, pt, inner=False):
        """Ensure that a given point is in the bounds to be a local coordinate.
        """
        seg_bounds = self.get_seg_bounds(i_seg, inner=inner)
        pt = self.get_global_coordinate(i_seg, pt)
        is_valid = True
        for v, (stat_ax, end_ax) in zip(pt, seg_bounds):
            is_valid &= (stat_ax <= v < end_ax)
        return is_valid

    def check_area_contained(self, i_seg, pt, radius):
        """Check that the given area is contained in segment i_seg.

        If not, fail with an AssertionError.
        """

        seg_bounds = self.get_seg_bounds(i_seg)
        seg_support = self.get_seg_support(i_seg)
        seg_bounds_inner = self.get_seg_bounds(i_seg, inner=True)

        update_bounds = [[v - r, v + r + 1] for v, r in zip(pt, radius)]
        assert self.is_contained_coordinate(i_seg, pt, inner=True)
        for i in range(self.n_axis):
            assert (update_bounds[i][0] >= 0 or
                    seg_bounds[i][0] == seg_bounds_inner[i][0])
            assert (update_bounds[i][1] <= seg_support[i]
                    or seg_bounds[i][1] == seg_bounds_inner[i][1])

    def get_touched_overlap_slices(self, i_seg, pt, radius):
        """Return a list of slices in the overlap area, touched a rectangle

        Parameter
        ---------
        i_seg : int
            Indice of the considered segment.
        pt : list of int
            Coordinate of the given update.
        radius: int or list of int
            Radius of the update. If an integer is given, use the same integer
            for all axis.

        Return
        ------
        touched_slices : list of slices
            Slices to select parts in the overlap area touched by the given
            area. The slices can have some overlap
        """
        seg_bounds = self.get_seg_bounds(i_seg)
        seg_support = self.get_seg_support(i_seg)
        seg_bounds_inner = self.get_seg_bounds(i_seg, inner=True)

        update_bounds = [[min(max(0, v - r), size_valid_ax),
                          max(min(v + r + 1, size_valid_ax), 0)]
                         for v, r, size_valid_ax in zip(pt, radius,
                                                        seg_support)]
        inner_bounds = [
            [start_in_ax - start_ax, end_in_ax - start_ax]
            for (start_ax, _), (start_in_ax, end_in_ax) in zip(
                seg_bounds, seg_bounds_inner)
        ]

        updated_slices = []
        pre_slice = (Ellipsis,)
        post_slice = tuple([slice(start, end)
                            for start, end in update_bounds[1:]])
        for (start, end), (start_inner, end_inner) in zip(
                update_bounds, inner_bounds):
            if start < start_inner:
                assert start_inner <= end <= end_inner
                updated_slices.append(
                    pre_slice + (slice(start, start_inner),) + post_slice
                )
            if end > end_inner:
                assert start_inner <= start <= end_inner
                updated_slices.append(
                    pre_slice + (slice(end_inner, end),) + post_slice
                )
            pre_slice = pre_slice + (slice(start, end),)
            post_slice = post_slice[1:]

        return updated_slices

    def get_padding_to_overlap(self, i_seg):

        seg_bounds = self.get_seg_bounds(i_seg)
        seg_inner_bounds = self.get_seg_bounds(i_seg, inner=True)
        padding_support = []
        for overlap_ax, (start_ax, end_ax), (start_in_ax, end_in_ax) in zip(
                self.overlap, seg_bounds, seg_inner_bounds):
            padding_support += [
                (overlap_ax - (start_in_ax - start_ax),
                 overlap_ax - (end_ax - end_in_ax))
            ]
        return padding_support

    def reset(self):
        # Re-activate all the segments
        self.set_active_segments(range(self.effective_n_seg))
