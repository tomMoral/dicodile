import pytest
import numpy as np

from dicodile.utils.segmentation import Segmentation


def test_segmentation_coverage():
    sig_support = (108, 53)

    for h_seg in [5, 7, 9, 13, 17]:
        for w_seg in [3, 11]:
            z = np.zeros(sig_support)
            segments = Segmentation(n_seg=(h_seg, w_seg),
                                    signal_support=sig_support)
            assert tuple(segments.n_seg_per_axis) == (h_seg, w_seg)
            seg_slice = segments.get_seg_slice(0)
            seg_support = segments.get_seg_support(0)
            assert seg_support == z[seg_slice].shape
            z[seg_slice] += 1
            i_seg = segments.increment_seg(0)
            while i_seg != 0:
                seg_slice = segments.get_seg_slice(i_seg)
                seg_support = segments.get_seg_support(i_seg)
                assert seg_support == z[seg_slice].shape
                z[seg_slice] += 1
                i_seg = segments.increment_seg(i_seg)

            assert np.all(z == 1)

    z = np.zeros(sig_support)
    inner_bounds = [(8, 100), (3, 50)]
    inner_slice = tuple([slice(start, end) for start, end in inner_bounds])
    segments = Segmentation(n_seg=7, inner_bounds=inner_bounds,
                            full_support=sig_support)
    for i_seg in range(segments.effective_n_seg):
        seg_slice = segments.get_seg_slice(i_seg)
        z[seg_slice] += 1

    assert np.all(z[inner_slice] == 1)
    z[inner_slice] = 0
    assert np.all(z == 0)


def test_segmentation_coverage_overlap():
    sig_support = (505, 407)

    for overlap in [(3, 0), (0, 5), (3, 5), (12, 7)]:
        for h_seg in [5, 7, 9, 13, 15, 17]:
            for w_seg in [3, 11]:
                segments = Segmentation(n_seg=(h_seg, w_seg),
                                        signal_support=sig_support,
                                        overlap=overlap)
                z = np.zeros(sig_support)
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg, inner=True)
                    z[seg_slice] += 1
                    i_seg = segments.increment_seg(i_seg)
                non_overlapping = np.prod(sig_support)
                assert np.sum(z == 1) == non_overlapping

                z = np.zeros(sig_support)
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg)
                    z[seg_slice] += 1
                    i_seg = segments.increment_seg(i_seg)

                h_ov, w_ov = overlap
                h_seg, w_seg = segments.n_seg_per_axis
                expected_overlap = ((h_seg - 1) * sig_support[1] * 2 * h_ov)
                expected_overlap += ((w_seg - 1) * sig_support[0] * 2 * w_ov)

                # Compute the number of pixel where there is more than 2
                # segments overlappping.
                corner_overlap = 4 * (h_seg - 1) * (w_seg - 1) * h_ov * w_ov
                expected_overlap -= 2 * corner_overlap

                non_overlapping -= expected_overlap + corner_overlap
                assert non_overlapping == np.sum(z == 1)
                assert expected_overlap == np.sum(z == 2)
                assert corner_overlap == np.sum(z == 4)


def test_touched_segments():
    """Test detection of touched segments and records of active segments
    """
    rng = np.random.RandomState(42)

    H, W = sig_support = (108, 53)
    n_seg = (9, 3)
    for h_radius in [5, 7, 9]:
        for w_radius in [3, 11]:
            for _ in range(20):
                h0 = rng.randint(-h_radius, sig_support[0] + h_radius)
                w0 = rng.randint(-w_radius, sig_support[1] + w_radius)
                z = np.zeros(sig_support)
                segments = Segmentation(n_seg, signal_support=sig_support)

                touched_slice = (
                    slice(max(0, h0 - h_radius), min(H, h0 + h_radius + 1)),
                    slice(max(0, w0 - w_radius), min(W, w0 + w_radius + 1))
                )
                z[touched_slice] = 1

                touched_segments = segments.get_touched_segments(
                    (h0, w0), (h_radius, w_radius))
                segments.set_inactive_segments(touched_segments)
                n_active_segments = segments._n_active_segments

                expected_n_active_segments = segments.effective_n_seg
                for i_seg in range(segments.effective_n_seg):
                    seg_slice = segments.get_seg_slice(i_seg)
                    is_touched = np.any(z[seg_slice] == 1)
                    expected_n_active_segments -= is_touched

                    assert segments.is_active_segment(i_seg) != is_touched
                assert n_active_segments == expected_n_active_segments

    # Check an error is returned when touched radius is larger than seg_size
    segments = Segmentation(n_seg, signal_support=sig_support)
    with pytest.raises(ValueError, match="too large"):
        segments.get_touched_segments((0, 0), (30, 2))


def test_change_coordinate():
    sig_support = (505, 407)
    overlap = (12, 7)
    n_seg = (4, 4)
    segments = Segmentation(n_seg=n_seg, signal_support=sig_support,
                            overlap=overlap)

    for i_seg in range(segments.effective_n_seg):
        seg_bound = segments.get_seg_bounds(i_seg)
        seg_support = segments.get_seg_support(i_seg)
        origin = tuple([start for start, _ in seg_bound])
        assert segments.get_global_coordinate(i_seg, (0, 0)) == origin
        assert segments.get_local_coordinate(i_seg, origin) == (0, 0)

        corner = tuple([end for _, end in seg_bound])
        assert segments.get_global_coordinate(i_seg, seg_support) == corner
        assert segments.get_local_coordinate(i_seg, corner) == seg_support


def test_inner_coordinate():
    sig_support = (505, 407)
    overlap = (11, 11)
    n_seg = (4, 4)
    segments = Segmentation(n_seg=n_seg, signal_support=sig_support,
                            overlap=overlap)

    for h_rank in range(n_seg[0]):
        for w_rank in range(n_seg[1]):
            i_seg = h_rank * n_seg[1] + w_rank
            seg_support = segments.get_seg_support(i_seg)
            assert segments.is_contained_coordinate(i_seg, overlap,
                                                    inner=True)

            if h_rank == 0:
                assert segments.is_contained_coordinate(i_seg, (0, overlap[1]),
                                                        inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (overlap[0] - 1, overlap[1]), inner=True)

            if w_rank == 0:
                assert segments.is_contained_coordinate(i_seg, (overlap[0], 0),
                                                        inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (overlap[0], overlap[1] - 1), inner=True)

            if h_rank == 0 and w_rank == 0:
                assert segments.is_contained_coordinate(i_seg, (0, 0),
                                                        inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (overlap[0] - 1, overlap[1] - 1), inner=True)

            if h_rank == n_seg[0] - 1:
                assert segments.is_contained_coordinate(
                    i_seg,
                    (seg_support[0] - 1, seg_support[1] - overlap[1] - 1),
                    inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (seg_support[0] - overlap[0],
                            seg_support[1] - overlap[1] - 1), inner=True)

            if w_rank == n_seg[1] - 1:
                assert segments.is_contained_coordinate(
                    i_seg,
                    (seg_support[0] - overlap[0] - 1, seg_support[1] - 1),
                    inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (seg_support[0] - overlap[0] - 1,
                            seg_support[1] - overlap[1]), inner=True)

            if h_rank == n_seg[0] - 1 and w_rank == n_seg[1] - 1:
                assert segments.is_contained_coordinate(
                    i_seg, (seg_support[0] - 1, seg_support[1] - 1),
                    inner=True)
            else:
                assert not segments.is_contained_coordinate(
                    i_seg, (seg_support[0] - overlap[0],
                            seg_support[1] - overlap[1]), inner=True)


def test_touched_overlap_area():
    sig_support = (505, 407)
    overlap = (11, 9)
    n_seg = (8, 4)
    segments = Segmentation(n_seg=n_seg, signal_support=sig_support,
                            overlap=overlap)

    for i_seg in range(segments.effective_n_seg):
        seg_support = segments.get_seg_support(i_seg)
        seg_slice = segments.get_seg_slice(i_seg)
        seg_inner_slice = segments.get_seg_slice(i_seg, inner=True)
        if i_seg != 0:
            with pytest.raises(AssertionError):
                segments.check_area_contained(i_seg, (0, 0), overlap)
        for pt0 in [overlap, (overlap[0], 25), (25, overlap[1]), (25, 25),
                    (seg_support[0] - overlap[0] - 1, 25),
                    (25, seg_support[1] - overlap[1] - 1),
                    (seg_support[0] - overlap[0] - 1,
                     seg_support[1] - overlap[1] - 1)
                    ]:
            assert segments.is_contained_coordinate(i_seg, pt0, inner=True)
            segments.check_area_contained(i_seg, pt0, overlap)
            z = np.zeros(sig_support)
            pt_global = segments.get_global_coordinate(i_seg, pt0)
            update_slice = tuple([
                slice(max(v - r, 0), v + r + 1)
                for v, r in zip(pt_global, overlap)])

            z[update_slice] += 1
            z[seg_inner_slice] = 0

            # The returned slice are given in local coordinates. Take the
            # segment in z to use local coordinate.
            z_seg = z[seg_slice]

            updated_slices = segments.get_touched_overlap_slices(i_seg, pt0,
                                                                 overlap)
            # Assert that all selected coordinate are indeed in the update area
            for u_slice in updated_slices:
                assert np.all(z_seg[u_slice] == 1)

            # Assert that all coordinate updated in the overlap area have been
            # selected with at least one slice.
            for u_slice in updated_slices:
                z_seg[u_slice] *= 0
            assert np.all(z == 0)


def test_padding_to_overlap():
    n_seg = (4, 4)
    sig_support = (504, 504)
    overlap = (12, 7)

    seg = Segmentation(n_seg=n_seg, signal_support=sig_support,
                       overlap=overlap)
    seg_support_all = seg.get_seg_support(n_seg[1] + 1)
    for i_seg in range(np.prod(n_seg)):
        seg_support = seg.get_seg_support(i_seg)
        z = np.empty(seg_support)
        overlap = seg.get_padding_to_overlap(i_seg)
        z = np.pad(z, overlap, mode='constant')
        assert z.shape == seg_support_all


def test_segments():
    """Tests if the number of segments is computed correctly."""
    seg_support = [9]
    inner_bounds = [[0, 252]]
    full_support = (252,)

    seg = Segmentation(n_seg=None, seg_support=seg_support,
                       inner_bounds=inner_bounds, full_support=full_support)
    seg.compute_n_seg()

    assert seg.effective_n_seg == 28

    seg_support = [10]
    seg = Segmentation(n_seg=None, seg_support=seg_support,
                       inner_bounds=inner_bounds, full_support=full_support)
    seg.compute_n_seg()

    assert seg.effective_n_seg == 26
