"""Index and partition helpers shared by the timm ports."""

import numpy as np

from jimm.models import _maxxvit
from jimm.models.levit import _bias_index


def test_window_and_grid_partitions_invert_and_group_pixels():
    x = np.arange(2 * 14 * 14 * 3, dtype=np.float32).reshape(2, 14, 14, 3)
    size = (7, 7)
    windows = _maxxvit._window_partition(x, size)
    grids = _maxxvit._grid_partition(x, size)
    assert windows.shape == grids.shape == (8, 49, 3)
    np.testing.assert_array_equal(_maxxvit._window_reverse(windows, size, 14, 14), x)
    np.testing.assert_array_equal(_maxxvit._grid_reverse(grids, size, 14, 14), x)
    # Windows hold contiguous 7x7 tiles; grid partitions hold every second pixel.
    np.testing.assert_array_equal(windows[1].reshape(7, 7, 3), x[0, :7, 7:])
    np.testing.assert_array_equal(grids[1].reshape(7, 7, 3), x[0, ::2, 1::2])


def test_relative_position_index_matches_swin():
    expected = [[4, 3, 1, 0], [5, 4, 2, 1], [7, 6, 4, 3], [8, 7, 5, 4]]
    np.testing.assert_array_equal(_maxxvit._relative_position_index((2, 2)), expected)


def test_levit_bias_index_uses_absolute_offsets():
    index = _bias_index((3, 3))
    rows, cols = np.divmod(np.arange(9), 3)
    expected = np.abs(rows[:, None] - rows) * 3 + np.abs(cols[:, None] - cols)
    np.testing.assert_array_equal(index, expected)
    strided = _bias_index((3, 3), 2)
    assert strided.shape == (4, 9)
    np.testing.assert_array_equal(strided[1], expected[2])  # query (0, 2)
