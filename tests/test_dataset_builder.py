import numpy as np

from sheets.ml_logic.dataset_builder import *
from sheets.params import *

class TestDatasetBuilder():
    def test_create_onf_roll(self):
        in_arr = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0]
            ])
        onset_arr = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0]
            ])
        frame_arr = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0]
            ])

        onset_roll, frame_roll = create_onf_roll(in_arr)
        assert np.array_equal(onset_roll, onset_arr)
        assert np.array_equal(frame_roll, frame_arr)
