import pandas as pd
import json

import pytest

from sheets.ml_logic.dataloaders import get_pairs
from sheets.params import *


@pytest.fixture(scope="class")
def load_metadata_json():
    print("\n[Setup] Loading metadata file...")
    metadata_path = DATA_PATH / "maestro-v3.0.0.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    with open(metadata_path) as fp:
        metadata = json.load(fp)
    yield metadata

@pytest.mark.usefixtures("load_metadata_json")
class TestDataloaders:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_get_pairs(self, load_metadata_json):
        """
        Verify that get_pairs filters the splits and years correctly.
        """
        metadata = load_metadata_json

        tcs = [
            {'split': 'train', 'year_limit': [2006, 2014]},
            {'split': 'validation', 'year_limit': [2018]},
            {'split': 'test', 'year_limit': None},
            {'split': 'train', 'year_limit': None},
            {'split': 'validation', 'year_limit': [2008]},
            {'split': 'test', 'year_limit': [1000]},
            {'split': 'train', 'year_limit': None, 'count_limit': 5},
            {'split': 'train', 'year_limit': [2006, 2008, 2014], 'count_limit': 50},
        ]

        meta_df = pd.DataFrame(metadata)

        for tc in tcs:
            dl = get_pairs(**tc)

            if tc["year_limit"] is None:
                tc["year_limit"] = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
            subset = meta_df[meta_df["split"] == tc["split"]]
            subset = subset[meta_df["year"].isin(tc["year_limit"])]
            pd_pairs = []
            for _, row in subset.iterrows():
                pd_pairs.append({
                    "audio_path": str(DATA_PATH / 'mp3s' / row['audio_filename'].replace('.wav', '.mp3')),
                    "midi_path": str(DATA_PATH / 'midis' / row["midi_filename"]),
                    "duration": row["duration"],
                })

            count_limit = tc.get("count_limit")
            if count_limit is not None:
                pd_pairs[:count_limit]

            assert dl == pd_pairs
