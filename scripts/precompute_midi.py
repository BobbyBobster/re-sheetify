"""Precompute piano-roll clips (one MIDI parse per piece). Run from repo root:
python scripts/precompute_midi.py
"""

import math
from pathlib import Path

import numpy as np
import asyncio
import warnings

import pretty_midi

import sheets.ml_logic.dataloaders as dl
from sheets.params import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from google.cloud import storage


BUCKET_NAME = "sheetify_bobbybobster"
DATA_ROOT = Path("./data")
OUT_ROOT = DATA_ROOT / "precomputed_rolls"


def roll_path(midi_path: str, clip_i: int) -> Path:
    rel = Path(midi_path).relative_to(DATA_ROOT / "midis").with_suffix("")
    return OUT_ROOT / rel.parent / f"{rel.name}_{clip_i:05d}.npy"


async def upload_file(blob, path, semaphore):
    async with semaphore:
        try:
            await asyncio.to_thread(blob.upload_from_filename, str(path))
            print(f"✅ Uploaded file: {path}")
        except Exception as e:
            print(f"❌ Failed to upload {path}: {e}")


async def async_main(bucket):
    semaphore = asyncio.Semaphore(10)
    tasks = []

    # for year in [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]:
    for year in [2018]:
        print(f"📥 Precomputing year {year}")
        for split in ("train", "validation", "test"):
            print(f"📥 Precomputing {split} splits")
            for pair in dl.get_pairs(DATA_ROOT, year_limit=year, split=split):
                n_clips = math.floor(pair["duration"] / CLIP_DURATION)
                roll = pretty_midi.PrettyMIDI(pair["midi_path"]).get_piano_roll(
                    fs=PIANO_ROLL_FS
                )
                for i in range(n_clips):
                    path = roll_path(pair["midi_path"], i)

                    if path.exists():
                        continue

                    start_sec = i * CLIP_DURATION
                    pianoroll = dl.slice_pianoroll(
                        roll=roll,
                        start_sec=start_sec,
                    )

                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, pianoroll)

                    blob = bucket.blob(str(path))
                    tasks.append(upload_file(blob, path, semaphore))

            await asyncio.gather(*tasks)
            tasks.clear()
        print("✅ done")


if __name__ == "__main__":
    bucket_name = BUCKET_NAME
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"{bucket_name=}")
    asyncio.run(async_main(bucket))
