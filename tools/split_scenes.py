"""
Script that takes a dataset as input and splits it into train, validation, and test subsets.

Example of usage:
python -m tools.split_scenes --input_data_pickle_path data/synth_v1.a.filtered.pkl \
 --output_train_path data/synth_v1.a.filtered.train.pkl \
 --output_val_path   data/synth_v1.a.filtered.val.pkl \
 --output_test_path  data/synth_v1.a.filtered.test.pkl \
 --val_ratio 0.2 \
 --test_ratio 0.2
"""
import argparse
import os.path
import pickle
from typing import List


def split_dataset(scenes: List, val_ratio: float, test_ratio: float):
    assert val_ratio > 0.0
    assert test_ratio > 0.0
    assert val_ratio + test_ratio < 1.0

    n = len(scenes)
    assert n > 0

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train = scenes[: n_train]
    val = scenes[n_train: n_train + n_val]
    test = scenes[-n_test:]
    assert n == len(train) + len(val) + len(test)

    return train, val, test


def main(args):
    print(f"Loading dataset from {args.input_data_pickle_path}...")
    with open(args.input_data_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    print("Dataset loaded")
    print("")

    scenes = dataset["scenes"]
    train_scenes, val_scenes, test_scenes = split_dataset(scenes, args.val_ratio, args.test_ratio)

    for subset_scenes, subset_path in [
        (train_scenes, args.output_train_path),
        (val_scenes, args.output_val_path),
        (test_scenes, args.output_test_path),
    ]:
        dataset["scenes"] = subset_scenes
        dataset["__split_scenes_args"] = args
        print(f"Saving {len(subset_scenes)} scenes to: {os.path.abspath(subset_path)}")
        with open(subset_path, "wb") as f:
            pickle.dump(dataset, f)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_pickle_path', type=str, required=True,
                        help='Path to pickle file of dataset to be split.')
    parser.add_argument('--output_train_path', type=str, default=True,
                        help='Path to pickle file to be outputted for the train subset.')
    parser.add_argument('--output_val_path', type=str, required=True,
                        help='Path to pickle file to be outputted for the validation subset.')
    parser.add_argument('--output_test_path', type=str, required=True,
                        help='Path to pickle file to be outputted for the test subset.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of scenes to be put in the validation subset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of scenes to be put in the test subset')
    args = parser.parse_args()
    print(f"args={args}")
    main(args)
