"""
Program to load and visualize scenes of a dataset. See parser arguments for details.

Example of running the program:
python -m causalorca.bin.visualize_scenes --data_pickle_path data/synth_v1.a.tiny.pkl --logs_path data/visualizations/synth_v1.a.tiny/
"""
import argparse
import os
import pickle
import sys

from tqdm import tqdm

from causalorca.utils.plots import plot_summary_figures, \
    plot_per_scene_figures
from causalorca.utils.util import ensure_dir, redirect_stdout_and_stderr_to_file, nice_print, HORSE, \
    get_str_formatted_time


def main(args):
    with open(args.data_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    scenes = dataset["scenes"]
    config = dataset["config"]

    # Summary visualizations for all scenes
    plot_summary_figures(scenes, config, args.logs_path, usetex=not args.no_latex)
    if args.summary_only:
        return

    # Per scene visualizations
    scene_index_start = 0 if args.scene_index_start is None else args.scene_index_start
    scene_index_end = len(scenes) - 1 if args.scene_index_end is None else args.scene_index_end
    for i, scene in enumerate(tqdm(range(scene_index_start, scene_index_end + 1))):
        plot_per_scene_figures(i, scenes[i], config, args.logs_path, usetex=not args.no_latex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pickle_path', type=str, required=True,
                        help='Path to save logs')
    parser.add_argument('--logs_path', type=str, default=None,
                        help='Path to save the generated visualizations into. '
                             'If None, then the vizualizations are saved into '
                             'the directory where the data is located.')
    parser.add_argument('--no_latex', action='store_true',
                        help='Whether not to use latex when creating plots.'
                             ' Some systems might not have a tex distribution installed.')
    parser.add_argument('--summary_only', action='store_true',
                        help='Whether to create only summary visualisations,'
                             ' omitting the per-scene visualisations'
                             ' that might take a long time to generate')
    parser.add_argument('--scene_index_start', type=int, default=None,
                        help='Index of first scene for pre-scene visualisations.')
    parser.add_argument('--scene_index_end', type=int, default=None,
                        help='Index of last scene for pre-scene visualisations.')
    args = parser.parse_args()
    if args.logs_path is None:
        args.logs_path = os.path.dirname(args.data_pickle_path)
    ensure_dir(args.logs_path)

    redirect_stdout_and_stderr_to_file(os.path.join(args.logs_path, f"output__{get_str_formatted_time()}.txt"))

    nice_print(HORSE)
    print(f"args={args}")
    main(args)

    sys.stdout.flush()
    sys.stderr.flush()
