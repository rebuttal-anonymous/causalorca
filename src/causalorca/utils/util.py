from __future__ import annotations

import pathlib
import sys
from collections import namedtuple
from datetime import datetime

import numpy as np


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;cau;'. ;   |          )   \\ | ;;;;;;
          ,;*sal;\\/   |.        /   /` | ';;;*;
           ;;;ity/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class Logger:

    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


class FlushOnWriteLoggerWrapper(Logger):
    def __init__(self, logger: Logger):
        self.logger = logger

    def write(self, message):
        self.logger.write(message)
        self.flush()

    def flush(self):
        self.logger.flush()


class NewLineFlushAppendLoggerWrapper(Logger):

    def __init__(self, logger: Logger):
        self.logger = logger

    def write(self, message):
        self.logger.write(message)

    def flush(self):
        self.logger.write("\n")
        self.logger.flush()


def redirect_stdout_and_stderr_to_file(path):
    logger = FlushOnWriteLoggerWrapper(Logger(path))
    sys.stdout = logger
    sys.stderr = NewLineFlushAppendLoggerWrapper(logger)


Point = namedtuple("Point", ["x", "y"])


def is_static(a, b, c, threshold=1e-2):
    """
    Return whether the given 3 points correspond to a static agent
    """
    return np.linalg.norm(a - b) < threshold and np.linalg.norm(c - b) < threshold


def points2angle(a, b, c):
    """
    Return angle formed by 3 points
    """
    ba = a - b
    bc = c - b
    cosine = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1E-20), -1.0, 1.0)
    angle = np.arccos(cosine)
    return angle


def is_smooth(trajectories, thres=np.pi / 10):
    for trajectory in trajectories:
        trajectory = np.array(trajectory)
        for j in range(0, len(trajectory[:, 0]) - 3):
            p1 = np.array([trajectory[j, 0], trajectory[j, 1]])
            p2 = np.array([trajectory[j + 1, 0], trajectory[j + 1, 1]])
            p3 = np.array([trajectory[j + 2, 0], trajectory[j + 2, 1]])
            if not is_static(p1, p2, p3) and points2angle(p1, p2, p3) <= thres:
                return False
    return True


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm
