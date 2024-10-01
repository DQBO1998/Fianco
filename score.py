from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast
from fianco import number, Mat, YX, is_capt, win, vdir, Engine
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
from numpy import inf

import numpy as np
import numba as nb # type: ignore


def global_scr(wrt: int, end: Mat, brd: Mat, prm: NDArray[np.float32]) -> float:
    good = np.sum(brd[wrt] * prm)
    bad = np.sum(brd[1 - wrt] * prm)
    return good - bad


def scr_fn(wrt, )