from typing import Callable
import numpy

from . import distance_functions


def db_scan(
    data: numpy.ndarray,
    max_neighborhood_dist: float = 10.0,
    min_point_count_for_core: int = 4,
    distance_function: Callable = distance_functions.euclidian_distance,
):
    pass
