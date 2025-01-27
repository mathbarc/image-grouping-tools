import numpy
import pytest

from image_grouping_tool.clustering import distance_functions


@pytest.mark.parametrize(
    "first,second,expected",
    [
        [
            numpy.array([1.0, 1.0, 1.0]),
            numpy.array([0.0, 0.0, 0.0]),
            numpy.array([1.732]),
        ]
    ],
)
def test_euclidian_distance(
    first: numpy.ndarray, second: numpy.ndarray, expected: numpy.ndarray
):
    result = distance_functions.euclidian_distance(first, second)
    numpy.testing.assert_almost_equal(result, expected, 4)


@pytest.mark.parametrize(
    "first,second,expected",
    [
        [
            numpy.array([1.0, 1.0, 1.0]),
            numpy.array([-1.0, -1.0, -1.0]),
            numpy.array([2.0]),
        ]
    ],
)
def test_cosine_distance(
    first: numpy.ndarray, second: numpy.ndarray, expected: numpy.ndarray
):
    result = distance_functions.cosine_distance(first, second)
    numpy.testing.assert_almost_equal(result, expected, 4)
