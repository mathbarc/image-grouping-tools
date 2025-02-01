import numpy


def euclidian_distance(first: numpy.ndarray, second: numpy.ndarray) -> float:
    """Computes euclidian distance between two arrays. This function applies the following equation:

    .. math::
        \\sqrt{(x_0-y_0)^2 + ... + (x_n-y_n)^2}

    :param first: First array
    :param second: Second array
    :returns: Euclidian distance between arrays
    """
    dist = first - second
    dist = numpy.atleast_2d(dist)
    dist = numpy.sqrt(numpy.power(dist, 2).sum(axis=1))
    return dist


def cosine_distance(first: numpy.ndarray, second: numpy.ndarray) -> float:
    """Computes cosine distance between two arrays.

    .. math::
        \\frac{X \\times Y}{||X|| ||Y||}

    :param first: First array
    :param second: Second array
    :returns: Cosine distance between arrays
    """
    first_vector = numpy.atleast_2d(first)
    second_vector = numpy.atleast_2d(second)

    internal_product = numpy.dot(first_vector, second_vector.T).item()
    norm_first = numpy.sqrt(numpy.dot(first_vector, first_vector.T)).item()
    norm_second = numpy.sqrt(numpy.dot(second_vector, second_vector.T)).item()
    return 1.0 - (internal_product / (norm_first * norm_second))


class minkowski_distance:
    """Computes Minkowski distance between two arrays.

    .. math::
        \\sqrt[m]{(x_0-y_0)^m + ... + (x_n-y_n)^m}

    Constructor Arguments:

        power (float): Exponent used to compute Minkowki distance

    Callable Arguments:

    :param first: First array
    :param second: Second array
    :returns: Minkowski distance between arrays
    """

    def __init__(self, power):
        self.power = power
        self.ratio = 1.0 / power

    def __call__(self, first: numpy.ndarray, second: numpy.ndarray) -> float:
        """Computes Minkowski distance between two arrays.

        :param d1: First array
        :param d2: Second array
        :returns: Minkowski distance between arrays
        """
        dist_vector = numpy.abs(first - second)
        dist_vector = numpy.atleast_2d(dist_vector)
        return pow(pow(dist_vector, self.power).sum(axis=1), self.ratio)
