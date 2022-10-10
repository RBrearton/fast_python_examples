"""
The purpose of this example is to demonstrate the difference between scalar
operations on an array, and equivalent array operations on an array. This is
likely due to pointer arithmetic overhead, but I do not know that as a fact.
"""

from timeit import timeit

import numpy as np


def normal_subtract(big: np.ndarray, small: np.ndarray):
    """
    Those familiar with numpy will know that we can simply subtract the small
    array from the big array, because their 'fast' dimension matches. The
    result will look like this:
    interesting_array = [
      [big[0, 0] - small[0], big[0, 1] - small[1], big[0, 2] - small[2]]
      [big[1, 0] - small[0], big[1, 1] - small[1], big[1, 2] - small[2]]
      ...
      ...
      [big[N, 0] - small[0], big[N, 1] - small[1], big[N, 2] - small[2]]
    ]
    """
    big -= small
    return big


def fast_subtract(big: np.ndarray, small: np.ndarray):
    """
    This can also be done in a seemingly silly/childish way: explicitly.
    However, subtraction component by component reduces to subtraction by a
    scalar, which numpy is capable of doing pretty much optimally.
    """
    big[:, 0] -= small[0]
    big[:, 1] -= small[1]
    big[:, 2] -= small[2]
    return big


if __name__ == "__main__":
    # In scientific computing, one often has to deal with very large collections
    # of 3-vectors. At Diamond, for example, scattering Q-vectors are collected
    # in enormous quantities.

    # The Q vector is defined as the difference between the outgoing wavevector
    # of light and the incident wavevector. Very often, the incident wavevector
    # is a constant, while the outgoing wavevector varies for each pixel on a
    # detector. So, to calculate an array of Q vectors, we need to subtract the
    # same vector from a very large collection of vectors! Let's explore this:
    big_array = np.random.rand(100000000, 3)
    small_array = np.random.rand(3)

    # Now measure the execution time of the abovedefined functions.
    execution_time_1 = timeit(
        lambda: normal_subtract(big_array, small_array), number=1)
    execution_time_2 = timeit(
        lambda: fast_subtract(big_array, small_array), number=1)
    print(f"Natural code took {execution_time_1}s to execute.")
    print(f"Ugly code took {execution_time_2}s to execute.")
    print(f"Ugly method was {execution_time_1/execution_time_2}x faster.")

    # The conclusion? Well, one is clearly more natural than the other, but
    # there is a decent performance difference between the two (around a factor
    # of 4 on my machine). If this isn't bottlenecking: just write natural code
    # as usual. If this routine is bottlenecking though, wrapping a little loop
    # in a function, as I've done here, is a relatively tidy option.
