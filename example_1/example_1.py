"""
The purpose of this example is to demonstrate the difference between scalar
operations on an array, and equivalent array operations on an array. This is
likely due to pointer arithmetic overhead, but I do not know that as a fact.
"""

import time

import numpy as np

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

    # Those familiar with numpy will know that we can simply subtract the small
    # array from the big array, because their 'fast' dimension matches. The
    # result will look like this:
    # interesting_array = [
    #   [big[0, 0] - small[0], big[0, 1] - small[1], big[0, 2] - small[2]]
    #   [big[1, 0] - small[0], big[1, 1] - small[1], big[1, 2] - small[2]]
    #   ...
    #   ...
    #   [big[N, 0] - small[0], big[N, 1] - small[1], big[N, 2] - small[2]]
    # ]
    # where, in this case, N = 100000000
    interesting_array = big_array - small_array
