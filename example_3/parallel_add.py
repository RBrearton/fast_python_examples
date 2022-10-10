"""
A simple demonstration of how to write a parallel array add with pure python.
"""

import numpy as np

import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory


# We can use python to work out how many threads we have access to.
NUM_THREADS = multiprocessing.cpu_count()


def array_add(array_1: np.ndarray, array_2: np.ndarray, index: int):
    """
    Adds part of two arrays together. Specifically, the elements between:
        int(array_1.size/NUM_THREADS) * index
    and
        int(array_1.size/NUM_THREADS) * (index+1)
    are added together.
    """
    start = int(array_1.size/NUM_THREADS) * index
    stop = int(array_1.size/NUM_THREADS) * (index+1)
    array_1[start:stop] += array_2[start:stop]


if __name__ == '__main__':
    array_1 = np.random.rand(1000000)
    array_2 = np.random.rand(1000000)

    with Pool(processes=NUM_THREADS,  # The size of our pool.
              ) as pool:
        async_results = []
        for i in range(NUM_THREADS):
            async_results.append(
                pool.apply_async(
                    array_add,  # The function to call.
                    (array_1, array_2, i)  # The arguments to the function.
                )
            )

        # Wait for all the work to complete.
        for result in async_results:
            result.wait()
            if not result.successful():
                raise ValueError(
                    "Could not carry out map for an unknown reason. "
                    "Probably one of the threads segfaulted, or something.")
