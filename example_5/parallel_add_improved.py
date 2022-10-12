"""
A simple demonstration of how to write a parallel array add with pure python.

This is slightly improved compared to the previous example, but still not
particularly advanced. It is my hope that this is sufficient for you to discover
the rest by youselves!
"""


import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np


def _on_exit(shared_memory: SharedMemory) -> None:
    """
    Can be used with the atexit module. Makes sure that the shared memory is
    cleaned when called.
    """
    try:
        shared_memory.close()
        shared_memory.unlink()
    except FileNotFoundError:
        # The file has already been unlinked; do nothing.
        pass


def array_add(array_1: np.ndarray, array_2: np.ndarray, index: int,
              shared_mem_name: str, array_shape: Tuple[int], num_threads: int):
    """
    Adds part of two arrays together. Specifically, the elements between:
        int(array_1.size/NUM_THREADS) * index
    and
        int(array_1.size/NUM_THREADS) * (index+1)
    are added together.
    """
    start = int(array_1.size/num_threads) * index
    stop = int(array_1.size/num_threads) * (index+1)

    # Get the shared array.
    shared_memory = SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(
        array_shape, dtype=np.float64, buffer=shared_memory.buf)
    shared_array[start:stop] = array_1[start:stop] + array_2[start:stop]
    print("Part of the addition has been completed!")

    # Do your chores. Note that this is good practice, but not essential.
    shared_memory.close()


if __name__ == '__main__':
    # We can use python to work out how many threads we have access to.
    NUM_THREADS = multiprocessing.cpu_count()
    # The name we're going to give our shared memory block.
    SHARED_MEM_NAME = "output"
    # The number of elements in our shared memory array.
    ARRAY_SIZE = 100000
    # The shape of our shared memory array.
    ARRAY_SHAPE = (ARRAY_SIZE,)

    # Prepare the arrays that we'd like to add.
    array_a = np.random.rand(ARRAY_SIZE)
    array_b = np.random.rand(ARRAY_SIZE) + 1

    # Allocate a shared memory block in which we'll calculate the sum of the
    # arrays.
    shared_mem = SharedMemory(
        SHARED_MEM_NAME, create=True, size=array_a.nbytes)

    # Initialize the array to be full of zeros.
    sum_array = np.ndarray(ARRAY_SHAPE, dtype=array_a.dtype,
                           buffer=shared_mem.buf)
    # Set the array to be full of zeros.
    sum_array.fill(0)

    # Make our processing pool. NUM_THREADS sets the size of the pool. We use
    # python's context semantics to make sure that we don't leak processes.
    with Pool(processes=NUM_THREADS) as pool:
        # Each apply_async call returns an async_result, which we can use to
        # work out when all work we've assigned to the pool has completed.
        async_results = []
        for i in range(NUM_THREADS):
            # Tell our processing pool to do the work!
            async_results.append(
                pool.apply_async(
                    array_add,  # The function to call.
                    # The arguments to the function.
                    (array_a, array_b, i,
                     SHARED_MEM_NAME, ARRAY_SHAPE, NUM_THREADS)
                )
            )

        # Wait for all the work to complete.
        print("Waiting in main thread for work to complete.")
        for result in async_results:
            result.wait()

    # Wait for all the work to complete.
    for result in async_results:
        result.wait()

        # The following ensures that no uncaught exceptions were raised while
        # acquiring result. Very important!
        if not result.successful():
            raise ValueError(
                "Could not carry out map for an unknown reason. "
                "Probably one of the threads segfaulted, or something.")

    did_it_work = ((array_a+array_b) == sum_array).all()
    print(f"Did it work? {did_it_work}!")

    # Do your chores.
    _on_exit(shared_mem)
