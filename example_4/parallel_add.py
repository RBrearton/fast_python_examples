"""
A simple demonstration of how to write a parallel array add with pure python.

There are lots of bad programming practices in this very simple example! The
next example will contain better code; this is just the easiest way to introduce
the basics.
"""


import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# We can use python to work out how many threads we have access to.
NUM_THREADS = multiprocessing.cpu_count()
# The name we're going to give our shared memory block.
SHARED_MEM_NAME = "output"
# The number of elements in our shared memory array.
ARRAY_SIZE = 100000
# The shape of our shared memory array.
ARRAY_SHAPE = (ARRAY_SIZE,)


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

    # Get the shared array.
    shared_memory = SharedMemory(name=SHARED_MEM_NAME)
    shared_array = np.ndarray(
        ARRAY_SHAPE, dtype=np.float64, buffer=shared_memory.buf)

    shared_array[start:stop] = array_1[start:stop] + array_2[start:stop]
    print("Part of the addition has been completed!")


if __name__ == '__main__':
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
                    (array_a, array_b, i)  # The arguments to the function.
                )
            )

        # Wait for all the work to complete.
        print("Waiting in main thread for work to complete.")
        for result in async_results:
            result.wait()

    did_it_work = ((array_a+array_b) == sum_array).all()
    print(f"Did it work? {did_it_work}!")
