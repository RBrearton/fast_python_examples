"""Go away pylint"""

import numpy as np

import my_c_library

init_array = np.random.rand(10000000)

# Do something custom to our initial array.
final_array = my_c_library.do_something(init_array)
