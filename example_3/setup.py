"""
Simple setup.py file for building a C/numpy extension library.
"""

import sysconfig

from numpy.distutils.core import setup as numpy_setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.misc_util import get_info

EXTRA_COMPILE_ARGS = sysconfig.get_config_var('CFLAGS').split()
EXTRA_COMPILE_ARGS += ["-std=c99"]


def configuration(parent_package='', top_path=None):
    """
    Defines a simple configuration for a numpy C extension.
    """
    module_name = "more_examples"
    path_to_c_module = "more_examples.c"

    # Necessary for the half-float d-type.
    extra_info = get_info('npymath')

    config = Configuration('',
                           parent_package,
                           top_path)
    config.add_extension(module_name,
                         [path_to_c_module],
                         extra_info=extra_info,
                         language='c99',
                         extra_compile_args=EXTRA_COMPILE_ARGS)

    return config


if __name__ == "__main__":
    numpy_setup(configuration=configuration)
