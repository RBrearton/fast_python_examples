#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

static PyObject *print_arr(PyObject *dummy, PyObject *args)
{
    // Pointers to the arguments we're going to receive (an array).
    PyObject *array_arg = NULL;

    // Parse these arguments.
    if (!PyArg_ParseTuple(args, "O", &array_arg))
        return NULL;

    // Get a pointer to the beginning of the array.
    float *pointer_to_array = PyArray_GETPTR1((PyArrayObject *)array_arg, 0);

    // Work out how many elements we're dealing with here.
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)array_arg);
    int num_elements = shape[0];
    printf("We have %i elements in our array.\n", num_elements);

    // Print each of the elements of the array.
    for (int i = 0; i < num_elements; ++i)
    {
        printf("%f\n", pointer_to_array[i]);
    }

    // Make sure to remember to increment the number of times you refer to None.
    Py_IncRef(Py_None);
    return Py_None;
}

static PyMethodDef simple_print_methods[] = {
    {
        // This is the name your function takes on the python end.
        "print_arr",
        // This is your function on the C end.
        print_arr,
        // Specifies that your function can take a variable number of arguments.
        METH_VARARGS,
        // The __doc__ attribute on the python end.
        "Just a test, really.",
    },
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef simple_print_definition = {
    PyModuleDef_HEAD_INIT,
    // The name of your module on the python end.
    "simple_print",
    // Your module's __doc__
    "A little test NumPy/C module.",
    // Probably important lmao.
    -1,
    // The struct containing all our module's stuff.
    simple_print_methods};

PyMODINIT_FUNC PyInit_simple_print(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&simple_print_definition);
}
