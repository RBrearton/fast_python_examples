#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

static PyObject *print_arr_broken(PyObject *dummy, PyObject *args)
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

    // I didn't incref Py_None!! Hilarity will ensue (or frustration, depending
    // on your point of view).
    return Py_None;
}

/*
Note that in print_arr_broken, we're explicitly casting our (PyObject *) to a
(PyArrayObject *) a bunch of times. Perhaps a neater way of doing this is to
make a new PyArrayObject on the C end. Well... maybe it isn't neater, but at
least it demonstrates something important: how to instantiate a new PyObject
from within C, and how to not leak memory when doing so!
*/
static PyObject *print_arr_working(PyObject *dummy, PyObject *args)
{
    // Pointers to the arguments we're going to receive (an array).
    PyObject *array_arg = NULL;

    // Parse these arguments.
    if (!PyArg_ParseTuple(args, "O", &array_arg))
        return NULL;

    // Make a new PyArrayObject: this is equivalent to doing:
    //      np.array(array_arg)
    // on the python end.
    PyArrayObject *array = (PyArrayObject *)
        PyArray_FROM_OTF(array_arg, NPY_FLOAT32, NPY_IN_ARRAY);

    // Now we can write code that looks like the code in print_arr_broken, but
    // without all the explicit (PyArrayObject *) casts.
    // NOTE: we could delete all the casts in print_arr_broken and our code
    // would still work, but we'd get a load of warnings at compilation time.
    // Please, listen to your compiler! If your code is so crappy that you
    // get warnings, improve it!! Explicit casts are _good_!
    float *pointer_to_array = PyArray_GETPTR1(array, 0);

    // Work out how many elements we're dealing with here.
    npy_intp *shape = PyArray_SHAPE(array);
    int num_elements = shape[0];
    printf("We have %i elements in our array.\n", num_elements);

    // Print each of the elements of the array.
    for (int i = 0; i < num_elements; ++i)
    {
        printf("%f\n", pointer_to_array[i]);
    }

    // This time we don't just need to IncRef Py_None. We also need to DecRef
    // our PyArrayObject that we instantiated in this function. Note that the
    // Py_DecRef function expects a pointer to a PyObject, so we cast our
    // PyArrayObject pointer back to a PyObject pointer.
    Py_DecRef((PyObject *)array);
    Py_IncRef(Py_None);
    return Py_None;
}

static PyMethodDef more_examples_methods[] = {
    {
        // This is the name your function takes on the python end.
        "print_arr_broken",
        // This is your function on the C end.
        print_arr_broken,
        // Specifies that your function can take a variable number of arguments.
        METH_VARARGS,
        // The __doc__ attribute on the python end.
        "This will break the next time you use None.",
    },
    {
        // This is the name your function takes on the python end.
        "print_arr_working",
        // This is your function on the C end.
        print_arr_working,
        // Specifies that your function can take a variable number of arguments.
        METH_VARARGS,
        // The __doc__ attribute on the python end.
        "This one ought to work!",
    },
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef more_examples_definition = {
    PyModuleDef_HEAD_INIT,
    // The name of your module on the python end.
    "more_examples",
    // Your module's __doc__
    "A little test NumPy/C module.",
    // Probably important lmao.
    -1,
    // The struct containing all our module's stuff.
    more_examples_methods};

PyMODINIT_FUNC PyInit_more_examples(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&more_examples_definition);
}
