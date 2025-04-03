#include "jolopy.h"

#include "yoloc.h"

/* Implementation of the example function */
int jolopy_add(int a, int b) {
    yoloc_t yoloc;
    return yoloc_init(&yoloc);
}

/* Python wrapper for jolopy_add */
static PyObject* py_jolopy_add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    int result = jolopy_add(a, b);
    return PyLong_FromLong(result);
}

/* Method table */
static PyMethodDef JolopyMethods[] = {
    {"add", py_jolopy_add, METH_VARARGS, "Add two integers."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef jolopy_module = {
    PyModuleDef_HEAD_INIT,
    "jolopy",
    "C extension module for jolopy.",
    -1,
    JolopyMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_jolopy(void) {
    return PyModule_Create(&jolopy_module);
}
