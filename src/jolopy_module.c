#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "yoloc.h"

typedef struct {
    PyObject_HEAD
    yoloc_t detector;
} YOLODetector;

static void YOLODetector_dealloc(YOLODetector* self) {
    yoloc_free(&self->detector);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* YOLODetector_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    YOLODetector* self;
    self = (YOLODetector*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->detector.impl = NULL;
    }
    return (PyObject*)self;
}

static int YOLODetector_init(YOLODetector* self, PyObject* args, PyObject* kwds) {
    const char* model_path;
    int width, height, num_classes;
    
    if (!PyArg_ParseTuple(args, "siii", &model_path, &width, &height, &num_classes))
        return -1;
    
    int result = yoloc_init(&self->detector, model_path, width, height, num_classes);
    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize YOLO detector");
        return -1;
    }
    
    return 0;
}

static PyObject* YOLODetector_detect(YOLODetector* self, PyObject* args) {
    PyArrayObject* img_array;
    int max_detections = 100;
    
    if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &img_array, &max_detections))
        return NULL;
    
    if (PyArray_NDIM(img_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "Image must be 3-dimensional (H,W,C)");
        return NULL;
    }
    
    int height = PyArray_DIM(img_array, 0);
    int width = PyArray_DIM(img_array, 1);
    int channels = PyArray_DIM(img_array, 2);
    
    if (channels != 3) {
        PyErr_SetString(PyExc_ValueError, "Image must have 3 channels (RGB)");
        return NULL;
    }
    
    const uint8_t* img_data = (const uint8_t*)PyArray_DATA(img_array);
    
    yoloc_detection_t* detections = (yoloc_detection_t*)malloc(max_detections * sizeof(yoloc_detection_t));
    if (!detections) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for detections");
        return NULL;
    }
    
    int num_detections = yoloc_detect(&self->detector, img_data, width, height, channels, detections, max_detections);
    
    if (num_detections < 0) {
        free(detections);
        PyErr_SetString(PyExc_RuntimeError, "Detection failed");
        return NULL;
    }
    
    PyObject* detection_list = PyList_New(num_detections);
    for (int i = 0; i < num_detections; i++) {
        PyObject* det = Py_BuildValue("{s:i,s:f,s:i,s:i,s:i,s:i}",
                                     "class_id", detections[i].class_id,
                                     "confidence", detections[i].confidence,
                                     "x", detections[i].x,
                                     "y", detections[i].y,
                                     "w", detections[i].w,
                                     "h", detections[i].h);
        PyList_SetItem(detection_list, i, det);
    }
    
    free(detections);
    return detection_list;
}

// Getters and setters
static PyObject* YOLODetector_get_confidence_threshold(YOLODetector* self, PyObject* Py_UNUSED(ignored)) {
    float threshold = yoloc_get_confidence_threshold(&self->detector);
    return PyFloat_FromDouble(threshold);
}

static PyObject* YOLODetector_set_confidence_threshold(YOLODetector* self, PyObject* args) {
    float threshold;
    if (!PyArg_ParseTuple(args, "f", &threshold))
        return NULL;
    
    yoloc_set_confidence_threshold(&self->detector, threshold);
    Py_RETURN_NONE;
}

static PyObject* YOLODetector_get_score_threshold(YOLODetector* self, PyObject* Py_UNUSED(ignored)) {
    float threshold = yoloc_get_score_threshold(&self->detector);
    return PyFloat_FromDouble(threshold);
}

static PyObject* YOLODetector_set_score_threshold(YOLODetector* self, PyObject* args) {
    float threshold;
    if (!PyArg_ParseTuple(args, "f", &threshold))
        return NULL;
    
    yoloc_set_score_threshold(&self->detector, threshold);
    Py_RETURN_NONE;
}

static PyObject* YOLODetector_get_nms_threshold(YOLODetector* self, PyObject* Py_UNUSED(ignored)) {
    float threshold = yoloc_get_nms_threshold(&self->detector);
    return PyFloat_FromDouble(threshold);
}

static PyObject* YOLODetector_set_nms_threshold(YOLODetector* self, PyObject* args) {
    float threshold;
    if (!PyArg_ParseTuple(args, "f", &threshold))
        return NULL;
    
    yoloc_set_nms_threshold(&self->detector, threshold);
    Py_RETURN_NONE;
}

static PyObject* YOLODetector_get_letterbox(YOLODetector* self, PyObject* Py_UNUSED(ignored)) {
    int letterbox = yoloc_get_letterbox(&self->detector);
    return PyBool_FromLong(letterbox);
}

static PyObject* YOLODetector_set_letterbox(YOLODetector* self, PyObject* args) {
    int letterbox;
    if (!PyArg_ParseTuple(args, "p", &letterbox))
        return NULL;
    
    yoloc_set_letterbox(&self->detector, letterbox);
    Py_RETURN_NONE;
}

static PyMethodDef YOLODetector_methods[] = {
    {"detect", (PyCFunction)YOLODetector_detect, METH_VARARGS, "Detect objects in an image"},
    {"get_confidence_threshold", (PyCFunction)YOLODetector_get_confidence_threshold, METH_NOARGS, "Get confidence threshold"},
    {"set_confidence_threshold", (PyCFunction)YOLODetector_set_confidence_threshold, METH_VARARGS, "Set confidence threshold"},
    {"get_score_threshold", (PyCFunction)YOLODetector_get_score_threshold, METH_NOARGS, "Get score threshold"},
    {"set_score_threshold", (PyCFunction)YOLODetector_set_score_threshold, METH_VARARGS, "Set score threshold"},
    {"get_nms_threshold", (PyCFunction)YOLODetector_get_nms_threshold, METH_NOARGS, "Get NMS threshold"},
    {"set_nms_threshold", (PyCFunction)YOLODetector_set_nms_threshold, METH_VARARGS, "Set NMS threshold"},
    {"get_letterbox", (PyCFunction)YOLODetector_get_letterbox, METH_NOARGS, "Get letterbox flag"},
    {"set_letterbox", (PyCFunction)YOLODetector_set_letterbox, METH_VARARGS, "Set letterbox flag"},
    {NULL}  /* Sentinel */
};

static PyTypeObject YOLODetectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "jolopy.YOLODetector",
    .tp_doc = "YOLO object detector",
    .tp_basicsize = sizeof(YOLODetector),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = YOLODetector_new,
    .tp_init = (initproc)YOLODetector_init,
    .tp_dealloc = (destructor)YOLODetector_dealloc,
    .tp_methods = YOLODetector_methods,
};

/* Method table */
static PyMethodDef JolopyMethods[] = {
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
    PyObject* m;
    
    if (PyType_Ready(&YOLODetectorType) < 0)
        return NULL;
    
    m = PyModule_Create(&jolopy_module);
    if (m == NULL)
        return NULL;
    
    import_array(); // Initialize NumPy
    
    Py_INCREF(&YOLODetectorType);
    PyModule_AddObject(m, "YOLODetector", (PyObject*)&YOLODetectorType);
    
    return m;
}
