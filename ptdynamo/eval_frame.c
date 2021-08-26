#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pystate.h>

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
#undef Py_BUILD_CORE
#endif

#if PY_VERSION_HEX < 0x03090000
PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *arg) {
  PyObject *args = PyTuple_Pack(1, arg);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}
#endif

static PyObject *skip_files = NULL;
static PyObject *execute_sentinel = NULL;
static PyObject *eval_frame_callback = NULL;
size_t extra_index = -1;
static void ignored(void *obj) {}
static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate);

// PyAPI_FUNC(int) _PyCode_GetExtra(PyObject *code, Py_ssize_t index, void
// **extra); PyAPI_FUNC(int) _PyCode_SetExtra(PyObject *code, Py_ssize_t index,
// void *extra);

static PyObject *custom_eval_frame(PyFrameObject *frame, int throw_flag) {
  if(PySet_Contains(skip_files, frame->f_code->co_filename)) {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }

  PyThreadState *tstate = PyThreadState_GET();
  PyObject *callback = set_eval_frame(Py_None, tstate);
  PyObject *result = PyObject_CallOneArg(callback, (PyObject *)frame);
  set_eval_frame(callback, tstate);

  if (result == NULL) {
    return NULL; // exception
  } else if (result == execute_sentinel) {
    Py_DECREF(result);
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  } else {
    return result;
  }
}

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate) {
  PyObject *old_callback = eval_frame_callback;
  eval_frame_callback = new_callback;
  if (new_callback == Py_None) {
    // disable eval frame hook
    tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
  } else if (old_callback == Py_None) {
    // enable eval frame hook
    tstate->interp->eval_frame = &custom_eval_frame;
  }
  return old_callback;
}

static PyObject *set_eval_frame_py(PyObject *dummy, PyObject *args) {
  PyObject *callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    return NULL;
  }
  if (callback != Py_None && !PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  Py_INCREF(callback);
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyObject *get_execute_sentinel() { return execute_sentinel; }

static PyObject* get_skip_files() { return skip_files; }

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"get_execute_sentinel", get_execute_sentinel, METH_NOARGS, NULL},
    {"get_skip_files", get_skip_files, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "ptdynamo.eval_frame",
    "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit_eval_frame(void) {
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);
  Py_XDECREF(execute_sentinel);
  execute_sentinel = PyTuple_New(0);
  Py_XDECREF(skip_files);
  skip_files = PySet_New(NULL);
  Py_XDECREF(eval_frame_callback);
  eval_frame_callback = Py_None;
  Py_XINCREF(eval_frame_callback);
  return PyModule_Create(&_module);
}
