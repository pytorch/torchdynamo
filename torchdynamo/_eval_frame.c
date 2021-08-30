#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pystate.h>

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
//#include "internal/pycore_frame.h"
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
static PyObject *eval_frame_callback =
    NULL; // TODO(jansel): make this threadlocal to support MT
size_t extra_index = -1;
static void ignored(void *obj) {}
static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate);

// PyAPI_FUNC(int) _PyCode_GetExtra(PyObject *code, Py_ssize_t index, void
// **extra); PyAPI_FUNC(int) _PyCode_SetExtra(PyObject *code, Py_ssize_t index,
// void *extra);

inline static PyCodeObject *get_extra(PyCodeObject *code) {
  void *extra = NULL;
  _PyCode_GetExtra((PyObject*)code, extra_index, &extra);
  return (PyCodeObject *)extra;
}

inline static void set_extra(PyCodeObject *code, PyCodeObject *extra) {
  _PyCode_SetExtra((PyObject*)code, extra_index, extra);
}

inline static PyObject *swap_code_and_run(PyFrameObject *frame,
                                          PyCodeObject *code, int throw_flag) {
  if(code != frame->f_code) {
      Py_INCREF(code);
      Py_DECREF(frame->f_code);
  }
  frame->f_code = code;
  return _PyEval_EvalFrameDefault(frame, throw_flag);
}

static PyObject *custom_eval_frame(PyFrameObject *frame, int throw_flag) {
  // if ( //PySet_Contains(skip_files, frame->f_code->co_filename) ||
  //     frame->f_lasti != -1) {
  //   return _PyEval_EvalFrameDefault(frame, throw_flag);
  // }

  PyCodeObject *extra = get_extra(frame->f_code);
  if (extra != NULL) {
    // used cached version
    return swap_code_and_run(frame, extra, throw_flag);
  }

  PyThreadState *tstate = PyThreadState_GET();
  PyObject *callback = set_eval_frame(Py_None, tstate);
  PyObject *result = PyObject_CallOneArg(callback, (PyObject *)frame);
  if (result == NULL || !PyCode_Check(result)) {
    return NULL; // exception
  }
  extra = (PyCodeObject *)result;
  set_extra(extra, extra);  // avoid double compile
  set_extra(frame->f_code, extra);
  set_eval_frame(callback, tstate);
  return swap_code_and_run(frame, extra, throw_flag);
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

static PyObject *get_skip_files(void) { return skip_files; }

static PyObject *py_abort(void) { abort(); }

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"get_skip_files", (PyCFunction)&get_skip_files, METH_NOARGS, NULL},
    {"abort", (PyCFunction)&py_abort, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "_eval_frame",
    "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit__eval_frame(void) {
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);
  Py_XDECREF(skip_files);
  skip_files = PySet_New(NULL);
  Py_XDECREF(eval_frame_callback);
  eval_frame_callback = Py_None;
  Py_XINCREF(eval_frame_callback);
  return PyModule_Create(&_module);
}
