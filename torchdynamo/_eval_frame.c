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

#include "_caching.h"

static PyObject *eval_frame_callback =
    NULL; // TODO(jansel): make this threadlocal to support MT
size_t extra_index = -1;
static void ignored(void *obj) {}
static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate);

inline static CacheEntry *get_extra(PyCodeObject *code) {
  void *extra = NULL;
  _PyCode_GetExtra((PyObject *)code, extra_index, &extra);
  return (PyCodeObject *)extra;
}

inline static void set_extra(PyCodeObject *code, CacheEntry *extra) {
  // TODO(jansel): would it be faster to bypass this?
  _PyCode_SetExtra((PyObject *)code, extra_index, extra);
}

inline static PyObject *swap_code_and_run(PyFrameObject *frame,
                                          PyCodeObject *code, int throw_flag) {
  Py_INCREF(code);
  Py_DECREF(frame->f_code);
  frame->f_code = code;
  return _PyEval_EvalFrameDefault(frame, throw_flag);
}

#define SKIP_CODE ((void *)0x1)

static PyObject *custom_eval_frame(PyFrameObject *frame, int throw_flag) {
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE) {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }
  Py_INCREF(frame->f_locals);

  PyCodeObject *cached_code =
      cached_code_lookup(extra, frame->f_locals, frame->f_globals);
  if (cached_code != NULL) {
    // used cached version
    Py_DECREF(frame->f_locals);
    return swap_code_and_run(frame, cached_code, throw_flag);
  }
  // cache miss

  PyThreadState *tstate = PyThreadState_GET();
  PyObject *callback = set_eval_frame(Py_None, tstate);

  PyObject *result = PyObject_CallOneArg(callback, (PyObject *)frame);
  if (unlikely(result == NULL)) {
    printf("ERROR: Unexpected failure callback hook\n");
    return NULL; // exception
  } else if (result != Py_None) {
    // setup guarded cache
    extra = new_cached_code(extra, frame->f_locals, frame->f_globals, result);
    cached_code = cached_code_lookup(extra, frame->f_locals, frame->f_globals);
    if (unlikely(cached_code == NULL)) {
      printf("ERROR: Unexpected failure in cached_code_lookup\n");
      return NULL;
    }
  } else {
    // compile failed, skip this frame next time
    extra = SKIP_CODE;
    cached_code = frame->f_code;
  }

  Py_DECREF(result);
  set_extra(cached_code, SKIP_CODE); // avoid double compile
  set_extra(frame->f_code, extra);
  Py_DECREF(frame->f_locals);
  set_eval_frame(callback, tstate);
  return swap_code_and_run(frame, cached_code, throw_flag);
}

static PyObject *custom_eval_frame_run_only(PyFrameObject *frame,
                                            int throw_flag) {
  // do not dynamically compile anything, just reuse prior compiles
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == NULL || extra == SKIP_CODE) {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }

  // TODO(jansel): investigate directly using the "fast" representation
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }
  // Py_INCREF(frame->f_locals); // is this needed?
  PyCodeObject *cached_code =
      cached_code_lookup(extra, frame->f_locals, frame->f_globals);
  // Py_DECREF(frame->f_locals); // is this needed?
  if (cached_code != NULL) {
    // used cached version
    return swap_code_and_run(frame, cached_code, throw_flag);
  } else {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
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

static PyObject *set_eval_frame_run_only(PyObject *dummy, PyObject *args) {
  PyThreadState_GET()->interp->eval_frame = &custom_eval_frame_run_only;
  PyObject *old_callback = eval_frame_callback;
  Py_INCREF(Py_None);
  eval_frame_callback = Py_None;
  return old_callback;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"set_eval_frame_run_only", set_eval_frame_run_only, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "_eval_frame",
    "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit__eval_frame(void) {
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);
  Py_XDECREF(eval_frame_callback);
  eval_frame_callback = Py_None;
  Py_XINCREF(eval_frame_callback);
  return PyModule_Create(&_module);
}
