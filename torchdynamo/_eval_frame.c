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

#define unlikely(x) __builtin_expect((x), 0)

#define NULL_CHECK(val)                                                        \
  if (unlikely((val) == NULL)) {                                               \
    printf("NULL ERROR: %s:%d\n", __FILE__, __LINE__);                         \
    PyErr_Print();                                                             \
    abort();                                                                   \
  } else {                                                                     \
  }

#define SKIP_CODE ((void *)0x1)

// TODO(jansel): make this threadlocal to support MT
static PyObject *eval_frame_callback = NULL;

static PyObject *noargs = NULL; /* empty tuple */

size_t extra_index = -1;

static void ignored(void *obj) {}

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate);

#if PY_VERSION_HEX < 0x03090000
static inline PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *arg) {
  PyObject *args = PyTuple_Pack(1, arg);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}
#endif

typedef struct cache_entry {
  // check the guards: lambda: <locals of user function>: bool
  PyObject *check_fn;
  // modified user bytecode (protected by check_fn's guards)
  PyCodeObject *code;
  // on a cache miss, linked list of next thing to try
  struct cache_entry *next;
} CacheEntry;

static CacheEntry *create_cache_entry(CacheEntry *next,
                                      PyObject *guarded_code) {
  CacheEntry *e = (CacheEntry *)malloc(sizeof(CacheEntry));
  NULL_CHECK(e);
  e->check_fn = PyObject_GetAttrString(guarded_code, "check_fn");
  NULL_CHECK(e->check_fn);
  e->code = (PyCodeObject *)PyObject_GetAttrString(guarded_code, "code");
  NULL_CHECK(e->code);
  e->next = next;
  return e;
}

/*
// TODO(jansel): need to clean things up eventually
static void destroy_cache_entry(CacheEntry* e) {
    Py_XDECREF(e->check_fn);
    Py_XDECREF(e->code);
    if (e->next != NULL) {
      destroy_cache_entry(e->next);
    }
    free(e);
}
*/

static PyCodeObject *lookup(CacheEntry *e, PyObject *f_locals) {
  if (e == NULL) {
    return NULL;
  }
  PyObject *valid = PyObject_Call(e->check_fn, noargs, f_locals);
  NULL_CHECK(valid);
  Py_DECREF(valid);
  if (valid == Py_True) {
    return e->code;
  }
  return lookup(e->next, f_locals);
}

inline static CacheEntry *get_extra(PyCodeObject *code) {
  CacheEntry *extra = NULL;
  _PyCode_GetExtra((PyObject *)code, extra_index, (void *)&extra);
  return extra;
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

static PyObject *custom_eval_frame(PyFrameObject *frame, int throw_flag) {
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE) {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }

  PyThreadState *tstate = PyThreadState_GET();
  // don't run custom_eval_frame() for guard function
  PyObject *callback = set_eval_frame(Py_None, tstate);

  PyCodeObject *cached_code = lookup(extra, frame->f_locals);
  if (cached_code != NULL) {
    // used cached version
    set_eval_frame(callback, tstate);
    return swap_code_and_run(frame, cached_code, throw_flag);
  }
  // cache miss

  PyObject *result = PyObject_CallOneArg(callback, (PyObject *)frame);
  NULL_CHECK(result);
  if (result != Py_None) {
    // setup guarded cache
    extra = create_cache_entry(extra, result);
    cached_code = extra->code;
  } else {
    // compile failed, skip this frame next time
    extra = SKIP_CODE;
    cached_code = frame->f_code;
  }
  Py_DECREF(result);

  set_extra(cached_code, SKIP_CODE); // avoid double compile
  set_extra(frame->f_code, extra);
  set_eval_frame(callback, tstate);
  return swap_code_and_run(frame, cached_code, throw_flag);
}

static PyObject *custom_eval_frame_run_only(PyFrameObject *frame,
                                            int throw_flag) {
  // do not dynamically compile anything, just reuse prior compiles
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE || extra == NULL) {
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
  // TODO(jansel): investigate directly using the "fast" representation
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }
  PyCodeObject *cached_code = lookup(extra, frame->f_locals);
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
  Py_INCREF(Py_None);
  eval_frame_callback = Py_None;
  noargs = PyTuple_New(0);
  return PyModule_Create(&_module);
}
