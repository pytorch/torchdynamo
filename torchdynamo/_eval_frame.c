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

//#define TORCHDYNAMO_DEBUG
#define bool char
#define false 0
#define true 1
#define unlikely(x) __builtin_expect((x), 0)

#define NULL_CHECK(val)                                                        \
  if (unlikely((val) == NULL)) {                                               \
    fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__);                \
    PyErr_Print();                                                             \
    abort();                                                                   \
  } else {                                                                     \
  }

#define CHECK(cond)                                                            \
  if (unlikely(!(cond))) {                                                     \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__);        \
    abort();                                                                   \
  } else {                                                                     \
  }

#ifdef TORCHDYNAMO_DEBUG

#define DEBUG_CHECK(cond) CHECK(cond)
#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)                                                  \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
#define DEBUG_TRACE0(msg)                                                      \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)

#else

#define DEBUG_CHECK(cond)
#define DEBUG_NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)
#define DEBUG_TRACE0(msg)

#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void *)0x1)

// TODO(jansel): make this threadlocal to support MT
static PyObject *eval_frame_callback = NULL;

static PyObject *noargs = NULL; /* cached empty tuple */

size_t extra_index = -1;

static void ignored(void *obj) {}

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate);

static inline PyObject *call_callback(PyObject *callable, PyObject *frame,
                                      long cache_len) {
  PyObject *args = Py_BuildValue("(Ol)", frame, cache_len);
  NULL_CHECK(args);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

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
  DEBUG_NULL_CHECK(e);
  e->check_fn = PyObject_GetAttrString(guarded_code, "check_fn");
  NULL_CHECK(e->check_fn);
  e->code = (PyCodeObject *)PyObject_GetAttrString(guarded_code, "code");
  NULL_CHECK(e->code);
  e->next = next;
  return e;
}

static void destroy_cache_entry(CacheEntry *e) {
  if (e == NULL || e == SKIP_CODE) {
    return;
  }
  Py_XDECREF(e->check_fn);
  Py_XDECREF(e->code);
  destroy_cache_entry(e->next);
  free(e);
}

inline static const char *name(PyFrameObject *frame) {
  DEBUG_CHECK(PyUnicode_Check(frame->f_code->co_name));
  return PyUnicode_AsUTF8(frame->f_code->co_name);
}

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

static long cache_size(CacheEntry *e) {
  if (e == NULL) {
    return 0;
  }
  return 1 + cache_size(e->next);
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

/*
inline static PyObject *eval_custom_code_inplace(PyFrameObject *frame,
                                                 PyCodeObject *code,
                                                 int throw_flag) {
  PyCodeObject *prior_code = frame->f_code;
  frame->f_code = code;
  PyObject *result = _PyEval_EvalFrameDefault(frame, throw_flag);
  frame->f_code = prior_code;
  return result;
} */

inline static PyObject *eval_custom_code_shadow_frame(PyThreadState *tstate,
                                                      PyFrameObject *frame,
                                                      PyCodeObject *code,
                                                      int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals = code->co_nlocals;

  if ((code->co_flags & CO_NOFREE) == 0) {
    ncells = PyTuple_GET_SIZE(code->co_cellvars);
    nfrees = PyTuple_GET_SIZE(code->co_freevars);
  }

  DEBUG_NULL_CHECK(tstate);
  DEBUG_NULL_CHECK(frame);
  DEBUG_NULL_CHECK(code);
  DEBUG_CHECK(ncells == PyTuple_GET_SIZE(frame->f_code->co_cellvars));
  DEBUG_CHECK(nfrees == PyTuple_GET_SIZE(frame->f_code->co_freevars));
  DEBUG_CHECK(nlocals == frame->f_code->co_nlocals);

  PyFrameObject *shadow =
      _PyFrame_New_NoTrack(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  PyObject **fastlocals_old = frame->f_localsplus;
  PyObject **fastlocals_new = shadow->f_localsplus;

  for (Py_ssize_t i = 0; i < ncells + nfrees + nlocals; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  PyObject *result = _PyEval_EvalFrameDefault(shadow, throw_flag);

  // cleanup code copied from cpython/.../call.c
  if (Py_REFCNT(shadow) > 1) {
    Py_DECREF(shadow);
    PyObject_GC_Track(shadow);
  } else {
    ++tstate->recursion_depth;
    Py_DECREF(shadow);
    --tstate->recursion_depth;
  }
  return result;
}

inline static PyObject *eval_custom_code(PyThreadState *tstate,
                                         PyFrameObject *frame,
                                         PyCodeObject *code, int throw_flag) {
  // if (code->co_stacksize <= frame->f_code->co_stacksize) {
  //   return eval_custom_code_inplace(frame, code, throw_flag);
  // } else {
  // need to create a new frame object in order to have a bigger stack
  return eval_custom_code_shadow_frame(tstate, frame, code, throw_flag);
  // }
}

static PyObject *custom_eval_frame(PyFrameObject *frame, int throw_flag) {
  DEBUG_TRACE("begin %s %s %i %i %i %i", name(frame),
              PyUnicode_AsUTF8(frame->f_code->co_filename), frame->f_lineno,
              frame->f_lasti, frame->f_iblock, frame->f_executing);
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE) {
    DEBUG_TRACE("skip %s", name(frame));
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    DEBUG_TRACE("error %s", name(frame));
    return NULL;
  }
  DEBUG_CHECK(PyDict_CheckExact(frame->f_locals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_globals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_builtins));

  PyThreadState *tstate = PyThreadState_GET();
  // don't run custom_eval_frame() for guard function
  PyObject *callback = set_eval_frame(Py_None, tstate);

  PyCodeObject *cached_code = lookup(extra, frame->f_locals);
  if (cached_code != NULL) {
    // used cached version
    DEBUG_TRACE("cache hit %s", name(frame));
    set_eval_frame(callback, tstate);
    return eval_custom_code(tstate, frame, cached_code, throw_flag);
  }
  // cache miss

  PyObject *result =
      call_callback(callback, (PyObject *)frame, cache_size(extra));
  if (result == NULL) {
    // internal exception, returning here will leak the exception into user code
    // this is useful for debugging -- but we dont want it to happen outside of
    // testing
    return NULL;
  } else if (result != Py_None) {
    DEBUG_TRACE("create cache %s", name(frame));
    extra = create_cache_entry(extra, result);
    Py_DECREF(result);
    set_extra(frame->f_code, extra);
    set_eval_frame(callback, tstate);
    return eval_custom_code(tstate, frame, extra->code, throw_flag);
  } else {
    DEBUG_TRACE("create skip %s", name(frame));
    Py_DECREF(result);
    set_extra(frame->f_code, SKIP_CODE);
    set_eval_frame(callback, tstate);
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
}

static PyObject *custom_eval_frame_run_only(PyFrameObject *frame,
                                            int throw_flag) {
  // do not dynamically compile anything, just reuse prior compiles
  DEBUG_TRACE("begin %s", name(frame));
  CacheEntry *extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE || extra == NULL) {
    DEBUG_TRACE("skip %s", name(frame));
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
  // TODO(jansel): investigate directly using the "fast" representation
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    DEBUG_TRACE("error %s", name(frame));
    return NULL;
  }
  PyCodeObject *cached_code = lookup(extra, frame->f_locals);
  if (cached_code != NULL) {
    // used cached version
    DEBUG_TRACE("cache hit %s", name(frame));
    return eval_custom_code(PyThreadState_GET(), frame, cached_code,
                            throw_flag);
  } else {
    DEBUG_TRACE("cache miss %s", name(frame));
    return _PyEval_EvalFrameDefault(frame, throw_flag);
  }
}

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate) {
  PyObject *old_callback = eval_frame_callback;
  eval_frame_callback = new_callback;
  if (new_callback == Py_None) {
    // disable eval frame hook
    DEBUG_TRACE0("disable");
    tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
  } else if (old_callback == Py_None) {
    // enable eval frame hook
    DEBUG_TRACE0("enable");
    tstate->interp->eval_frame = &custom_eval_frame;
  }
  return old_callback;
}

static PyObject *set_eval_frame_py(PyObject *dummy, PyObject *args) {
  PyObject *callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (callback != Py_None && !PyCallable_Check(callback)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  Py_INCREF(callback);
  DEBUG_TRACE("python enabled=%d", callback != Py_None);
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyObject *set_eval_frame_run_only(PyObject *dummy, PyObject *args) {
  DEBUG_TRACE0("enable: run_only");
  PyThreadState_GET()->interp->eval_frame = &custom_eval_frame_run_only;
  PyObject *old_callback = eval_frame_callback;
  Py_INCREF(Py_None);
  eval_frame_callback = Py_None;
  return old_callback;
}

static PyObject *reset_code(PyObject *dummy, PyObject *args) {
  PyObject *code = NULL;
  if (!PyArg_ParseTuple(args, "O:code", &code)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (!PyCode_Check(code)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }

  destroy_cache_entry(get_extra((PyCodeObject *)code));
  set_extra((PyCodeObject *)code, NULL);
  Py_RETURN_NONE;
}

static PyObject *unsupported(PyObject *dummy, PyObject *args) {
  // a dummy C function used in testing
  PyObject *obj1 = NULL;
  PyObject *obj2 = NULL;
  if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2)) {
    return NULL;
  }
  Py_INCREF(obj2);
  return obj2;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"set_eval_frame_run_only", set_eval_frame_run_only, METH_NOARGS, NULL},
    {"reset_code", reset_code, METH_VARARGS, NULL},
    {"unsupported", unsupported, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "_eval_frame",
    "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit__eval_frame(void) {
  CHECK(sizeof(unsigned long) == sizeof(void *));
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);
  Py_INCREF(Py_None);
  eval_frame_callback = Py_None;
  noargs = PyTuple_New(0);
  return PyModule_Create(&_module);
}
