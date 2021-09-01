#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
struct CacheEntry {
  PyObject *guards;
  PyCodeObject *code;
  CacheEntry *next;
};
#else
typedef void CacheEntry;
#endif

#ifdef __cplusplus
extern "C" {
#endif

PyCodeObject *get_cached_code(CacheEntry *cache, PyObject *f_locals,
                              PyObject *f_globals);

CacheEntry *set_cached_code(CacheEntry *cache, PyObject *f_locals,
                            PyObject *f_globals, PyObject *result);

#ifdef __cplusplus
}
#endif

#if PY_VERSION_HEX < 0x03090000
static inline PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *arg) {
  PyObject *args = PyTuple_Pack(1, arg);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}
#endif


static inline PyObject *PyObject_CallTwoArg(PyObject *callable, PyObject *arg1, PyObject *arg2) {
  PyObject *args = PyTuple_Pack(2, arg1, arg2);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}
