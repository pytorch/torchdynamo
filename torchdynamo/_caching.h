#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
#include <vector>

class GuardCheck {
public:
  virtual bool check(PyObject *locals_or_globals) = 0;
  GuardCheck() = default;
  virtual ~GuardCheck() = default;
};

typedef std::vector<GuardCheck *> GuardsVector;

class CacheEntry {
public:
  CacheEntry(CacheEntry *next, PyObject *f_locals, PyObject *f_globals,
             PyObject *guarded_code);

  PyCodeObject *lookup(PyObject *f_locals, PyObject *f_globals);

  inline ~CacheEntry() {
    for (auto i : locals_checks_) {
      delete i;
    }
    for (auto i : globals_checks_) {
      delete i;
    }
    Py_XDECREF(code_);
    delete next_;
  }

private:
  GuardsVector locals_checks_;
  GuardsVector globals_checks_;
  PyCodeObject *code_;
  CacheEntry *next_;
};
#else
typedef void CacheEntry;
#endif

#ifdef __cplusplus
extern "C" {
#endif

PyCodeObject *cached_code_lookup(CacheEntry *cache, PyObject *f_locals,
                                 PyObject *f_globals);

CacheEntry *new_cached_code(CacheEntry *cache, PyObject *f_locals,
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