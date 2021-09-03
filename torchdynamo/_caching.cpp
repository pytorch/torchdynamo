#include "_caching.h"
#include <iostream>

#define unlikely(x) __builtin_expect((x), 0)
#define NULL_CHECK(val)                                                        \
  if (unlikely((val) == NULL)) {                                               \
    std::cerr << "NULL ERROR: " << __FILE__ << ":" << __LINE__ << std::endl;   \
    abort();                                                                   \
  } else {                                                                     \
  }

namespace {

void abort_error(const char *msg) {
  std::cerr << "ERROR: " << msg << std::endl;
  abort();
}

PyObject *_get_item_type(PyObject *scope, PyObject *key) {
  PyObject *value = PyObject_GetItem(scope, key); // new ref
  if (value == NULL) {
    return NULL;
  }
  PyObject *objtype = (PyObject *)Py_TYPE(value); // borrows
  Py_INCREF(objtype);
  Py_DECREF(value);
  return objtype;
}

void init_value_guards(GuardsVector &out, PyObject *in, PyObject *scope) {
  size_t len = PyObject_Length(in);
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    PyObject *key = PyList_GetItem(in, i); // borrows
    NULL_CHECK(key);
    Py_INCREF(key);
    PyObject *value = PyObject_GetItem(scope, key); // new ref
    NULL_CHECK(value);
    PyObject *valref = PyWeakref_NewRef(value, NULL);
    NULL_CHECK(valref);
    Py_DECREF(value);
    out.emplace_back(std::make_pair(key, valref));
  }
}

bool check_value_guards(const GuardsVector &guards, PyObject *scope) {
  for (auto item : guards) {
    PyObject *key = item.first;
    PyObject *value1 = PyWeakref_GET_OBJECT(item.second);
    PyObject *value2 = PyObject_GetItem(scope, key); // new ref
    Py_XDECREF(value2);
    if (value1 == NULL || value1 != value2) {
      return false;
    }
  }
  return true;
}

void init_type_guards(GuardsVector &out, PyObject *in, PyObject *scope) {
  size_t len = PyObject_Length(in);
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    PyObject *key = PyList_GetItem(in, i); // borrows
    NULL_CHECK(key);
    Py_INCREF(key);
    PyObject *objtype = _get_item_type(scope, key);
    NULL_CHECK(objtype);
    out.emplace_back(std::make_pair(key, objtype));
  }
}

bool check_type_guards(const GuardsVector &guards, PyObject *scope) {
  if (scope == NULL) {
    abort_error("scope is null");
  }
  for (auto item : guards) {
    PyObject *key = item.first;
    PyObject *objtype1 = item.second;
    PyObject *objtype2 = _get_item_type(scope, key);
    if (objtype1 != objtype2) {
      return false;
    }
  }
  return true;
}

} // namespace

extern "C" PyCodeObject *
cached_code_lookup(CacheEntry *cache, PyObject *f_locals, PyObject *f_globals) {
  if (cache == NULL) {
    return NULL;
  }
  return cache->lookup(f_locals, f_globals);
}

extern "C" CacheEntry *new_cached_code(CacheEntry *cache, PyObject *f_locals,
                                       PyObject *f_globals, PyObject *result) {
  return new CacheEntry(cache, f_locals, f_globals, result);
}

PyCodeObject *CacheEntry::lookup(PyObject *f_locals, PyObject *f_globals) {
  if (check_value_guards(value_locals_, f_locals) &&
      check_value_guards(value_globals_, f_globals) &&
      check_type_guards(type_locals_, f_locals) &&
      check_type_guards(type_globals_, f_globals)) {
    return code_;
  }
  if (next_ != NULL) {
    return next_->lookup(f_locals, f_globals);
  }
  return NULL;
}

CacheEntry::CacheEntry(CacheEntry *next, PyObject *f_locals,
                       PyObject *f_globals, PyObject *guarded_code)
    : next_(next) {
  PyObject *code = PyObject_GetAttrString(guarded_code, "code");
  PyObject *value_locals = PyObject_GetAttrString(guarded_code, "value_locals");
  PyObject *value_globals =
      PyObject_GetAttrString(guarded_code, "value_globals");
  PyObject *type_locals = PyObject_GetAttrString(guarded_code, "type_locals");
  PyObject *type_globals = PyObject_GetAttrString(guarded_code, "type_globals");
  if (code == NULL || value_locals == NULL || value_globals == NULL ||
      type_locals == NULL || type_globals == NULL) {
    abort_error("callback must return a GuardedCode()");
  }

  init_value_guards(value_locals_, value_locals, f_locals);
  init_value_guards(value_globals_, value_globals, f_globals);
  init_type_guards(type_locals_, type_locals, f_locals);
  init_type_guards(type_globals_, type_globals, f_globals);

  Py_DECREF(value_locals);
  Py_DECREF(value_globals);
  Py_DECREF(type_locals);
  Py_DECREF(type_globals);
  code_ = (PyCodeObject *)code;
}
