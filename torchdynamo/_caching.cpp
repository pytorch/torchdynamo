#include "_caching.h"
#include <iostream>

#define NULL_CHECK(val)                                                        \
  if (unlikely((val) == NULL)) {                                               \
    std::cerr << "NULL ERROR: " << __FILE__ << ":" << __LINE__ << std::endl;   \
    PyErr_Print();                                                             \
    abort();                                                                   \
  } else {                                                                     \
  }

namespace {

void abort_error(const char *msg) {
  std::cerr << "ERROR: " << msg << std::endl;
  abort();
}

bool check_guards(const GuardsVector &guards, PyObject *scope) {
  if (scope == NULL) {
    abort_error("scope is null");
  }
  for (auto item : guards) {
    if (!item->check(scope)) {
      return false;
    }
  }
  return true;
}

PyObject *_get_item_type(PyObject *scope, PyObject *key) {
  PyObject *value = PyObject_GetItem(scope, key); // new ref
  if (value == NULL) {
    return NULL;
  }
  PyObject *objtype = (PyObject *)Py_TYPE(value); // borrows
  Py_DECREF(value);
  return objtype;
}

class TypeCheck : public GuardCheck {
  PyObject *key_;
  PyObject *objtype_;

public:
  TypeCheck(PyObject *key, PyObject *scope) {
    key_ = key;
    Py_INCREF(key_);

    objtype_ = _get_item_type(scope, key);
    NULL_CHECK(objtype_);
    Py_INCREF(objtype_);
  }
  ~TypeCheck() {
    Py_DECREF(key_);
    Py_DECREF(objtype_);
  }
  bool check(PyObject *scope) {
    return objtype_ == _get_item_type(scope, key_);
  }
};

void init_type_guards(GuardsVector &out, PyObject *in, PyObject *scope) {
  size_t len = PyObject_Length(in);
  out.reserve(len + out.size());
  for (size_t i = 0; i < len; ++i) {
    PyObject *key = PyList_GetItem(in, i); // borrows
    NULL_CHECK(key);
    out.emplace_back(new TypeCheck(key, scope));
  }
}

class ValueCheckWeakRef : public GuardCheck {
  PyObject *key_;
  PyObject *valref_;

public:
  ValueCheckWeakRef(PyObject *key, PyObject *value) {
    key_ = key;
    Py_INCREF(key_);
    valref_ = PyWeakref_NewRef(value, NULL); // new ref
    NULL_CHECK(valref_);
  }
  ~ValueCheckWeakRef() {
    Py_DECREF(key_);
    Py_DECREF(valref_);
  }
  bool check(PyObject *scope) {
    PyObject *value1 = PyWeakref_GET_OBJECT(valref_);
    // PyObject *value1 = PyWeakref_GetObject(valref_);
    // NULL_CHECK(value1);
    PyObject *value2 = PyObject_GetItem(scope, key_); // new ref
    Py_XDECREF(value2);
    return value1 == value2;
  }
};

class ValueCheck : public GuardCheck {
  PyObject *key_;
  PyObject *value_;

public:
  ValueCheck(PyObject *key, PyObject *value) {
    key_ = key;
    Py_INCREF(key_);
    value_ = value;
    Py_INCREF(value_);
  }
  ~ValueCheck() {
    Py_DECREF(key_);
    Py_DECREF(value_);
  }
  bool check(PyObject *scope) {
    PyObject *value2 = PyObject_GetItem(scope, key_); // new ref
    Py_XDECREF(value2);
    return value_ == value2;
  }
};

void init_value_guards(GuardsVector &out, PyObject *in, PyObject *scope) {
  size_t len = PyObject_Length(in);
  out.reserve(len + out.size());
  for (size_t i = 0; i < len; ++i) {
    PyObject *key = PyList_GetItem(in, i); // borrows
    NULL_CHECK(key);
    PyObject *value = PyObject_GetItem(scope, key); // new ref
    NULL_CHECK(value);
    if (value == Py_None || value == Py_True || value == Py_False) {
      out.emplace_back(new ValueCheck(key, value));
    } else {
      out.emplace_back(new ValueCheckWeakRef(key, value));
    }
    Py_DECREF(value);
  }
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
  if (check_guards(locals_checks_, f_locals) &&
      check_guards(globals_checks_, f_globals)) {
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

  init_value_guards(locals_checks_, value_locals, f_locals);
  init_value_guards(globals_checks_, value_globals, f_globals);
  init_type_guards(locals_checks_, type_locals, f_locals);
  init_type_guards(globals_checks_, type_globals, f_globals);

  Py_DECREF(value_locals);
  Py_DECREF(value_globals);
  Py_DECREF(type_locals);
  Py_DECREF(type_globals);
  code_ = (PyCodeObject *)code;
}
