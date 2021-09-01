
#include "_caching.h"

extern "C" PyCodeObject *get_cached_code(CacheEntry *cache, PyObject *f_locals,
                                         PyObject *f_globals) {
  if (cache == NULL) {
    return NULL;
  }
  if(cache->guards == NULL) {
      return cache->code;
  }
  PyObject* yesno = PyObject_CallTwoArg(cache->guards, f_locals, f_globals);
  Py_DECREF(yesno);
  if(yesno == Py_True) {
      return cache->code;
  }
  return get_cached_code(cache->next, f_locals, f_globals);
}

extern "C" CacheEntry *set_cached_code(CacheEntry *cache, PyObject *f_locals,
                                       PyObject *f_globals, PyObject *result) {

  return new CacheEntry{NULL, (PyCodeObject *)result, NULL};
}
