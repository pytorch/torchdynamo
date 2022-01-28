#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <torch/extension.h>

namespace {

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  bool grad_mode_enabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        grad_mode_enabled(at::GradMode::is_enabled()) {}
};

class TensorCheck {
public:
  TensorCheck(const LocalState &state, PyTypeObject *pt, const at::Tensor &v,
              bool dynamic_shapes)
      : pytype(pt), dispatch_key_(state.apply(v.key_set()).raw_repr()),
        dtype_(v.dtype().toScalarType()),
        requires_grad_(state.grad_mode_enabled && v.requires_grad()),
        dynamic_shapes_(dynamic_shapes) {
    auto ndim = v.ndimension();
    const auto &sizes = v.sizes();
    const auto &strides = v.strides();
    sizes_.reserve(ndim);
    strides_.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      sizes_.emplace_back(sizes[i]);
      strides_.emplace_back(strides[i]);
    }
  }

  bool check(const LocalState &state, const at::Tensor &v) {
    if (dispatch_key_ != state.apply(v.key_set()).raw_repr() ||
        dtype_ != v.dtype().toScalarType() ||
        requires_grad_ != (state.grad_mode_enabled && v.requires_grad())) {
      return false;
    }
    size_t ndim = static_cast<size_t>(v.ndimension());
    if (ndim != sizes_.size()) {
      return false;
    }
    if (!dynamic_shapes_) {
      const auto &sizes = v.sizes();
      const auto &strides = v.strides();
      for (size_t i = 0; i < ndim; ++i) {
        if (sizes_[i] != sizes[i] || strides_[i] != strides[i]) {
          return false;
        }
      }
    }
    return true;
  }

  PyTypeObject *pytype;

private:
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  at::ScalarType dtype_;
  bool requires_grad_;
  bool dynamic_shapes_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

typedef std::vector<TensorCheck> ChecksList;

typedef struct {
  PyObject_HEAD;
  ChecksList *checks;
} TensorGuards;

static void TensorGuards_dealloc(TensorGuards *self) {
  if (self->checks != NULL) {
    delete self->checks;
    self->checks = NULL;
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *TensorGuards_new(PyTypeObject *type, PyObject *args,
                                  PyObject *kwds) {
  TensorGuards *self = (TensorGuards *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->checks = new ChecksList();
  }
  return (PyObject *)self;
}

static int TensorGuards_init(TensorGuards *self, PyObject *args,
                             PyObject *kwds) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return -1;
  }
  PyObject *dynamic_shapes_py = PyDict_GetItemString(kwds, "dynamic_shapes");
  if (dynamic_shapes_py == NULL) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_shapes=...");
    return -1;
  }
  bool dynamic_shapes = PyObject_IsTrue(dynamic_shapes_py);

  auto &checks = *self->checks;
  ssize_t len = PyTuple_GET_SIZE(args);
  checks.reserve(len);
  LocalState state;
  for (ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyTuple_GET_ITEM(args, i);
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      return -1;
    }
    checks.emplace_back(TensorCheck(state, Py_TYPE(item),
                                    THPVariable_Unpack(item), dynamic_shapes));
  }
  return 0;
}

PyObject *TensorGuards_check(TensorGuards *self, PyObject *args) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return NULL;
  }
  auto &checks = *self->checks;
  ssize_t len = PyTuple_GET_SIZE(args);

  if (static_cast<ssize_t>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return NULL;
  }

  LocalState state;

  for (ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyTuple_GET_ITEM(args, i);
    if (Py_TYPE(item) != checks[i].pytype) {
      Py_RETURN_FALSE;
    }
    if (!checks[i].check(state, THPVariable_Unpack(item))) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

static PyMethodDef TensorGuards_methods[] = {
    {"check", (PyCFunction)TensorGuards_check, METH_VARARGS, ""},
    {NULL} /* Sentinel */
};

static PyTypeObject TensorGuardsType = {
    // NOLINTNEXTLINE
    PyVarObject_HEAD_INIT(NULL, 0)};

static PyObject *check_type_id(PyObject *dummy, PyObject *args) {
  // faster `lambda obj, expected: id(type(obj)) == expected`
  PyObject *obj;
  unsigned long expected;
  if (!PyArg_ParseTuple(args, "Ok", &obj, &expected)) {
    return NULL;
  }
  if (Py_TYPE(obj) == (void *)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject *check_obj_id(PyObject *dummy, PyObject *args) {
  // faster `lambda obj, expected: id(obj) == expected`
  PyObject *obj;
  unsigned long expected;
  if (!PyArg_ParseTuple(args, "Ok", &obj, &expected)) {
    return NULL;
  }
  if (obj == (void *)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, NULL},
    {"check_obj_id", check_obj_id, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT, "_guards",
                                     "Module containing checks on tensors", -1,
                                     _methods};

} // namespace

PyMODINIT_FUNC PyInit__guards(void) {
  // initialize TensorGuardsType
  TensorGuardsType.tp_name = "torchdynamo._guards.TensorGuards";
  TensorGuardsType.tp_basicsize = sizeof(TensorGuards);
  TensorGuardsType.tp_itemsize = 0;
  TensorGuardsType.tp_dealloc = (destructor)TensorGuards_dealloc;
  TensorGuardsType.tp_flags = Py_TPFLAGS_DEFAULT;
  TensorGuardsType.tp_doc = "Check properties of a torch.Tensor";
  TensorGuardsType.tp_methods = TensorGuards_methods;
  TensorGuardsType.tp_init = (initproc)TensorGuards_init;
  TensorGuardsType.tp_new = TensorGuards_new;

  PyObject *m;
  if (PyType_Ready(&TensorGuardsType) < 0)
    return NULL;

  m = PyModule_Create(&_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&TensorGuardsType);
  if (PyModule_AddObject(m, "TensorGuards", (PyObject *)&TensorGuardsType) <
      0) {
    Py_DECREF(&TensorGuardsType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
