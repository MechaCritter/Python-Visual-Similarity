// Access to the search structures defined by the vendored hnswlib bindings.
#pragma once

#include <pybind11/pybind11.h>

/*
 * The vendored bindings define the Index and BFIndex classes this extension is
 * built on, and end with a module entry point declared through the
 * PYBIND11_PLUGIN macro. That entry point names its module "hnswlib" and is
 * deprecated since pybind11 3.0, so it is not the one this extension exports.
 *
 * Redefining the macro across the include turns the upstream entry point into
 * an inline function nothing references, which no binary carries and no
 * interpreter can reach. The module exported here is the one declared with
 * PYBIND11_MODULE in _hnswlib.cpp, and the vendored file stays byte-identical
 * with its upstream source.
 */
#ifdef PYBIND11_PLUGIN
#    undef PYBIND11_PLUGIN
#endif
#define PYBIND11_PLUGIN(name) inline PyObject *unused_upstream_plugin()

#if defined(_MSC_VER)
#    pragma warning(push)
// C4996: the upstream entry point builds its module through a deprecated
// pybind11 constructor.
#    pragma warning(disable : 4996)
#elif defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "../_vendored/hnswlib/python-bindings/bindings.cpp"

#if defined(_MSC_VER)
#    pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#undef PYBIND11_PLUGIN
