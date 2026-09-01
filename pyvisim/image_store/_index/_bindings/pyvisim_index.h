// Search structures extending the vendored hnswlib ones with what pyvisim needs.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "upstream_index.h"


/**
 * HNSW graph index that can drop its graph and be built again.
 *
 * The upstream index refuses to initialise a second graph, so replacing a
 * gallery means freeing the current one first.
 */
class PyvisimIndex : public Index<float> {
 public:
    PyvisimIndex(const std::string &space_name, const int dim) : Index<float>(space_name, dim) {
        cur_l = 0;
    }


    /**
     * Free the graph, leaving the index ready for another init_index call.
     */
    void resetIndex() {
        if (appr_alg) {
            delete appr_alg;
            appr_alg = nullptr;
        }
        cur_l = 0;
        ep_added = true;
        index_inited = false;
    }


    /**
     * Build an index from the parameters getIndexParams produced.
     *
     * The parameters are read by the upstream index, whose search structure is
     * then taken over, so unpickling stays the upstream implementation.
     */
    static PyvisimIndex *createFromParams(const py::dict &params) {
        std::unique_ptr<Index<float>> source(Index<float>::createFromParams(params));
        std::unique_ptr<PyvisimIndex> index(new PyvisimIndex(source->space_name, source->dim));
        index->adoptState(*source);
        return index.release();
    }


    /**
     * Copy an index through the parameters it serialises itself into.
     */
    static PyvisimIndex *createFromIndex(const PyvisimIndex &index) {
        return createFromParams(index.getIndexParams());
    }

 private:
    /**
     * Take the search structure and the build state of another index over.
     *
     * The source is left without a graph and without a metric space, so
     * destroying it does not free what was taken over.
     */
    void adoptState(Index<float> &source) {
        delete l2space;
        l2space = source.l2space;
        source.l2space = nullptr;
        appr_alg = source.appr_alg;
        source.appr_alg = nullptr;

        index_inited = source.index_inited;
        ep_added = source.ep_added;
        normalize = source.normalize;
        num_threads_default = source.num_threads_default;
        seed = source.seed;
        default_ef = source.default_ef;
        // An index that was serialised before it was initialised holds no
        // element count to take over.
        cur_l = source.index_inited ? source.cur_l : 0;
    }
};


/**
 * Brute-force index that can drop its vectors and hand them back.
 *
 * The upstream index keeps the vectors it is given but exposes no way to read
 * them, which leaves every caller that needs them holding a second copy.
 */
class PyvisimBFIndex : public BFIndex<float> {
 public:
    PyvisimBFIndex(const std::string &space_name, const int dim) : BFIndex<float>(space_name, dim) {
        cur_l = 0;
    }


    /**
     * Free the stored vectors, leaving the index ready for another init_index
     * call.
     */
    void resetIndex() {
        if (alg) {
            delete alg;
            alg = nullptr;
        }
        cur_l = 0;
        index_inited = false;
    }


    /**
     * Read the vectors stored under the labels of a one-dimensional array back,
     * as a matrix ("numpy") or as a list of lists ("list").
     *
     * Every vector is copied out of the index, so the caller receives a block
     * of its own and the index stays the only owner of what it stores.
     */
    py::object getData(py::object ids = py::none(), std::string return_type = "numpy") {
        if (return_type != "numpy" && return_type != "list") {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        if (!alg) {
            throw std::runtime_error("The index is not initiated.");
        }

        std::vector<std::vector<float>> data;
        for (auto label : requestedLabels(ids)) {
            data.push_back(getDataByLabel(label));
        }

        if (return_type == "list") {
            return py::cast(data);
        }
        return py::array_t<float, py::array::c_style | py::array::forcecast>(py::cast(data));
    }

 private:
    /**
     * Read the labels a get_items call asks for out of its argument.
     */
    static std::vector<size_t> requestedLabels(const py::object &ids) {
        if (ids.is_none()) {
            return std::vector<size_t>();
        }
        py::array_t<size_t, py::array::c_style | py::array::forcecast> labels(ids);
        if (labels.request().ndim == 0) {
            throw std::invalid_argument(
                "get_items accepts a list of indices and returns a list of vectors");
        }
        return std::vector<size_t>(labels.data(), labels.data() + labels.size());
    }


    /**
     * Copy the vector stored under one label out of the index.
     *
     * The vectors sit in a single block, each one followed by its label, so a
     * label is resolved to its slot before the vector is read off it.
     */
    std::vector<float> getDataByLabel(size_t label) const {
        auto slot = alg->dict_external_to_internal.find(label);
        if (slot == alg->dict_external_to_internal.end()) {
            throw std::runtime_error("Label " + std::to_string(label) + " is not in the index");
        }
        const char *element = alg->data_ + slot->second * alg->size_per_element_;
        const float *vector = reinterpret_cast<const float *>(element);
        return std::vector<float>(vector, vector + dim);
    }
};
