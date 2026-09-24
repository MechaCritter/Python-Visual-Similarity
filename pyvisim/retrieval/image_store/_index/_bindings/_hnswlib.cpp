// Module exporting the search structures the image store is built on.
#include <pybind11/pybind11.h>

#include "pyvisim_index.h"


PYBIND11_MODULE(_hnswlib, m) {
    m.doc() = "hnswlib search structures, extended with the surface pyvisim searches through.";

    py::class_<PyvisimIndex>(m, "Index")
        .def(py::init(&PyvisimIndex::createFromParams), py::arg("params"))
        /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&PyvisimIndex::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &PyvisimIndex::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("reset_index", &PyvisimIndex::resetIndex)
        .def("knn_query",
            &PyvisimIndex::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items",
            &PyvisimIndex::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_items",
            &PyvisimIndex::getData,
            py::arg("ids") = py::none(),
            py::arg("return_type") = "numpy")
        .def("get_ids_list", &PyvisimIndex::getIdsList)
        .def("set_ef", &PyvisimIndex::set_ef, py::arg("ef"))
        .def("set_num_threads", &PyvisimIndex::set_num_threads, py::arg("num_threads"))
        .def("index_file_size", &PyvisimIndex::indexFileSize)
        .def("save_index", &PyvisimIndex::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &PyvisimIndex::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("mark_deleted", &PyvisimIndex::markDeleted, py::arg("label"))
        .def("unmark_deleted", &PyvisimIndex::unmarkDeleted, py::arg("label"))
        .def("resize_index", &PyvisimIndex::resizeIndex, py::arg("new_size"))
        .def("get_max_elements", &PyvisimIndex::getMaxElements)
        .def("get_current_count", &PyvisimIndex::getCurrentCount)
        .def_readonly("space", &PyvisimIndex::space_name)
        .def_readonly("dim", &PyvisimIndex::dim)
        .def_readwrite("num_threads", &PyvisimIndex::num_threads_default)
        .def_property("ef",
            [](const PyvisimIndex &index) {
                return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
            },
            [](PyvisimIndex &index, const size_t ef_) {
                index.default_ef = ef_;
                if (index.appr_alg)
                    index.appr_alg->ef_ = ef_;
            })
        .def_property_readonly("max_elements", [](const PyvisimIndex &index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const PyvisimIndex &index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const PyvisimIndex &index) {
            return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M", [](const PyvisimIndex &index) {
            return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const PyvisimIndex &index) {  // __getstate__
                /* Return a dict, wrapped in a tuple, fully encoding the state of the index */
                return py::make_tuple(index.getIndexParams());
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return PyvisimIndex::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const PyvisimIndex &index) {
            return "<pyvisim.Index(space='" + index.space_name + "', dim="
                + std::to_string(index.dim) + ")>";
        });

    py::class_<PyvisimBFIndex>(m, "BFIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &PyvisimBFIndex::init_new_index, py::arg("max_elements"))
        .def("reset_index", &PyvisimBFIndex::resetIndex)
        .def("knn_query",
            &PyvisimBFIndex::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items", &PyvisimBFIndex::addItems, py::arg("data"), py::arg("ids") = py::none())
        .def("get_items",
            &PyvisimBFIndex::getData,
            py::arg("ids") = py::none(),
            py::arg("return_type") = "numpy")
        .def("delete_vector", &PyvisimBFIndex::deleteVector, py::arg("label"))
        .def("set_num_threads", &PyvisimBFIndex::set_num_threads, py::arg("num_threads"))
        .def("save_index", &PyvisimBFIndex::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &PyvisimBFIndex::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0)
        .def("get_max_elements", &PyvisimBFIndex::getMaxElements)
        .def("get_current_count", &PyvisimBFIndex::getCurrentCount)
        .def_readonly("space", &PyvisimBFIndex::space_name)
        .def_readonly("dim", &PyvisimBFIndex::dim)
        .def_readwrite("num_threads", &PyvisimBFIndex::num_threads_default)

        .def("__repr__", [](const PyvisimBFIndex &index) {
            return "<pyvisim.BFIndex(space='" + index.space_name + "', dim="
                + std::to_string(index.dim) + ")>";
        });
}
