#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "dstar.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(dstar_lite, m) {
    py::class_<Dstar>(m, "Dstar")
        .def(py::init<const py::array_t<double, py::array::c_style>&, int, bool>(), py::arg("mapArg").noconvert(true), py::arg("maxStepsArg"), py::arg("scale_diag_cost"))
        .def("init", &Dstar::init)
        .def("updateStart", &Dstar::updateStart)
        .def("updateCells", &Dstar::updateCells, py::arg("indexes").noconvert(true), py::arg("values").noconvert(true))
        .def("replan", &Dstar::replan)
        .def("getPath", &Dstar::getPath)
        // .def("getMap", &Dstar::getMap)
        .def("getGValues", &Dstar::getGValues)
        .def("getRHSValues", &Dstar::getRHSValues)
        .def("getKeys", &Dstar::getKeys)
        .def("printMap", &Dstar::printMap);

        
}
