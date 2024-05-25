#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <map>
#include <condition_variable>
#include <memory>
#include <queue>
#include "plugin_interface.h"
#include "plugin_loader.h"

namespace py = pybind11;

class PyPluginInterface : public PluginInterface {
public:
    using PluginInterface::PluginInterface;

    int deepspeed_py_aio_write(const torch::Tensor& buffer, 
                               const char* filename, 
                               const int block_size, 
                               const int queue_depth, 
                               const bool single_submit, 
                               const bool overlap_events, 
                               const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            deepspeed_py_aio_write,
            buffer,
            filename,
            block_size,
            queue_depth,
            single_submit,
            overlap_events,
            validate
        );
    }

    int deepspeed_py_aio_read(torch::Tensor& buffer,
                              const char* filename,
                              const int block_size,
                              const int queue_depth,
                              const bool single_submit,
                              const bool overlap_events,
                              const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            deepspeed_py_aio_read,
            buffer,
            filename,
            block_size,
            queue_depth,
            single_submit,
            overlap_events,
            validate
        );
    }

    int deepspeed_py_memcpy(torch::Tensor& dest, const torch::Tensor& src) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            deepspeed_py_memcpy,
            dest,
            src
        );
    }

    // torch::Tensor alloc(const size_t num_elem, const at::ScalarType& elem_type) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         torch::Tensor,
    //         PluginInterface,
    //         alloc,
    //         num_elem,
    //         elem_type
    //     );
    // }

    // bool free(torch::Tensor& locked_tensor) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         bool,
    //         PluginInterface,
    //         free,
    //         locked_tensor
    //     );
    // }

    // char* data_ptr() const override {
    //     PYBIND11_OVERRIDE_PURE(
    //         char*,
    //         PluginInterface,
    //         data_ptr
    //     );
    // }

    // void fini() override {
    //     PYBIND11_OVERRIDE_PURE(
    //         void,
    //         PluginInterface,
    //         fini
    //     );
    // }

    // void run() override {
    //     PYBIND11_OVERRIDE_PURE(
    //         void,
    //         PluginInterface,
    //         run
    //     );
    // }

    const int get_block_size() const override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            get_block_size
        );
    }

    const int get_queue_depth() const override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            get_queue_depth
        );
    }

    const bool get_single_submit() const override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            PluginInterface,
            get_single_submit
        );
    }

    const bool get_overlap_events() const override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            PluginInterface,
            get_overlap_events
        );
    }

    const int get_thread_count() const override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            get_thread_count
        );
    }

    int read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            read,
            buffer,
            filename,
            validate
        );
    }

    int write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            write,
            buffer,
            filename,
            validate
        );
    }

    int pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            pread,
            buffer,
            filename,
            validate,
            async
        );
    }

    int pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            pwrite,
            buffer,
            filename,
            validate,
            async
        );
    }

    int sync_pread(torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            sync_pread,
            buffer,
            filename
        );
    }

    int sync_pwrite(const torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            sync_pwrite,
            buffer,
            filename
        );
    }

    int async_pread(torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            async_pread,
            buffer,
            filename
        );
    }

    int async_pwrite(const torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            async_pwrite,
            buffer,
            filename
        );
    }

    torch::Tensor new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) override {
        PYBIND11_OVERRIDE_PURE(
            torch::Tensor,
            PluginInterface,
            new_cpu_locked_tensor,
            num_elem,
            example_tensor
        );
    }

    bool free_cpu_locked_tensor(torch::Tensor& tensor) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            PluginInterface,
            free_cpu_locked_tensor,
            tensor
        );
    }

    int wait() override {
        PYBIND11_OVERRIDE_PURE(
            int,
            PluginInterface,
            wait
        );
    }

    // void _stop_threads() override {
    //     PYBIND11_OVERRIDE_PURE(
    //         void,
    //         PluginInterface,
    //         _stop_threads
    //     );
    // }

    // void _schedule_aio_work(std::shared_ptr<io_op_desc_t> scheduled_op) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         void,
    //         PluginInterface,
    //         _schedule_aio_work,
    //         scheduled_op
    //     );
    // }

    // bool _is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes) override {
    //     PYBIND11_OVERRIDE_PURE(
    //         bool,
    //         PluginInterface,
    //         _is_valid_parallel_aio_op,
    //         read_op,
    //         num_bytes
    //     );
    // }
};

PYBIND11_MODULE(_C, m) {
    py::class_<PluginInterface, PyPluginInterface>(m, "PluginInterface")
        .def(py::init<>())
        .def("deepspeed_py_aio_write", &PluginInterface::deepspeed_py_aio_write)
        .def("deepspeed_py_aio_read", &PluginInterface::deepspeed_py_aio_read)
        .def("deepspeed_py_memcpy", &PluginInterface::deepspeed_py_memcpy)
        // .def("alloc", &PluginInterface::alloc)
        // .def("free", &PluginInterface::free)
        // .def("data_ptr", &PluginInterface::data_ptr)
        // .def("fini", &PluginInterface::fini)
        // .def("run", &PluginInterface::run)

        .def("get_block_size", &PluginInterface::get_block_size)
        .def("get_queue_depth", &PluginInterface::get_queue_depth)
        .def("get_single_submit", &PluginInterface::get_single_submit)
        .def("get_overlap_events", &PluginInterface::get_overlap_events)
        .def("get_thread_count", &PluginInterface::get_thread_count)

        .def("read", &PluginInterface::read)
        .def("write", &PluginInterface::write)

        .def("pread", &PluginInterface::pread)
        .def("pwrite", &PluginInterface::pwrite)

        .def("sync_pread", &PluginInterface::sync_pread)
        .def("sync_pwrite", &PluginInterface::sync_pwrite)
        .def("async_pread", &PluginInterface::async_pread)
        .def("async_pwrite", &PluginInterface::async_pwrite)

        .def("new_cpu_locked_tensor", &PluginInterface::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &PluginInterface::free_cpu_locked_tensor)

        .def("wait", &PluginInterface::wait)

        // .def("_stop_threads", &PluginInterface::_stop_threads)
        // .def("_schedule_aio_work", &PluginInterface::_schedule_aio_work)
        // .def("_is_valid_parallel_aio_op", &PluginInterface::_is_valid_parallel_aio_op);

    py::class_<PluginLoader>(m, "PluginLoader")
        .def(py::init<>())
        .def("register_plugin", &PluginLoader::register_plugin)
        .def("get_plugin", &PluginLoader::get_plugin, py::return_value_policy::reference);
}
