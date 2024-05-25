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

    torch::Tensor deepspeed_pin_tensor_t::alloc(const size_t num_elem, const at::ScalarType& elem_type) override {
        PYBIND11_OVERRIDE_PURE(
            torch::Tensor,
            deepspeed_pin_tensor_t,
            alloc,
            num_elem,
            elem_type
        );
    }

    bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            deepspeed_pin_tensor_t,
            free,
            locked_tensor
        );
    }

    char* io_op_desc_t::data_ptr() const override {
        PYBIND11_OVERRIDE_PURE(
            char*,
            io_op_desc_t,
            data_ptr
        );
    }

    void io_op_desc_t::fini() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            io_op_desc_t,
            fini
        );
    }

    void deepspeed_aio_thread_t::run() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            deepspeed_aio_thread_t,
            run
        );
    }

    const int deepspeed_aio_handle_t::get_block_size() const override {
        PYBIND11_OVERRIDE_PURE(
            const int,
            deepspeed_aio_handle_t,
            get_block_size
        );
    }

    const int deepspeed_aio_handle_t::get_queue_depth() const override {
        PYBIND11_OVERRIDE_PURE(
            const int,
            deepspeed_aio_handle_t,
            get_queue_depth
        );
    }

    const bool deepspeed_aio_handle_t::get_single_submit() const override {
        PYBIND11_OVERRIDE_PURE(
            const bool,
            deepspeed_aio_handle_t,
            get_single_submit
        );
    }

    const bool deepspeed_aio_handle_t::get_overlap_events() const override {
        PYBIND11_OVERRIDE_PURE(
            const bool,
            deepspeed_aio_handle_t,
            get_overlap_events
        );
    }

    const int deepspeed_aio_handle_t::get_thread_count() const override {
        PYBIND11_OVERRIDE_PURE(
            const int,
            deepspeed_aio_handle_t,
            get_thread_count
        );
    }

    int deepspeed_aio_handle_t::read(torch::Tensor& buffer, const char* filename, const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            read,
            buffer,
            filename,
            validate
        );
    }

    int deepspeed_aio_handle_t::write(const torch::Tensor& buffer, const char* filename, const bool validate) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            write,
            buffer,
            filename,
            validate
        );
    }

    int deepspeed_aio_handle_t::pread(const torch::Tensor& buffer,
                                      const char* filename,
                                      const bool validate,
                                      const bool async) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            pread,
            buffer,
            filename,
            validate,
            async
        );
    }

    int deepspeed_aio_handle_t::pwrite(const torch::Tensor& buffer,
                                       const char* filename,
                                       const bool validate,
                                       const bool async) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            pwrite,
            buffer,
            filename,
            validate,
            async
        );
    }

    int deepspeed_aio_handle_t::sync_pread(torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            sync_pread,
            buffer,
            filename
        );
    }

    int deepspeed_aio_handle_t::sync_pwrite(const torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            sync_pwrite,
            buffer,
            filename
        );
    }

    int deepspeed_aio_handle_t::async_pread(torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            async_pread,
            buffer,
            filename
        );
    }

    int deepspeed_aio_handle_t::async_pwrite(torch::Tensor& buffer, const char* filename) override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            async_pwrite,
            buffer,
            filename
        );
    }

    torch::Tensor deepspeed_aio_handle_t::new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) override {
        PYBIND11_OVERRIDE_PURE(
            torch::Tensor,
            deepspeed_aio_handle_t,
            new_cpu_locked_tensor,
            num_elem,
            example_tensor
        );
    }

    bool deepspeed_aio_handle_t::free_cpu_locked_tensor(torch::Tensor& tensor) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            deepspeed_aio_handle_t,
            free_cpu_locked_tensor,
            tensor
        );
    }

    int deepspeed_aio_handle_t::wait() override {
        PYBIND11_OVERRIDE_PURE(
            int,
            deepspeed_aio_handle_t,
            wait
        );
    }

    void deepspeed_aio_handle_t::_stop_threads() override {
        PYBIND11_OVERRIDE_PURE(
            void,
            deepspeed_aio_handle_t,
            _stop_threads
        );
    }

    void deepspeed_aio_handle_t::_schedule_aio_work(std::shared_ptr<io_op_desc_t> scheduled_op) override {
        PYBIND11_OVERRIDE_PURE(
            void,
            deepspeed_aio_handle_t,
            _schedule_aio_work,
            scheduled_op
        );
    }

    bool deepspeed_aio_handle_t::_is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            deepspeed_aio_handle_t,
            _is_valid_parallel_aio_op,
            read_op,
            num_bytes
        );
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aio_read", &deepspeed_py_aio_read, "DeepSpeed Asynchronous I/O Read");
    m.def("aio_write", &deepspeed_py_aio_write, "DeepSpeed Asynchronous I/O Write");
    m.def("deepspeed_memcpy", &deepspeed_py_memcpy, "DeepSpeed Memory Copy");

    py::class_<PluginInterface, PyPluginInterface>(m, "PluginInterface")
        .def(py::init<>())
        .def("deepspeed_py_aio_write", &PluginInterface::deepspeed_py_aio_write)
        .def("deepspeed_py_aio_read", &PluginInterface::deepspeed_py_aio_read)
        .def("deepspeed_py_memcpy", &PluginInterface::deepspeed_py_memcpy);

    py::class_<PluginInterface::deepspeed_pin_tensor_t, std::shared_ptr<PluginInterface::deepspeed_pin_tensor_t>>(m, "deepspeed_pin_tensor_t")
        .def(py::init<>())
        .def("alloc", &PluginInterface::deepspeed_pin_tensor_t::alloc)
        .def("free", &PluginInterface::deepspeed_pin_tensor_t::free);

    py::class_<PluginInterface::io_op_desc_t, std::shared_ptr<PluginInterface::io_op_desc_t>>(m, "io_op_desc_t")
        .def(py::init<bool, const torch::Tensor&, int, const char*, long long int, bool>())
        .def("data_ptr", &PluginInterface::io_op_desc_t::data_ptr)
        .def("fini", &PluginInterface::io_op_desc_t::fini);

    py::class_<PluginInterface::deepspeed_aio_thread_t, std::shared_ptr<PluginInterface::deepspeed_aio_thread_t>>(m, "deepspeed_aio_thread_t")
        .def(py::init<int, deepspeed_aio_config_t&>())
        .def("run", &PluginInterface::deepspeed_aio_thread_t::run);

    py::class_<PluginInterface::deepspeed_aio_handle_t, std::shared_ptr<PluginInterface::deepspeed_aio_handle_t>>(m, "deepspeed_aio_handle_t")
        .def(py::init<int, int, bool, bool, int>())
        .def("get_block_size", &PluginInterface::deepspeed_aio_handle_t::get_block_size)
        .def("get_queue_depth", &PluginInterface::deepspeed_aio_handle_t::get_queue_depth)
        .def("get_single_submit", &PluginInterface::deepspeed_aio_handle_t::get_single_submit)
        .def("get_overlap_events", &PluginInterface::deepspeed_aio_handle_t::get_overlap_events)
        .def("get_thread_count", &PluginInterface::deepspeed_aio_handle_t::get_thread_count)
        .def("read", &PluginInterface::deepspeed_aio_handle_t::read)
        .def("write", &PluginInterface::deepspeed_aio_handle_t::write)
        .def("pread", &PluginInterface::deepspeed_aio_handle_t::pread)
        .def("pwrite", &PluginInterface::deepspeed_aio_handle_t::pwrite)
        .def("sync_pread", &PluginInterface::deepspeed_aio_handle_t::sync_pread)
        .def("sync_pwrite", &PluginInterface::deepspeed_aio_handle_t::sync_pwrite)
        .def("async_pread", &PluginInterface::deepspeed_aio_handle_t::async_pread)
        .def("async_pwrite", &PluginInterface::deepspeed_aio_handle_t::async_pwrite)
        .def("new_cpu_locked_tensor", &PluginInterface::deepspeed_aio_handle_t::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &PluginInterface::deepspeed_aio_handle_t::free_cpu_locked_tensor)
        .def("wait", &PluginInterface::deepspeed_aio_handle_t::wait)
        .def("_stop_threads", &PluginInterface::deepspeed_aio_handle_t::_stop_threads)
        .def("_schedule_aio_work", &PluginInterface::deepspeed_aio_handle_t::_schedule_aio_work)
        .def("_is_valid_parallel_aio_op", &PluginInterface::deepspeed_aio_handle_t::_is_valid_parallel_aio_op);

    py::class_<PluginLoader>(m, "PluginLoader")
    .def(py::init<>())
    .def("register_plugin", &PluginLoader::register_plugin)
    .def("get_plugin", &PluginLoader::get_plugin, py::return_value_policy::reference);
}