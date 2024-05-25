#pragma once

#include <pybind11/pybind11.h>
#include <deepspeed_aio_common.h> // this is from common folder where async io functions are defined
#include <stdlib.h>
#include <torch/extension.h>
#include <map>
#include <condition_variable>
#include <memory>
#include <queue>

// This is the base class where every functionality mentioned here is compulsory to be implemented in the plugins created

class PluginInterface {
public:
    virtual ~PluginInterface() = default;

    // 1. from deepspeed_py_aio.h
    virtual int deepspeed_py_aio_write(const torch::Tensor& buffer,
                                       const char* filename,
                                       const int block_size,
                                       const int queue_depth,
                                       const bool single_submit,
                                       const bool overlap_events,
                                       const bool validate) = 0;

    virtual int deepspeed_py_aio_read(torch::Tensor& buffer,
                                      const char* filename,
                                      const int block_size,
                                      const int queue_depth,
                                      const bool single_submit,
                                      const bool overlap_events,
                                      const bool validate) = 0;

    // 2. from deepspeed_py_copy.h
    virtual int deepspeed_py_memcpy(torch::Tensor& dest, const torch::Tensor& src) = 0;

    // 3. from deepspeed_pin_tensor.h
    struct deepspeed_pin_tensor_t {
        std::map<void*, size_t> _locked_tensors;

        virtual ~deepspeed_pin_tensor_t() {}

        virtual torch::Tensor alloc(const size_t num_elem, const at::ScalarType& elem_type) = 0;
        virtual bool free(torch::Tensor& locked_tensor) = 0;
    };

    // virtual deepspeed_pin_tensor_t* get_pin_tensor() = 0;

    // 4. from deepspeed_aio_thread.h
    struct io_op_desc_t {
        const bool _read_op;
        torch::Tensor _buffer;
        int _fd;
        const std::string _filename;
        const long long int _num_bytes;
        torch::Tensor _cpu_buffer;
        torch::Tensor _contiguous_buffer;
        const bool _validate;

        io_op_desc_t(const bool read_op,
                    const torch::Tensor& buffer,
                    const int fd,
                    const char* filename,
                    const long long int num_bytes,
                    const bool validate)
            : _read_op(read_op),
            _buffer(buffer),
            _fd(fd),
            _filename(filename),
            _num_bytes(num_bytes),
            _validate(validate) {}

        virtual ~io_op_desc_t() {}

        virtual char* data_ptr() const = 0;
        virtual void fini() = 0;
    };

    // virtual io_op_desc_t* create_io_op_desc_t(bool read_op,
    //                                           const torch::Tensor& buffer,
    //                                           int fd,
    //                                           const char* filename,
    //                                           long long int num_bytes,
    //                                           bool validate) = 0;

    struct thread_sync_t {
        std::mutex _mutex;
        std::condition_variable _cond_var;
    };

    // virtual thread_sync_t* get_thread_sync() = 0;

    struct deepspeed_aio_thread_t {
        const int _tid;
        deepspeed_aio_config_t& _aio_config;

        std::unique_ptr<struct aio_context> _aio_ctxt;
        std::queue<std::shared_ptr<io_op_desc_t>> _work_queue;
        std::queue<std::shared_ptr<io_op_desc_t>> _complete_queue;

        bool _time_to_exit;

        struct thread_sync_t _work_sync;
        struct thread_sync_t _complete_sync;

        deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config)
            : _tid(tid),
              _aio_config(aio_config) {}

        virtual ~deepspeed_aio_thread_t {}

        virtual void run() = 0;
    };

    // virtual deepspeed_aio_thread_t* get_aio_thread(const int tid, deepspeed_aio_config_t& aio_config) = 0;


    // 5. from deepspeed_py_aio_handle.h
    struct deepspeed_aio_handle_t {
        std::unique_ptr<struct aio_context> _aio_ctxt;
        const bool _single_submit;
        const bool _overlap_events;
        const int _num_threads;
        deepspeed_aio_config_t _aio_config;

        std::vector<std::shared_ptr<deepspeed_aio_thread_t>> _thread_contexts;
        std::vector<std::thread> _threads;
        int _num_pending_ops;
        std::unique_ptr<deepspeed_pin_tensor_t> _pinned_tensor_mgr;

        deepspeed_aio_handle_t(const int block_size,
                            const int queue_depth,
                            const bool single_submit,
                            const bool overlap_events,
                            const int num_threads)
            :_aio_ctxt(std::make_unique<aio_context>(block_size, queue_depth)),
            _single_submit(single_submit),
            _overlap_events(overlap_events),
            _num_threads(num_threads),
            _aio_config(block_size, queue_depth, single_submit, overlap_events, false),
            _num_pending_ops(0),
            _pinned_tensor_mgr(std::make_unique<deepspeed_pin_tensor_t>()) {}

        virtual ~deepspeed_aio_handle_t() = default;

        virtual const int get_block_size() const = 0;
        virtual const int get_queue_depth() const = 0;
        virtual const bool get_single_submit() const = 0;
        virtual const bool get_overlap_events() const = 0;
        virtual const int get_thread_count() const = 0;

        virtual int read(torch::Tensor& buffer, const char* filename, const bool validate) = 0;

        virtual int write(const torch::Tensor& buffer, const char* filename, const bool validate) = 0;

        virtual int pread(const torch::Tensor& buffer,
                const char* filename,
                const bool validate,
                const bool async) = 0;

        virtual int pwrite(const torch::Tensor& buffer,
                const char* filename,
                const bool validate,
                const bool async) = 0;

        virtual int sync_pread(torch::Tensor& buffer, const char* filename) = 0;

        virtual int sync_pwrite(const torch::Tensor& buffer, const char* filename) = 0;

        virtual int async_pread(torch::Tensor& buffer, const char* filename) = 0;

        virtual int async_pwrite(const torch::Tensor& buffer, const char* filename) = 0;

        // TODO: Make API's args to be shape and dtype.
        virtual torch::Tensor new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor) = default;

        virtual bool free_cpu_locked_tensor(torch::Tensor&) = 0;

        virtual int wait() = 0;

        virtual void _stop_threads() = 0;

        virtual void _schedule_aio_work(std::shared_ptr<io_op_desc_t> scheduled_op) = 0;

        std::shared_ptr<io_op_desc_t> _wait_for_aio_work();

        virtual bool _is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes) = 0;
    };

    // virtual deepspeed_aio_handle_t* get_aio_handle(const int block_size,
    //                                                 const int queue_depth,
    //                                                 const bool single_submit,
    //                                                 const bool overlap_events,
    //                                                 const int num_threads) = 0;

};