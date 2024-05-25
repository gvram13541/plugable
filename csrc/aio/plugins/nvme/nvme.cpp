#pragma once

#include <pybind11/pybind11.h>
#include <deepspeed_aio_common.h> // this is from common folder where async io functions are defined
#include <torch/extension.h>
#include <./interface.h>

#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include <condition_variable>
#include <memory>
#include <cassert>
#include <chrono>

#include <map>
#include <queue>
#include <string>
#include <vector>

#include <assert.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#define TILE (1024 * 1024 * 1024)

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16
#else
#if defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
#endif
#endif

#define ROUND_DOWN(size, step) ((size) & ~((step) - 1))

#if defined(__AVX512__) or defined(__AVX256__)
union AVX_Data
{
#if defined(__AVX512__)
    __m512 data;
#else
    __m256 data;
#endif
};
#endif

#if defined(__ENABLE_CANN__)
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#endif

#define DEBUG_DS_AIO_READ 0
#define DEBUG_DS_AIO_WRITE 0

using namespace std;
using namespace std::chrono;

static const std::string c_library_name = "deepspeed_aio";

class NVME : public PluginInterface
{
public:
    ~NVME() override = default;
    virtual int deepspeed_py_aio_write(const torch::Tensor &buffer,
                                       const char *filename,
                                       const int block_size,
                                       const int queue_depth,
                                       const bool single_submit,
                                       const bool overlap_events,
                                       const bool validate) override
    {
        const auto start_time = std::chrono::high_resolution_clock::now();
        deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);

        const auto fd = open_file(filename, false);
        if (fd == -1)
        {
            return -1;
        }

        auto write_buffer = (char *)buffer.data_ptr();
        const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
        std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_write_bytes, write_buffer));
        std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

        if (config._overlap_events)
        {
            do_aio_operation_overlap(false, aio_ctxt, xfer_ctxt, &config, nullptr);
        }
        else
        {
            do_aio_operation_sequential(false, aio_ctxt, xfer_ctxt, &config, nullptr);
        }
        const std::chrono::duration<double> aio_time = std::chrono::high_resolution_clock::now() - start_time;

        close(fd);

        if (validate)
        {
            validate_aio_operation(false, filename, write_buffer, num_write_bytes);
        }

        const std::chrono::duration<double> fn_time =
            std::chrono::high_resolution_clock::now() - start_time;
        std::cout << "Elapsed time(usec): "
                  << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
                  << std::endl;
        return 0;
    }

    virtual int deepspeed_py_aio_read(torch::Tensor &buffer,
                                      const char *filename,
                                      const int block_size,
                                      const int queue_depth,
                                      const bool single_submit,
                                      const bool overlap_events,
                                      const bool validate) override
    {
        const auto start_time = std::chrono::high_resolution_clock::now();
        long long num_file_bytes;
        if (-1 == get_file_size(filename, num_file_bytes))
        {
            const auto error_code = errno;
            report_file_error(filename, " fstat for read", error_code);
            return -1;
        }

        deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);
        const auto fd = open_file(filename, true);
        if (fd == -1)
        {
            return -1;
        }

        auto read_buffer = (char *)buffer.data_ptr();
        assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

        std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_file_bytes, read_buffer));
        std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

        if (config._overlap_events)
        {
            do_aio_operation_overlap(true, aio_ctxt, xfer_ctxt, &config, nullptr);
        }
        else
        {
            do_aio_operation_sequential(true, aio_ctxt, xfer_ctxt, &config, nullptr);
        }
        const std::chrono::duration<double> aio_time =
            std::chrono::high_resolution_clock::now() - start_time;

        close(fd);

        if (validate)
        {
            validate_aio_operation(true, filename, read_buffer, num_file_bytes);
        }

        const std::chrono::duration<double> fn_time =
            std::chrono::high_resolution_clock::now() - start_time;
        std::cout << "Elapsed time(usec): "
                  << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
                  << std::endl;
        return 0;
    }

    static void helper_memcpy_1(float *dest, float *src, size_t param_size)
    {
        size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

        rounded_size = ROUND_DOWN(param_size, SIMD_WIDTH);

        for (size_t t = 0; t < rounded_size; t += TILE)
        {
            size_t copy_size = TILE;
            if ((t + TILE) > rounded_size)
                copy_size = rounded_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t i = t; i < offset; i += SIMD_WIDTH)
            {
                AVX_Data src_4;
                src_4.data = SIMD_LOAD(src + i);

                SIMD_STORE(dest + i, src_4.data);
            }
        }

#endif

        if (param_size > rounded_size)
        {
#pragma omp parallel for
            for (size_t k = rounded_size; k < param_size; k++)
            {
                dest[k] = src[k];
            }
        }
    }

    static void helper_memcpy_4(float *dest, float *src, size_t param_size)
    {
        size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

        rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

        for (size_t t = 0; t < rounded_size; t += TILE)
        {
            size_t copy_size = TILE;
            if ((t + TILE) > rounded_size)
                copy_size = rounded_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2))
            {
                AVX_Data src_4[4];
                src_4[0].data = SIMD_LOAD(src + i);
                src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
                src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
                src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);

                SIMD_STORE(dest + i, src_4[0].data);
                SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
                SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
                SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
            }
        }
#endif
        if (param_size > rounded_size)
            helper_memcpy_1((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
    }

    static void helper_memcpy_8(float *dest, float *src, size_t param_size)
    {
        size_t rounded_size = 0;

#if defined(__AVX512__) or defined(__AVX256__)

        rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

        for (size_t t = 0; t < rounded_size; t += TILE)
        {
            size_t copy_size = TILE;
            if ((t + TILE) > rounded_size)
                copy_size = rounded_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3))
            {
                AVX_Data src_4[8];
                src_4[0].data = SIMD_LOAD(src + i);
                src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
                src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
                src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);
                src_4[4].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 2));
                src_4[5].data = SIMD_LOAD(src + i + SIMD_WIDTH * 5);
                src_4[6].data = SIMD_LOAD(src + i + SIMD_WIDTH * 6);
                src_4[7].data = SIMD_LOAD(src + i + SIMD_WIDTH * 7);

                SIMD_STORE(dest + i, src_4[0].data);
                SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
                SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
                SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
                SIMD_STORE(dest + i + (SIMD_WIDTH << 2), src_4[4].data);
                SIMD_STORE(dest + i + SIMD_WIDTH * 5, src_4[5].data);
                SIMD_STORE(dest + i + SIMD_WIDTH * 6, src_4[6].data);
                SIMD_STORE(dest + i + SIMD_WIDTH * 7, src_4[7].data);
            }
        }
#endif
        if (param_size > rounded_size)
            helper_memcpy_4((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
    }

    virtual int deepspeed_py_memcpy(torch::Tensor &dest, const torch::Tensor &src) override
    {
        auto dest_c = dest.contiguous();
        auto src_c = src.contiguous();

        float *dest_ptr = (float *)dest_c.data_ptr();
        float *src_ptr = (float *)src_c.data_ptr();

        helper_mempcy_8(dest_ptr, src_ptr, dest_c.size(0));

        return 0;
    }

    struct MyPinTensor : public deepspeed_pin_tensor_t
    {
        ~deepspeed_pin_tensor_t() override
        {
            for (auto iter = _locked_tensors.begin(); iter != _locked_tensors.end(); ++iter)
            {
                munlock(iter->first, iter->second);
            }
            _locked_tensors.clear();
        }
        torch::Tensor alloc(size_t num_elem, at::ScalarType elem_type) override
        {
            const auto num_bytes = num_elem * elementSize(elem_type);
            auto pinned_buffer = ds_page_aligned_alloc(num_bytes, true);
            assert(nullptr != pinned_buffer);

            _locked_tensors[pinned_buffer] = num_bytes;

            auto options = torch::TensorOptions().dtype(elem_type).device(torch::kCPU);

            return at::from_blob(pinned_buffer, static_cast<long int>(num_bytes), options);
        }

        bool free(torch::Tensor &locked_tensor) override
        {
            auto addr = locked_tensor.data_ptr();
            if (_locked_tensors.find(addr) != _locked_tensors.end())
            {
                munlock(addr, _locked_tensors[addr]);
                _locked_tensors.erase(addr);
                return true;
            }

            return false;
        }
    };

    // virtual deepspeed_pin_tensor_t *get_pin_tensor() override
    // {
    //     return new MyPinTensor();
    // }

    struct Myio_op_desc_t : public io_op_desc_t
    {
        // Check Here
        Myio_op_desc_t(const bool read_op,
                       const torch::Tensor &buffer,
                       const int fd,
                       const char *filename,
                       const long long int num_bytes,
                       const bool validate):io_op_desc_t(read_op,
                                                         buffer,
                                                         fd,
                                                         filename,
                                                         num_bytes,
                                                         validate)
        {
            _cpu_buffer = (_buffer.is_cuda() || _buffer.is_xpu()
#if defined(__ENABLE_CANN__)
                           || torch_npu::utils::is_npu(_buffer)
#endif
                               )
                              ? _buffer.to(torch::kCPU).pin_memory()
                              : _buffer;
            _contiguous_buffer = _cpu_buffer.contiguous();
        }

        char *data_ptr() override const { return (char *)_contiguous_buffer.data_ptr(); }

        void fini() override
        {
            if (_read_op && _buffer.is_cuda())
            {
                _buffer.copy_(_cpu_buffer.to(torch::kCUDA));
            }
            if (_read_op && _buffer.is_xpu())
            {
                _buffer.copy_(_cpu_buffer.to(torch::kXPU));
            }
#if defined(__ENABLE_CANN__)
            if (_read_op && torch_npu::utils::is_npu(_buffer))
            {
                auto device = at::Device("npu:0");
                _buffer.copy_(_cpu_buffer.to(device));
            }
#endif
        }

    }

    // virtual io_op_desc_t *
    // create_io_op_desc_t(bool read_op,
    //                     const torch::Tensor &buffer,
    //                     int fd,
    //                     const char *filename,
    //                     long long int num_bytes,
    //                     bool validate) override
    // {
    //     return new Myio_op_desc_t();
    // }

    struct Mydeepspeed_aio_thread_t : public deepspeed_aio_thread_t
    {
        Mydeepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config) : deepspeed_aio_thread_t(tid, aio_config) {}

        void run() override
        {
            while (true)
            {
                std::shared_ptr<io_op_desc_t> next_io_op = nullptr;

                {
                    std::unique_lock<std::mutex> lock(_work_sync._mutex);
                    _work_sync._cond_var.wait(lock,
                                              [this]
                                              { return (!_work_queue.empty() || _time_to_exit); });
                    if (!_work_queue.empty())
                    {
                        next_io_op = _work_queue.front();
                        _work_queue.pop();
                    }
                }

                if (next_io_op)
                {
                    const auto base_offset = next_io_op->_num_bytes * _tid;

                    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
                        next_io_op->_fd, base_offset, next_io_op->_num_bytes, next_io_op->data_ptr()));

                    if (_aio_config._overlap_events)
                    {
                        do_aio_operation_overlap(
                            next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
                    }
                    else
                    {
                        do_aio_operation_sequential(
                            next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
                    }

                    {
                        std::lock_guard<std::mutex> lock(_complete_sync._mutex);
                        _complete_queue.push(next_io_op);
                    }
                    _complete_sync._cond_var.notify_one();
                }

                if (_time_to_exit)
                {
                    break;
                }
            }
        }

    } 

    // static void _start_aio_thread(std::shared_ptr<Mydeepspeed_aio_thread_t> ctxt) { ctxt->run(); }
    // above call is also correct but need to pass only objects of type Mydeep_dpeed_aio_thread_t
    // and if we make below one we can pass any deirved objects of type deepspeed_aio_thread_t
    // i am using below for flexibility
    static void _start_aio_thread(std::shared_ptr<deepspeed_aio_thread_t> ctxt) { ctxt->run(); }


    // virtual deepspeed_aio_thread_t *get_aio_thread(const int tid, deepspeed_aio_config_t &aio_config) override
    // {
    //     return new Mydeepspeed_aio_thread_t();
    // }

    struct Mydeepspeed_aio_handle_t : public deepspeed_aio_handle_t
    {
        // Doubt
        Mydeepspeed_aio_handle_t (const int block_size,
                         const int queue_depth,
                         const bool single_submit,
                         const bool overlap_events,
                         const int num_threads)
        : deepspeed_aio_handle_t(block_size, queue_depth, single_submit, overlap_events, num_threads) {}
        {
            for (auto i = 0; i < num_threads; ++i)
            {
                _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t>(i, _aio_config));
            }

            for (auto &ctxt : _thread_contexts)
            {
                _threads.push_back(std::thread(_start_aio_thread, ctxt));
            }
        }
        // Doubt
        ~deepspeed_aio_handle_t() override
        {
            _stop_threads();
            for (auto &thr : _threads)
            {
                thr.join();
            }
        }

        const int get_block_size() const
        {
            return _aio_ctxt ? _aio_ctxt->_block_size : -1;
        }

        const int get_queue_depth() const
        {
            return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
        }

        const bool get_single_submit() const { return _single_submit; }

        const bool get_overlap_events() const { return _overlap_events; }

        const int get_thread_count() const { return _num_threads; }

        int read(torch::Tensor &buffer, const char *filename, const bool validate) override
        {
            const auto start_time = std::chrono::high_resolution_clock::now();

            assert(_aio_ctxt);

            long long num_file_bytes;
            if (-1 == get_file_size(filename, num_file_bytes))
            {
                const auto error_code = errno;
                report_file_error(filename, " fstat for read", error_code);
                return -1;
            }
            assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);
            for (auto &ctxt : _thread_contexts)
            {
                {
                    std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
                    ctxt->_time_to_exit = true;
                }
                ctxt->_work_sync._cond_var.notify_one();
            }
        }

        int wait() override
        {
            assert(_num_pending_ops > 0);
            auto num_completed_ops = 0;

            while (_num_pending_ops > 0)
            {
                auto completed_op = _wait_for_aio_work();

                completed_op->fini();

                close(completed_op->_fd);

                if (completed_op->_validate)
                {
                    validate_aio_operation(completed_op->_read_op,
                                           completed_op->_filename.c_str(),
                                           completed_op->data_ptr(),
                                           _num_threads * completed_op->_num_bytes);
                }
                --_num_pending_ops;
                ++num_completed_ops;
            }

            return num_completed_ops;
        }

        bool _is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes) override
        {
            const auto op_string = read_op ? "Read" : "Write";
            if (num_bytes % get_thread_count())
            {
                std::cout << "deepspeed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
                          << " not divisible by thread count = " << get_thread_count() << std::endl;
                return false;
            }

            return true;
        }

        int pread(const torch::Tensor &buffer,
                  const char *filename,
                  const bool validate,
                  const bool async) override
        {
            long long num_file_bytes;
            if (-1 == get_file_size(filename, num_file_bytes))
            {
                const auto error_code = errno;
                report_file_error(filename, " fstat for read", error_code);
                return -1;
            }
            const auto buffer_bytes = static_cast<long long int>(buffer.nbytes());
            if (buffer_bytes != num_file_bytes)
            {
                std::cout << filename << ": buffer nbytes != file bytes " << buffer_bytes
                          << " != " << num_file_bytes << std::endl;
            }
            assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);
            assert((num_file_bytes % _num_threads) == 0);

            if (!_is_valid_parallel_aio_op(true, num_file_bytes))
            {
                return -1;
            }

            const auto fd = open_file(filename, true);
            if (fd == -1)
            {
                return -1;
            }

            auto scheduled_op = std::make_shared<io_op_desc_t>(
                true, buffer, fd, filename, (num_file_bytes / _num_threads), validate);

            _schedule_aio_work(scheduled_op);

            if (async)
            {
                return 0;
            }

            return wait();
        }

        int pwrite(const torch::Tensor &buffer,
                   const char *filename,
                   const bool validate,
                   const bool async) override
        {
            const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
            assert((num_write_bytes % _num_threads) == 0);

            if (!_is_valid_parallel_aio_op(false, num_write_bytes))
            {
                return -1;
            }

            const auto fd = open_file(filename, false);
            if (fd == -1)
            {
                return -1;
            }

            auto scheduled_op = std::make_shared<io_op_desc_t>(
                false, buffer, fd, filename, (num_write_bytes / _num_threads), validate);

            _schedule_aio_work(scheduled_op);

            if (async)
            {
                return 0;
            }

            return wait();
        }

        int sync_pread(torch::Tensor &buffer, const char *filename) override
        {
            return pread(buffer, filename, false, false);
        }

        int sync_pwrite(const torch::Tensor &buffer, const char *filename) override
        {
            return pwrite(buffer, filename, false, false);
        }

        int async_pread(torch::Tensor &buffer, const char *filename) override
        {
            return pread(buffer, filename, false, true);
        }

        int async_pwrite(const torch::Tensor &buffer, const char *filename) override
        {
            return pwrite(buffer, filename, false, true);
        }

        at::Tensor new_cpu_locked_tensor(const size_t num_elem,
                                         const torch::Tensor &example_tensor) override
        {
            return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
        }

        bool free_cpu_locked_tensor(torch::Tensor &locked_tensor) override
        {
            return _pinned_tensor_mgr->free(locked_tensor);
        }
    }

    // virtual deepspeed_aio_handle_t *
    // get_aio_handle(const int block_size,
    //                const int queue_depth,
    //                const bool single_submit,
    //                const bool overlap_events,
    //                const int num_threads)
    // {
    //     return new Mydeepspeed_aio_handle_t;
    // }
};


extern "C" PluginInterface* create_plugin(const std::map<std::string, std::string>& args) {
    return new NVME(); 
}






































// #pragma once

// #include <pybind11/pybind11.h>
// #include <deepspeed_aio_common.h> // this is from common folder where async io functions are defined
// #include <torch/extension.h>
// #include <./interface.h>

// #include <stdlib.h>
// #include <cstring>
// #include <fstream>
// #include <iostream>

// #include <condition_variable>
// #include <memory>
// #include <cassert>
// #include <chrono>

// #include <map>
// #include <queue>
// #include <string>
// #include <vector>

// #include <assert.h>
// #include <string.h>
// #include <fcntl.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>
// #include <omp.h>

// #define TILE (1024 * 1024 * 1024)

// #if defined(__AVX512__)
// #define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
// #define SIMD_LOAD(x) _mm512_loadu_ps(x)
// #define SIMD_SET(x) _mm512_set1_ps(x)
// #define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
// #define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
// #define SIMD_SQRT(x) _mm512_sqrt_ps(x)
// #define SIMD_DIV(x, y) _mm512_div_ps(x, y)
// #define SIMD_WIDTH 16
// #else
// #if defined(__AVX256__)
// #define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
// #define SIMD_LOAD(x) _mm256_loadu_ps(x)
// #define SIMD_SET(x) _mm256_set1_ps(x)
// #define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
// #define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
// #define SIMD_SQRT(x) _mm256_sqrt_ps(x)
// #define SIMD_DIV(x, y) _mm256_div_ps(x, y)
// #define SIMD_WIDTH 8
// #endif
// #endif

// #define ROUND_DOWN(size, step) ((size) & ~((step)-1))

// #if defined(__AVX512__) or defined(__AVX256__)
// union AVX_Data {
// #if defined(__AVX512__)
//     __m512 data;
// #else
//     __m256 data;
// #endif
// };
// #endif

// #if defined(__ENABLE_CANN__)
// #include "torch_npu/csrc/framework/utils/OpAdapter.h"
// #include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
// #endif

// #define DEBUG_DS_AIO_READ 0
// #define DEBUG_DS_AIO_WRITE 0

// using namespace std;
// using namespace std::chrono;


// static const std::string c_library_name = "deepspeed_aio";


// class NVME : public PluginInterface {
// public:
//     ~NVME() override = default;

//     // 1. from deepspeed_py_aio.cpp
//     int deepspeed_py_aio_write(const torch::Tensor& buffer,
//                                const char* filename,
//                                const int block_size,
//                                const int queue_depth,
//                                const bool single_submit,
//                                const bool overlap_events,
//                                const bool validate) override {
//         const auto start_time = std::chrono::high_resolution_clock::now();
//         deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);

//         const auto fd = open_file(filename, false);
//         if (fd == -1) { return -1; }

//         auto write_buffer = (char*)buffer.data_ptr();
//         const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
//         std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_write_bytes, write_buffer));
//         std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

//         if (config._overlap_events) {
//             do_aio_operation_overlap(false, aio_ctxt, xfer_ctxt, &config, nullptr);
//         } else {
//             do_aio_operation_sequential(false, aio_ctxt, xfer_ctxt, &config, nullptr);
//         }
//         const std::chrono::duration<double> aio_time =
//             std::chrono::high_resolution_clock::now() - start_time;

//         close(fd);

//         if (validate) { validate_aio_operation(false, filename, write_buffer, num_write_bytes); }

//         const std::chrono::duration<double> fn_time =
//             std::chrono::high_resolution_clock::now() - start_time;
//         std::cout << "Elapsed time(usec): "
//                 << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
//                 << std::endl;
//         return 0;
//     }

//     int deepspeed_py_aio_read(torch::Tensor& buffer,
//                               const char* filename,
//                               const int block_size,
//                               const int queue_depth,
//                               const bool single_submit,
//                               const bool overlap_events,
//                               const bool validate) override {
//         const auto start_time = std::chrono::high_resolution_clock::now();
//         long long num_file_bytes;
//         if (-1 == get_file_size(filename, num_file_bytes)) {
//             const auto error_code = errno;
//             report_file_error(filename, " fstat for read", error_code);
//             return -1;
//         }

//         deepspeed_aio_config_t config(block_size, queue_depth, single_submit, overlap_events, false);
//         const auto fd = open_file(filename, true);
//         if (fd == -1) { return -1; }

//         auto read_buffer = (char*)buffer.data_ptr();
//         assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

//         std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_file_bytes, read_buffer));
//         std::unique_ptr<aio_context> aio_ctxt(new aio_context(config._block_size, config._queue_depth));

//         if (config._overlap_events) {
//             do_aio_operation_overlap(true, aio_ctxt, xfer_ctxt, &config, nullptr);
//         } else {
//             do_aio_operation_sequential(true, aio_ctxt, xfer_ctxt, &config, nullptr);
//         }
//         const std::chrono::duration<double> aio_time =
//             std::chrono::high_resolution_clock::now() - start_time;

//         close(fd);

//         if (validate) { validate_aio_operation(true, filename, read_buffer, num_file_bytes); }

//         const std::chrono::duration<double> fn_time =
//             std::chrono::high_resolution_clock::now() - start_time;
//         std::cout << "Elapsed time(usec): "
//                 << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
//                 << std::endl;
//         return 0;
//     }

//     // 2. from deepspeed_py_copy.cpp
//     static void helper_memcpy_1(float* dest, float* src, size_t param_size){
//         size_t rounded_size = 0;

//         #if defined(__AVX512__) or defined(__AVX256__)

//             rounded_size = ROUND_DOWN(param_size, SIMD_WIDTH);

//             for (size_t t = 0; t < rounded_size; t += TILE) {
//                 size_t copy_size = TILE;
//                 if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
//                 size_t offset = copy_size + t;
//         #pragma omp parallel for
//                 for (size_t i = t; i < offset; i += SIMD_WIDTH) {
//                     AVX_Data src_4;
//                     src_4.data = SIMD_LOAD(src + i);

//                     SIMD_STORE(dest + i, src_4.data);
//                 }
//             }

//         #endif

//             if (param_size > rounded_size) {
//         #pragma omp parallel for
//                 for (size_t k = rounded_size; k < param_size; k++) { dest[k] = src[k]; }
//             }
//     }

//     static void helper_memcpy_4(float* dest, float* src, size_t param_size){
//         size_t rounded_size = 0;

//         #if defined(__AVX512__) or defined(__AVX256__)

//             rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

//             for (size_t t = 0; t < rounded_size; t += TILE) {
//                 size_t copy_size = TILE;
//                 if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
//                 size_t offset = copy_size + t;
//         #pragma omp parallel for
//                 for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
//                     AVX_Data src_4[4];
//                     src_4[0].data = SIMD_LOAD(src + i);
//                     src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
//                     src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
//                     src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);

//                     SIMD_STORE(dest + i, src_4[0].data);
//                     SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
//                     SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
//                     SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
//                 }
//             }
//         #endif
//             if (param_size > rounded_size)
//                 helper_memcpy_1((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
//     }

// static void helper_mempcy_8(float* dest, float* src, size_t param_size){
//     size_t rounded_size = 0;

//     #if defined(__AVX512__) or defined(__AVX256__)

//         rounded_size = ROUND_DOWN(param_size, (SIMD_WIDTH << 2));

//         for (size_t t = 0; t < rounded_size; t += TILE) {
//             size_t copy_size = TILE;
//             if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
//             size_t offset = copy_size + t;
//     #pragma omp parallel for
//             for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
//                 AVX_Data src_4[8];
//                 src_4[0].data = SIMD_LOAD(src + i);
//                 src_4[1].data = SIMD_LOAD(src + i + SIMD_WIDTH);
//                 src_4[2].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 1));
//                 src_4[3].data = SIMD_LOAD(src + i + SIMD_WIDTH * 3);
//                 src_4[4].data = SIMD_LOAD(src + i + (SIMD_WIDTH << 2));
//                 src_4[5].data = SIMD_LOAD(src + i + SIMD_WIDTH * 5);
//                 src_4[6].data = SIMD_LOAD(src + i + SIMD_WIDTH * 6);
//                 src_4[7].data = SIMD_LOAD(src + i + SIMD_WIDTH * 7);

//                 SIMD_STORE(dest + i, src_4[0].data);
//                 SIMD_STORE(dest + i + SIMD_WIDTH, src_4[1].data);
//                 SIMD_STORE(dest + i + (SIMD_WIDTH << 1), src_4[2].data);
//                 SIMD_STORE(dest + i + SIMD_WIDTH * 3, src_4[3].data);
//                 SIMD_STORE(dest + i + (SIMD_WIDTH << 2), src_4[4].data);
//                 SIMD_STORE(dest + i + SIMD_WIDTH * 5, src_4[5].data);
//                 SIMD_STORE(dest + i + SIMD_WIDTH * 6, src_4[6].data);
//                 SIMD_STORE(dest + i + SIMD_WIDTH * 7, src_4[7].data);
//             }
//         }
//     #endif
//         if (param_size > rounded_size)
//             helper_memcpy_4((dest + rounded_size), (src + rounded_size), (param_size - rounded_size));
//     }

//     int deepspeed_py_memcpy(torch::Tensor& dest, const torch::Tensor& src) override {
//         auto dest_c = dest.contiguous();
//         auto src_c = src.contiguous();

//         float* dest_ptr = (float*)dest_c.data_ptr();
//         float* src_ptr = (float*)src_c.data_ptr();

//         helper_mempcy_8(dest_ptr, src_ptr, dest_c.size(0));
//         return 0;
//     }

//     ~deepspeed_pin_tensor_t::deepspeed_pin_tensor_t() override{
//         for (auto iter = _locked_tensors.begin(); iter != _locked_tensors.end(); ++iter) {
//             munlock(iter->first, iter->second);
//         }
//         _locked_tensors.clear();
//     }

//     torch::Tensor deepspeed_pin_tensor_t::alloc(const size_t num_elem, const at::ScalarType& elem_type) override
//     {
//         const auto num_bytes = num_elem * elementSize(elem_type);
//         auto pinned_buffer = ds_page_aligned_alloc(num_bytes, true);
//         assert(nullptr != pinned_buffer);

//         _locked_tensors[pinned_buffer] = num_bytes;

//         auto options = torch::TensorOptions().dtype(elem_type).device(torch::kCPU);

//         return at::from_blob(pinned_buffer, static_cast<long int>(num_bytes), options);
//     }

//     bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor) override
//     {
//         auto addr = locked_tensor.data_ptr();
//         if (_locked_tensors.find(addr) != _locked_tensors.end()) {
//             munlock(addr, _locked_tensors[addr]);
//             _locked_tensors.erase(addr);
//             return true;
//         }

//         return false;
//     }


//     io_op_desc_t::io_op_desc_t(const bool read_op,
//                            const torch::Tensor& buffer,
//                            const int fd,
//                            const char* filename,
//                            const long long int num_bytes,
//                            const bool validate)
//         : _read_op(read_op),
//         _buffer(buffer),
//         _fd(fd),
//         _filename(filename),
//         _num_bytes(num_bytes),
//         _validate(validate)
//         {
//             _cpu_buffer = (_buffer.is_cuda() || _buffer.is_xpu()
//         #if defined(__ENABLE_CANN__)
//                         || torch_npu::utils::is_npu(_buffer)
//         #endif
//                             )
//                             ? _buffer.to(torch::kCPU).pin_memory()
//                             : _buffer;
//             _contiguous_buffer = _cpu_buffer.contiguous();
//         }

//     char* io_op_desc_t::data_ptr() override const { return (char*)_contiguous_buffer.data_ptr(); }

//     void io_op_desc_t::fini() override
//     {
//         if (_read_op && _buffer.is_cuda()) { _buffer.copy_(_cpu_buffer.to(torch::kCUDA)); }
//         if (_read_op && _buffer.is_xpu()) { _buffer.copy_(_cpu_buffer.to(torch::kXPU)); }
//     #if defined(__ENABLE_CANN__)
//         if (_read_op && torch_npu::utils::is_npu(_buffer)) {
//             auto device = at::Device("npu:0");
//             _buffer.copy_(_cpu_buffer.to(device));
//         }
//     #endif
//     }

//     deepspeed_aio_thread_t::deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config)
//         : _tid(tid),
//         _aio_config(aio_config),
//         _aio_ctxt(new aio_context(aio_config._block_size, aio_config._queue_depth)),
//         _time_to_exit(false)
//     {
//     }

//     deepspeed_aio_thread_t::~deepspeed_aio_thread_t() {}

//     void deepspeed_aio_thread_t::run() override
//     {
//         while (true) {
//             std::shared_ptr<struct io_op_desc_t*> next_io_op = nullptr;

//             {
//                 std::unique_lock<std::mutex> lock(_work_sync->_mutex);
//                 _work_sync->_cond_var.wait(lock,
//                                         [this] { return (!_work_queue.empty() || _time_to_exit); });
//                 if (!_work_queue.empty()) {
//                     next_io_op = _work_queue.front();
//                     _work_queue.pop();
//                 }
//             }

//             if (next_io_op) {
//                 const auto base_offset = next_io_op->_num_bytes * _tid;

//                 std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
//                     next_io_op->_fd, base_offset, next_io_op->_num_bytes, next_io_op->data_ptr()));

//                 if (_aio_config._overlap_events) {
//                     do_aio_operation_overlap(
//                         next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
//                 } else {
//                     do_aio_operation_sequential(
//                         next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
//                 }

//                 {
//                     std::lock_guard<std::mutex> lock(_complete_sync._mutex);
//                     _complete_queue.push(next_io_op);
//                 }
//                 _complete_sync->_cond_var.notify_one();
//             }

//             if (_time_to_exit) { break; }
//         }
//     }

//     static void _start_aio_thread(std::shared_ptr<struct deepspeed_aio_thread_t*> ctxt) { ctxt->run(); }

//     deepspeed_aio_handle_t::deepspeed_aio_handle_t(const int block_size,
//                                                 const int queue_depth,
//                                                 const bool single_submit,
//                                                 const bool overlap_events,
//                                                 const int num_threads)
//         : _aio_ctxt(new aio_context(block_size, queue_depth)),
//         _single_submit(single_submit),
//         _overlap_events(overlap_events),
//         _num_threads(num_threads),
//         _aio_config(block_size, queue_depth, single_submit, overlap_events, false),
//         _num_pending_ops(0),
//         _pinned_tensor_mgr(new deepspeed_pin_tensor_t())
//     {
//         for (auto i = 0; i < num_threads; ++i) {
//             _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t*>(i, _aio_config));
//         }

//         for (auto& ctxt : _thread_contexts) {
//             _threads.push_back(std::thread(_start_aio_thread, ctxt));
//         }
//     }

//     deepspeed_aio_handle_t::~deepspeed_aio_handle_t()
//     {
//         _stop_threads();
//         for (auto& thr : _threads) { thr.join(); }
//     }

//     const int deepspeed_aio_handle_t::get_block_size() const
//     {
//         return _aio_ctxt ? _aio_ctxt->_block_size : -1;
//     }

//     const int deepspeed_aio_handle_t::get_queue_depth() const
//     {
//         return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
//     }

//     const bool deepspeed_aio_handle_t::get_single_submit() const { return _single_submit; }

//     const bool deepspeed_aio_handle_t::get_overlap_events() const { return _overlap_events; }

//     const int deepspeed_aio_handle_t::get_thread_count() const { return _num_threads; }

//     int deepspeed_aio_handle_t::read(torch::Tensor& buffer, const char* filename, const bool validate) override
//     {
//         const auto start_time = std::chrono::high_resolution_clock::now();

//         assert(_aio_ctxt);

//         long long num_file_bytes;
//         if (-1 == get_file_size(filename, num_file_bytes)) {
//             const auto error_code = errno;
//             report_file_error(filename, " fstat for read", error_code);
//             return -1;
//         }
//         assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);
//         for (auto& ctxt : _thread_contexts) {
//             {
//                 std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
//                 ctxt->_time_to_exit = true;
//             }
//             ctxt->_work_sync._cond_var.notify_one();
//         }
//     }

//     int deepspeed_aio_handle_t::wait() override
//     {
//         assert(_num_pending_ops > 0);
//         auto num_completed_ops = 0;

//         while (_num_pending_ops > 0) {
//             auto completed_op = _wait_for_aio_work();

//             completed_op->fini();

//             close(completed_op->_fd);

//             if (completed_op->_validate) {
//                 validate_aio_operation(completed_op->_read_op,
//                                     completed_op->_filename.c_str(),
//                                     completed_op->data_ptr(),
//                                     _num_threads * completed_op->_num_bytes);
//             }
//             --_num_pending_ops;
//             ++num_completed_ops;
//         }

//         return num_completed_ops;
//     }

//     bool deepspeed_aio_handle_t::_is_valid_parallel_aio_op(const bool read_op,
//                                                         const long long int num_bytes) override
//     {
//         const auto op_string = read_op ? "Read" : "Write";
//         if (num_bytes % get_thread_count()) {
//             std::cout << "deepspeed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
//                     << " not divisible by thread count = " << get_thread_count() << std::endl;
//             return false;
//         }

//         return true;
//     }

//     int deepspeed_aio_handle_t::pread(const torch::Tensor& buffer,
//                                     const char* filename,
//                                     const bool validate,
//                                     const bool async) override
//     {
//         long long num_file_bytes;
//         if (-1 == get_file_size(filename, num_file_bytes)) {
//             const auto error_code = errno;
//             report_file_error(filename, " fstat for read", error_code);
//             return -1;
//         }
//         const auto buffer_bytes = static_cast<long long int>(buffer.nbytes());
//         if (buffer_bytes != num_file_bytes) {
//             std::cout << filename << ": buffer nbytes != file bytes " << buffer_bytes
//                     << " != " << num_file_bytes << std::endl;
//         }
//         assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);
//         assert((num_file_bytes % _num_threads) == 0);

//         if (!_is_valid_parallel_aio_op(true, num_file_bytes)) { return -1; }

//         const auto fd = open_file(filename, true);
//         if (fd == -1) { return -1; }

//         auto scheduled_op = std::make_shared<io_op_desc_t>(
//             true, buffer, fd, filename, (num_file_bytes / _num_threads), validate);

//         _schedule_aio_work(scheduled_op);

//         if (async) { return 0; }

//         return wait();
//     }

//     int deepspeed_aio_handle_t::pwrite(const torch::Tensor& buffer,
//                                     const char* filename,
//                                     const bool validate,
//                                     const bool async) override
//     {
//         const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
//         assert((num_write_bytes % _num_threads) == 0);

//         if (!_is_valid_parallel_aio_op(false, num_write_bytes)) { return -1; }

//         const auto fd = open_file(filename, false);
//         if (fd == -1) { return -1; }

//         auto scheduled_op = std::make_shared<io_op_desc_t>(
//             false, buffer, fd, filename, (num_write_bytes / _num_threads), validate);

//         _schedule_aio_work(scheduled_op);

//         if (async) { return 0; }

//         return wait();
//     }

//     int deepspeed_aio_handle_t::sync_pread(torch::Tensor& buffer, const char* filename) override
//     {
//         return pread(buffer, filename, false, false);
//     }

//     int deepspeed_aio_handle_t::sync_pwrite(const torch::Tensor& buffer, const char* filename) override
//     {
//         return pwrite(buffer, filename, false, false);
//     }

//     int deepspeed_aio_handle_t::async_pread(torch::Tensor& buffer, const char* filename) override
//     {
//         return pread(buffer, filename, false, true);
//     }

//     int deepspeed_aio_handle_t::async_pwrite(const torch::Tensor& buffer, const char* filename) override
//     {
//         return pwrite(buffer, filename, false, true);
//     }

//     at::Tensor deepspeed_aio_handle_t::new_cpu_locked_tensor(const size_t num_elem,
//                                                             const torch::Tensor& example_tensor) override
//     {
//         return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
//     }

//     bool deepspeed_aio_handle_t::free_cpu_locked_tensor(torch::Tensor& locked_tensor) override
//     {
//         return _pinned_tensor_mgr->free(locked_tensor);
//     }


// };