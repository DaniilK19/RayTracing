// GPU memory for the renderer's host code: CUDA's errors as exceptions, and an array in the GPU's memory that owns it.
// Shared by renderer.cu, quality.cu and GpuTests' files. Host code only: the kernels never see these.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace gpu
{
// Turns a CUDA error code into an exception. Most CUDA functions return a cudaError_t instead of throwing. The runtime
// also keeps the last error for cudaGetLastError, which reports it again later: a caught failure (an allocation too
// large for the GPU) would then make the next launch look as if it had failed. So an error reported here is cleared.
// (An error in a running kernel can't be cleared: it ends CUDA in the process, and every later call reports it.)
inline void check(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

// An array in the GPU's memory that owns it, like a std::vector: cudaMalloc in the constructor and cudaFree in the
// destructor (RAII), so the memory is freed however render() ends, an exception included.
template <typename T>
class device_array
{
public:
    explicit device_array(std::size_t size) : size_(size)
    {
        if (size_ > 0)
        {
            check(cudaMalloc(&data_, size_ * sizeof(T)), "allocating GPU memory");
        }
    }

    explicit device_array(const std::vector<T>& values) : device_array(values.size())
    {
        if (size_ > 0)
        {
            check(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "copying to the GPU");
        }
    }

    ~device_array() { cudaFree(data_); } // nothing to do for nullptr

    // A copy would free the same memory twice; a move hands it over (a move-constructed array leaves the other one
    // empty, an assignment swaps the two)
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;
    device_array(device_array&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr))
    {
    }
    device_array& operator=(device_array&& other) noexcept
    {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
        return *this; // the other array frees what this one held
    }

    T* data() const { return data_; }
    std::size_t size() const { return size_; }

    void fill_with_zeros() { fill_with_zeros(size_); }

    // The first `count` elements only
    void fill_with_zeros(std::size_t count)
    {
        if (count > 0)
        {
            check(cudaMemset(data_, 0, count * sizeof(T)), "clearing GPU memory");
        }
    }

    std::vector<T> to_host() const { return to_host(size_); }

    std::vector<T> to_host(std::size_t count) const
    {
        std::vector<T> values(count);
        if (count > 0)
        {
            check(cudaMemcpy(values.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost), "copying from the GPU");
        }
        return values;
    }

private:
    std::size_t size_;
    T* data_ = nullptr;
};
}
