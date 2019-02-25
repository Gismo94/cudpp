#pragma once

#include <cstddef>
#include <cstdint>

void* allocate_cuda_buffer(size_t size);
void* allocate_managed_cuda_buffer(size_t size);
template<typename T>
T* allocate_managed_cuda_buffer_of(size_t size) {
    return (T*) allocate_managed_cuda_buffer(size * sizeof(T));
}

void free_cuda_buffer(void* ptr);

bool check_cuda_memory_32(size_t bytes_needed);
bool check_cuda_memory_64(size_t bytes_needed);

size_t check_cuda_memory_free();

void gpu_init();

void cuda_copy_device_to_device(unsigned int* d_in, unsigned int* d_out,
            size_t num_items);
/*
//extern "C"
void cuda_copy_device_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_device_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);
*/

void cuda_copy_text_to_device(unsigned char* d_in, unsigned char* d_out,
            size_t num_items);

void cuda_copy_host_to_device(unsigned int* d_in, unsigned int* d_out,
            size_t num_items);
/*
//extern "C"
void cuda_copy_host_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_host_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);
*/
void cuda_copy_device_to_host(unsigned int* d_in, unsigned int* d_out,
            size_t num_items);
/*
//extern "C"
void cuda_copy_device_to_host(uint32_t* d_in, uint32_t* d_out,
            size_t num_items);
//extern "C"
void cuda_copy_device_to_host(uint64_t* d_in, uint64_t* d_out,
            size_t num_items);
*/
