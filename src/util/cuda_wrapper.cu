#include<iostream>
#include<cstdint>
#include<cstddef>

#include <cuda.h>
#include "cuda_util.cuh"

#include "cuda_wrapper_interface.h"


void* allocate_cuda_buffer(size_t size) {
    void* ret = nullptr;
    cuda_check(cudaMalloc(&ret, size), "cudaMalloc");
    return ret;
}

void* allocate_managed_cuda_buffer(size_t size) {
    void* ret = nullptr;
    cuda_check(cudaMallocManaged(&ret, size), "cudaMallocManaged");
    return ret;
}

void free_cuda_buffer(void* ptr) {
    cuda_check(cudaFree(ptr), "cudaFree");
}

template <typename size_type>
bool check_cuda_memory(size_t bytes_needed) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    bool sufficient_memory;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    sufficient_memory = free_bytes > bytes_needed ? true : false;
    return sufficient_memory;
}

size_t check_cuda_memory_free() {
    //size_t bytes_needed = sizeof(size_type)*num_arrays*num_items + 0.001*sizeof(size_type)*num_items*2;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

bool check_cuda_memory_32(size_t bytes_needed) {
    return check_cuda_memory<uint32_t>(bytes_needed);
}

bool check_cuda_memory_64(size_t bytes_needed) {
    return check_cuda_memory<uint64_t>(bytes_needed);
}


void cuda_copy_device_to_device(unsigned int* d_in, unsigned int* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));
}
/*
void cuda_copy_device_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyDeviceToDevice));
}

void cuda_copy_device_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));
}
*/
void cuda_copy_text_to_device(unsigned char* d_in, unsigned char* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(unsigned char),
            cudaMemcpyHostToDevice));
}

void cuda_copy_host_to_device(unsigned int* d_in, unsigned int* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(unsigned int),
            cudaMemcpyHostToDevice));
}
/*
void cuda_copy_host_to_device(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyHostToDevice));
}

void cuda_copy_host_to_device(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyHostToDevice));
}
*/

void cuda_copy_device_to_host(unsigned int* d_in, unsigned int* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(unsigned int),
            cudaMemcpyDeviceToHost));
}
/*
void cuda_copy_device_to_host(uint32_t* d_in, uint32_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
}

void cuda_copy_device_to_host(uint64_t* d_in, uint64_t* d_out,
            size_t num_items) {
    cuda_check(cudaMemcpy(d_out, d_in, num_items*sizeof(uint64_t),
            cudaMemcpyDeviceToHost));
}
*/
