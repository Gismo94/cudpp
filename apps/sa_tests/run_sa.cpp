// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ---------------------------------------unsigned----------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cstdio>
#include <cuda_runtime_api.h>

// includes, project
#include "cudpp.h"
#include "cuda_util.h"
#include "read_text.h"
#include "sa_gold.cpp"
#include "comparearrays.h"
#include <cuda_wrapper_interface.h>

#include <string>
//#include <stdio.h>  /* defines FILENAME_MAX */
// #define WINDOWS  /* uncomment this line to use it for windows.*/
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include<iostream>

std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

//#include <filesystem>

//namespace fs = std::experimental::filesystem;


// Returns size of the file. Creates a new unsigned char Pointer and fills it
// with the text contained in the given file.
unsigned char* get_file_content(std::string path_to_file, size_t &text_size) {
    auto context = read_text_context(path_to_file);
    text_size = context.size;
    auto out_text = context.read_text(text_size);
    return out_text;
}

/*
int get_file_content(std::string path_to_file, unsigned char* out_text,
            size_t& text_size) {
    FILE* file = std::fopen(path_to_file.c_str(), "r");
    if(!file) {
        return 1;
    }
    std::fseek(file, 0, SEEK_END);
    text_size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    out_text = (unsigned char*) realloc(out_text, text_size);
    std::fread(out_text, sizeof(unsigned char), text_size, file);
    std::fclose(file);
    return 0;
}*/


int compute_sa(const unsigned char* out_text, unsigned int* sa, size_t text_size) {
// Allocate cuda memory
    size_t free_bytes = check_cuda_memory_free();
    size_t mem_needed = text_size * sizeof(unsigned char);
    std::cout << "text_size: " << text_size << "| sizeof(unsigned char): "
    << sizeof(char) << std::endl;
    if(mem_needed > 0.015 * free_bytes) {
        std::cerr << "mem_needed: " << mem_needed << "| 0.14*free_bytes: "
        << 0.015 * free_bytes << std::endl;
        std::cerr << "Not enough gpu memory to compute sa. Free memory: "
        << ((free_bytes/1024)/1024) << "MB. Needed memory: "
        << (((mem_needed/ 0.015)/1024)/1024) << "MB." << std::endl;
        return 42;
    } else {
        std::cout << "Allocating memory on CUDA Device." << std::endl;
        /*
        unsigned char* cuda_text = (unsigned char*) allocate_cuda_buffer(text_size*sizeof(unsigned char));
        unsigned int* cuda_sa = (unsigned int*) allocate_cuda_buffer(text_size*sizeof(unsigned int));
        */
        unsigned char* cuda_text=NULL;
        CUDA_SAFE_CALL(cudaMalloc((void**)&cuda_text, text_size*sizeof(unsigned char)));
        unsigned int* cuda_sa=NULL;
        CUDA_SAFE_CALL(cudaMalloc((void**)&cuda_sa, text_size*sizeof(unsigned int)));
        // Prepare cudpp for computation

        std::cout << "Copying text to device." << std::endl;
        CUDA_SAFE_CALL(cudaMemcpy(cuda_text, out_text,
                    sizeof(unsigned char) * text_size, cudaMemcpyHostToDevice));

        //cuda_copy_text_to_device(out_text, cuda_text, text_size);
        std::cout << "Preparing cudpp to run." << std::endl;
        // Initialize CUDPP
        CUDPPConfiguration config;
        config.algorithm = CUDPP_SA;
        config.options = 0;
        config.datatype = CUDPP_UCHAR;

        CUDPPHandle plan;
        CUDPPResult result = CUDPP_SUCCESS;
        CUDPPHandle theCudpp;
        result = cudppCreate(&theCudpp);


        if (result != CUDPP_SUCCESS)
        {
            std::cerr << "Error initializing CUDPP Library" << std::endl;
        }
        result = cudppPlan(theCudpp, &plan, config, text_size, 1, 0);

        if(result != CUDPP_SUCCESS)
        {
            std::cerr << "Error in plan creation" << std::endl;
        }

        std::cout << "Running SACA." << std::endl;
        result = cudppSuffixArray(plan, cuda_text, cuda_sa, text_size);

        if(result != CUDPP_SUCCESS) {
            std::cerr << "Error while computing suffix array." << std::endl;
        }
        // Start computation
        std::cout << "Copy sa result to host." << std::endl;
        // Copy sa
        CUDA_SAFE_CALL(cudaMemcpy(sa, cuda_sa, sizeof(unsigned int) * text_size,
                cudaMemcpyDeviceToHost));
        //cuda_copy_device_to_host(cuda_sa, sa, text_size);

        // Clean up cudpp
        std::cout << "Clean up cudpp." << std::endl;
        result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS)
        {
            std::cerr << "Error destroying CUDPPPlan for Suffix Array"
            << std::endl;
        }

        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS)
        {
            std::cerr << "Error shutting down CUDPP Library." << std::endl;
        }
        std::cout << "Free cuda Memory." << std::endl;

        // Free cuda memory
        free_cuda_buffer(cuda_text);
        free_cuda_buffer(cuda_sa);
    }
    return 0;
}

void test_all_files(std::vector<std::string> file_paths) {
    size_t text_size;
    unsigned char* out_text=NULL;
    unsigned int* sa = NULL;
    unsigned int* reference = NULL;
    int error;
    bool result;
    for(std::string &file_path : file_paths) {
        std::cout << "Computing sa for file " << file_path << "." << std::endl;
        // Fill with content
        out_text = get_file_content(file_path, text_size);
        std::cout << out_text[0] << out_text[1] << out_text[3] << std::endl;
        // Got the text, now test!
        std::cout << "(re)allocating pointer for sa." << std::endl;
        sa = (unsigned int*) malloc(text_size*sizeof(unsigned int));
        std::cout << "Computing sa." << std::endl;
        error = compute_sa(out_text, sa, text_size);
        if(error != 0) {
                std::cerr << "Error while computing sa for file " << file_path
                << "."<< std::endl;
        }
        /*
        //std::cout << sa[0] << "," << sa[1] << "," << sa[2] << std::endl;
        // Compute reference sa
        std::cout << "Computing reference SA." << std::endl;
        reference = (unsigned int*) malloc(sizeof(unsigned int) * text_size);
        memset(reference, 0, sizeof(unsigned int) * text_size);

        computeSaGold(out_text, reference, text_size);

        result = compareArrays<unsigned int> (reference, sa, text_size);
        if(!result) {
            std::cout << "SA not computed correctly." << std::endl;
        }*/
        std::cout << "Freeing host memory." << std::endl;
        free(out_text);
        //free(reference);
        free(sa);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv)
{
    gpu_init();
    std::cout << "Current path: " << GetCurrentWorkingDir() << std::endl;
    std::string parent_path = "../../apps/sa_tests/data/";
    auto file_paths = std::vector<std::string>(3);
    file_paths[0] = parent_path + "pc_sources.2MB";
    file_paths[1] = parent_path + "pc_sources.1MB";
    file_paths[2] = parent_path + "pc_sources.50MB";
    test_all_files(file_paths);
    return 0;
}
