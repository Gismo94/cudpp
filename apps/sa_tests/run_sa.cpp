// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
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

// includes, project
#include "cudpp.h"
#include "read_text.h"
#include <cuda_wrapper_interface.h>

#include <string>
//#include <filesystem>

//namespace fs = std::experimental::filesystem;

// Returns size of the file. Creates a new unsigned char Pointer and fills it
// with the text contained in the given file.
size_t get_file_content(std::string path_to_file, unsigned char* out_text) {
    auto context = read_text_context(path_to_file);
    size_t text_size = context.size;
    out_text = (unsigned char*) realloc(out_text, text_size);
    context.read_text(out_text, text_size);
    return text_size;
}



int compute_sa(unsigned char* out_text, unsigned int* sa, size_t text_size) {
// Allocate cuda memory
    size_t free_bytes = check_cuda_memory_free();
    size_t mem_needed = text_size * sizeof(unsigned char);
    if(mem_needed > 0.015 * free_bytes) {
        std::cerr << "Not enough gpu memory to compute sa. Free memory: "
        << ((free_bytes/1024)/1024) << "MB. Needed memory: "
        << (((mem_needed/ 0.015)/1024)/1024) << "MB." << std::endl;
        return 42;
    } else {
        auto cuda_text = (unsigned char*) allocate_cuda_buffer(text_size*sizeof(unsigned char));
        auto cuda_sa = (unsigned int*) allocate_cuda_buffer(text_size*sizeof(unsigned int));
        // Prepare cudpp for computation
        cuda_copy_text_to_device(out_text, cuda_text, text_size);

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
        result = cudppSuffixArray(plan, cuda_text, cuda_sa, text_size);
        if(result != CUDPP_SUCCESS) {
            std::cerr << "Error while computing suffix array." << std::endl;
        }
        // Start computation

        // Copy sa
        cuda_copy_device_to_host(cuda_sa, sa, text_size);

        // Clean up cudpp

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

        // Free cuda memory
        free_cuda_buffer(cuda_text);
        free_cuda_buffer(cuda_sa);
    }
    return 0;
}

void test_all_files(vector<std::string> file_paths) {
    size_t text_size;
    unsigned char* out_text;
    unsigned int* sa;
    for(std::string &file_path : file_paths) {
        std::cout << "Computing sa for file " << file_path << "." << std::endl;
        // Fill with content
        text_size = get_file_content(file_path, out_text);
        // Got the text, now test!
        sa = (unsigned int*) realloc(sa, text_size);
        auto error = compute_sa(out_text, sa, text_size);
        if(error != 0) {
                std::cerr << "Error while computing sa for file " << file_path
                << "."<< std::endl;
        }
    }
    free(out_text);
    free(sa);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv)
{
    std::string parent_path = "../../../apps/data/";
    auto file_paths = std::vector(5);
    file_paths[0] = parent_path + "pc_sources.200MB";

    test_all_files(path_to_files);
    return 0;
}
