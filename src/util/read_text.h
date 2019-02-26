/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <string>
#include <fstream>
#include <iostream>


    /**
     * This struct contains the size of the text and provides a function to read the text file into a string.
     */
struct read_text_context {
    mutable std::ifstream file;
    size_t size = 0;

    read_text_context() = default;
    read_text_context(read_text_context&& other) = default;
    read_text_context& operator=(read_text_context&& other) = default;

    read_text_context(std::string filepath) {
        //file.open(filepath, std::ios::in|std::ios::binary);
        file.open(filepath, std::ios::in);
        if(!file) {
            std::cerr << "Could not open file." << std::endl;
        }
        if((file.rdstate() & std::ifstream::failbit) != 0)
        {
            std::cerr << "Failbit set." << std::endl;
        }
        if((file.rdstate() & std::ifstream::eofbit) != 0) {
            std::cerr << "File pointing to eof." << std::endl;
        }
        if((file.rdstate() & std::ifstream::badbit) != 0) {
            std::cerr << "IO Error." << std::endl;
        }
        file.seekg(0, file.end);
        //file.seekg(0, std::ios::end); // set the pointer to the end
        size = file.tellg() ; // get the length of the file
        if((file.rdstate() & std::ifstream::failbit) != 0)
        {
            std::cerr << "Failbit set." << std::endl;
        }
        if((file.rdstate() & std::ifstream::eofbit) != 0) {
            std::cerr << "File pointing to eof." << std::endl;
        }
        if((file.rdstate() & std::ifstream::badbit) != 0) {
            std::cerr << "IO Error." << std::endl;
        }
    }

    /**
     * \brief Reads content of a txt file into a string.
     * This function reads the content of a text file at the given path bytewise.
     */
    inline unsigned char* read_text(size_t text_size) const {
        //DCHECK_LE(text_size.size(), this->size);
        file.seekg(0, file.beg); // set the pointer to the beginning
        if((file.rdstate() & std::ifstream::failbit) != 0)
        {
            std::cerr << "Failbit set." << std::endl;
        }
        if((file.rdstate() & std::ifstream::eofbit) != 0) {
            std::cerr << "File pointing to eof." << std::endl;
        }
        if((file.rdstate() & std::ifstream::badbit) != 0) {
            std::cerr << "IO Error." << std::endl;
        }
        char* text = (char*) malloc(text_size);
        std::cout << "Allocated text memory." << std::endl;
        file.read(text, text_size);
        std::cout << "file read." << std::endl;
        if(!file) {
            std::cerr << "Could not read file. Read only " << file.gcount()
            << " characters." << std::endl;
        }
        std::cout << text[0] << text[1] << std::endl;
        if((file.rdstate() & std::ifstream::failbit) != 0)
        {
            std::cerr << "Failbit set." << std::endl;
        }
        if((file.rdstate() & std::ifstream::eofbit) != 0) {
            std::cerr << "File pointing to eof." << std::endl;
        }
        if((file.rdstate() & std::ifstream::badbit) != 0) {
            std::cerr << "IO Error." << std::endl;
        }
        file.close();
        std::cout << "file closed." << std::endl;
        return reinterpret_cast<unsigned char*>(text);
    }
    };


/******************************************************************************/
