#pragma once

#include <torch/torch.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined at compile time"
#endif

inline torch::Tensor load_tensor(std::string const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        assert(false);
    }
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    auto ivalue = torch::pickle_load(bytes);
    return ivalue.toTensor().to(torch::kFloat64).contiguous();
}

inline std::string data_path(std::string const& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}
