#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available:   " << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
    return 0;
}
