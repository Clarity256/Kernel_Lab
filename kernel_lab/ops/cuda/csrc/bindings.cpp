#include <torch/extension.h>

torch::Tensor softmax_forward(torch::Tensor x, int64_t dim);
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_forward", &softmax_forward, "Softmax forward");
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward");
}

