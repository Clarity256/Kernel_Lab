#include <ATen/Functions.h>
#include <torch/extension.h>

torch::Tensor softmax_forward(torch::Tensor x, int64_t dim) {
  return at::softmax(x, dim, x.scalar_type());
}
