#include <ATen/Functions.h>
#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  auto variance = x.pow(2).mean(-1, true);
  auto x_hat = x * at::rsqrt(variance + eps);
  return x_hat * weight;
}
