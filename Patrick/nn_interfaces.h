#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#pragma once

#include "tensor.h"
namespace utec::neural_network {
  using utec::algebra::Tensor;
  template<typename T>
  using Tensor2D = utec::algebra::Tensor<T, 2>;
  template<typename T>
  struct IOptimizer;
  template<typename T>
  struct ILayer {
    virtual ~ILayer() = default;
    virtual Tensor2D<T> forward(const Tensor2D<T>& x) = 0;
    virtual Tensor2D<T> backward(const Tensor2D<T>& grad) = 0;
    virtual void update_params(IOptimizer<T>& optimizer) {}
  };
  template<typename T>
  struct ILoss {
    virtual ~ILoss() = default;
    virtual T loss() const = 0;
    virtual Tensor2D<T> loss_gradient() const = 0;
  };

  template<typename T>
  struct IOptimizer {
    virtual ~IOptimizer() = default;
    virtual void update(Tensor<T,2>& params, const Tensor<T,2>& grads) = 0;
  };
}
#endif
