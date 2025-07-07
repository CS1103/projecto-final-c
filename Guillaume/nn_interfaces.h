//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "tensor.h"
#include <cstddef>

namespace utec::neural_network {
  template<typename T, size_t N>
  using Tensor = utec::algebra::Tensor<T, N>;
  template<typename T>
  class IOptimizer {
  public:
    virtual void update(utec::algebra::Tensor<T,2>& params,
                        const utec::algebra::Tensor<T,2>& grads) = 0;
    virtual void step() {}
    virtual ~IOptimizer() = default;
  };

  template<typename T>
  class ILayer {
  public:
    virtual utec::algebra::Tensor<T,2> forward(
        const utec::algebra::Tensor<T,2>& x) = 0;
    virtual utec::algebra::Tensor<T,2> backward(
        const utec::algebra::Tensor<T,2>& grad) = 0;
    virtual void update_params(IOptimizer<T>& optimizer) {}
    virtual ~ILayer() = default;
  };

  template<typename T, size_t D>
  class ILoss {
  public:
    virtual T loss() const = 0;
    virtual utec::algebra::Tensor<T,D> loss_gradient() const = 0;
    virtual ~ILoss() = default;
  };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
