//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "tensor.h"
#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T, size_t DIMS>
using Tensor = algebra::Tensor<T, DIMS>;

template<typename T>
class ReLU final : public ILayer<T> {
private:
    Tensor<T,2> last_input;
public:
    ReLU() : last_input(std::array<size_t, 2>{1, 1}) {}

    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        last_input = z;
        Tensor<T,2> result(z.shape());

        for (size_t i = 0; i < z.shape()[0]; ++i) {
            for (size_t j = 0; j < z.shape()[1]; ++j) {
                result(i, j) = (z(i, j) < 0) ? 0 : z(i, j);
            }
        }
        return result;
    }

    Tensor<T,2> backward(const Tensor<T,2>& g) override {
        Tensor<T,2> grad(g.shape());
        for (size_t i = 0; i < g.shape()[0]; ++i) {
            for (size_t j = 0; j < g.shape()[1]; ++j) {
                grad(i, j) = (last_input(i, j) > 0) ? g(i, j) : 0;
            }
        }
        return grad;
    }

    void update_params(IOptimizer<T>& optimizer) override {}
};

template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    Tensor<T,2> last_output;
public:
    Sigmoid() : last_output(std::array<size_t, 2>{1, 1}) {}

    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        last_output = Tensor<T,2>(z.shape());
        for (size_t i = 0; i < z.shape()[0]; ++i) {
            for (size_t j = 0; j < z.shape()[1]; ++j) {
                last_output(i, j) = 1.0 / (1.0 + std::exp(-z(i, j)));
            }
        }
        return last_output;
    }

    Tensor<T,2> backward(const Tensor<T,2>& g) override {
        Tensor<T,2> grad(g.shape());
        for (size_t i = 0; i < g.shape()[0]; ++i) {
            for (size_t j = 0; j < g.shape()[1]; ++j) {
                grad(i, j) = g(i, j) * last_output(i, j) * (1 - last_output(i, j));
            }
        }
        return grad;
    }

    void update_params(IOptimizer<T>& optimizer) override {}
};

} // namespace utec::neural_network

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H