//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#include "tensor.h"
#include "nn_interfaces.h"
#include <functional>
#include <random>
#include <array>

namespace utec::neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T,2> weights;
    utec::algebra::Tensor<T,2> bias;
    utec::algebra::Tensor<T,2> last_input;
    utec::algebra::Tensor<T,2> grad_weights;
    utec::algebra::Tensor<T,2> grad_bias;

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_features, size_t out_features, InitWFun init_w_fun, InitBFun init_b_fun)
        : weights(std::array<size_t, 2>{in_features, out_features}),
          bias(std::array<size_t, 2>{1, out_features}),
          last_input(std::array<size_t, 2>{1, 1}),
          grad_weights(std::array<size_t, 2>{in_features, out_features}),
          grad_bias(std::array<size_t, 2>{1, out_features}) {
        init_w_fun(weights);
        init_b_fun(bias);
    }

    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        last_input = x;
        utec::algebra::Tensor<T,2> output(std::array<size_t, 2>{x.shape()[0], weights.shape()[1]});

        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < weights.shape()[1]; ++j) {
                output(i,j) = bias(0,j);
                for (size_t k = 0; k < weights.shape()[0]; ++k) {
                    output(i,j) += x(i,k) * weights(k,j);
                }
            }
        }
        return output;
    }

    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& dZ) override {
        const auto batch_size = last_input.shape()[0];
        const auto in_features = weights.shape()[0];
        const auto out_features = weights.shape()[1];

        // Calcular gradiente de los pesos
        grad_weights.fill(0);
        for (size_t i = 0; i < in_features; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                for (size_t k = 0; k < batch_size; ++k) {
                    grad_weights(i,j) += last_input(k,i) * dZ(k,j);
                }
                grad_weights(i,j) /= batch_size;
            }
        }

        // Calcular gradiente del bias
        grad_bias.fill(0);
        for (size_t j = 0; j < out_features; ++j) {
            for (size_t k = 0; k < batch_size; ++k) {
                grad_bias(0,j) += dZ(k,j);
            }
            grad_bias(0,j) /= batch_size;
        }

        // Calcular gradiente de la entrada
        utec::algebra::Tensor<T,2> dX(std::array<size_t, 2>{batch_size, in_features});
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                T sum = 0;
                for (size_t k = 0; k < out_features; ++k) {
                    sum += dZ(i,k) * weights(j,k);
                }
                dX(i,j) = sum;
            }
        }

        return dX;
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights, grad_weights);
        optimizer.update(bias, grad_bias);
    }
};

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
