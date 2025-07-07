#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "interfaces.h"
#include "tensor.h"
#include <array>
#include <numeric>

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
    public:
        Dense(size_t in_f, size_t out_f,
              auto init_w_fun, auto init_b_fun)
            : weights_(std::array<size_t, 2>{in_f, out_f}),
              biases_(std::array<size_t, 2>{1, out_f}),
              grad_weights_(std::array<size_t, 2>{in_f, out_f}),
              grad_biases_(std::array<size_t, 2>{1, out_f}) {
            init_w_fun(weights_);
            init_b_fun(biases_);
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            input_ = x;
            // Z = X·W + b
            auto Z = matrix_product(x, weights_);
            for (size_t i = 0; i < Z.shape()[0]; ++i)
                for (size_t j = 0; j < Z.shape()[1]; ++j)
                    Z(i, j) += biases_(0, j);
            return Z;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            // Gradientes: dW = Xᵗ·dZ, db = suma filas de dZ, dX = dZ·Wᵗ
            grad_weights_ = matrix_product(transpose_2d(input_), dZ);

            // grad_biases_ = suma de filas de dZ
            for (size_t j = 0; j < dZ.shape()[1]; ++j) {
                T sum = 0;
                for (size_t i = 0; i < dZ.shape()[0]; ++i)
                    sum += dZ(i, j);
                grad_biases_(0, j) = sum;
            }

            return matrix_product(dZ, transpose_2d(weights_));
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights_, grad_weights_);
            optimizer.update(biases_, grad_biases_);
        }

    private:
        Tensor<T, 2> weights_, biases_;
        Tensor<T, 2> grad_weights_, grad_biases_;
        Tensor<T, 2> input_;
    };

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H