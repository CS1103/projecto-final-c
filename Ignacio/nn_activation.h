//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#include "nn_interfaces.h"

namespace utec::neural_network {

    template<typename T>
    class ReLU : public ILayer<T> {

        algebra::Tensor<T, 2> mask;

        public:
        algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2>& x) override {
            mask = algebra::Tensor<T, 2>(x.shape());
            algebra::Tensor<T, 2> result(x.shape());

            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    mask(i, j) = x(i, j) > 0 ? 1 : 0;
                    result(i, j) = x(i, j) > 0 ? x(i, j) : T{0};
                }
            }

            return result;
        }
        algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2>& grad) override {
            algebra::Tensor<T, 2> grad_input(grad.shape());

            for (size_t i = 0; i < grad.shape()[0]; i++) {
                for (size_t j = 0; j < grad.shape()[1]; ++j) {
                    grad_input(i, j) = grad(i, j) * mask(i, j);
                }
            }

            return grad_input;
        }

        void update_params(IOptimizer<T>& ) override {}
    };

    template <typename T>
    class Sigmoid final : public ILayer<T> {
        algebra::Tensor<T, 2> output;
        public:
        algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2>& x) override {
            output = Tensor<T, 2>(x.shape());
            for (size_t i = 0; i < x.shape()[0]; ++i) {
                for (size_t j = 0; j < x.shape()[1]; ++j) {
                    output(i, j) = 1 / (1 + std::exp(-x(i, j)));
                }
            }
            return output;
        }
        algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2>& grad) override {

            Tensor<T,2> grad_input(grad.shape());
            for (size_t i = 0; i < grad.shape()[0]; ++i) {
                for (size_t j = 0; j < grad.shape()[1]; ++j) {
                    grad_input(i,j) = grad(i,j) * output(i,j) * (1 - output(i,j));
                }
            }
            return grad_input;
        }

        void update_params(IOptimizer<T>& ) override {}
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
