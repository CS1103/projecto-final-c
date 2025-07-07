//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "tensor.h"
#include "nn_interfaces.h"
#include <memory>

namespace utec::neural_network {

template<typename T, size_t DIMS>
using Tensor = algebra::Tensor<T, DIMS>;

    template<typename T>
    class SGD final : public IOptimizer<T> {
    private:
        T learning_rate;
    public:
        explicit SGD(T lr = 0.01) : learning_rate(lr) {}

        void update(algebra::Tensor<T,2>& params,
                    const algebra::Tensor<T,2>& grads) override {
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    params(i,j) -= learning_rate * grads(i,j);
                }
            }
        }
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
    private:
        T learning_rate, beta1, beta2, epsilon;
        long t = 0;
        std::unique_ptr<algebra::Tensor<T,2>> m, v;
        bool initialized = false;
    public:
        explicit Adam(T lr = 0.001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
            : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}


void update(algebra::Tensor<T,2>& params,
            const algebra::Tensor<T,2>& grads) override {
            std::cout << std::fixed << std::setprecision(6);
            if (!initialized || (m && m->shape() != params.shape())) {
                m = std::make_unique<algebra::Tensor<T,2>>(params.shape());
                v = std::make_unique<algebra::Tensor<T,2>>(params.shape());
                m->fill(0);
                v->fill(0);
                initialized = true;
                t = 0;
            }
            t++;
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    (*m)(i,j) = beta1 * (*m)(i,j) + (1 - beta1) * grads(i,j);
                    (*v)(i,j) = beta2 * (*v)(i,j) + (1 - beta2) * grads(i,j) * grads(i,j);

                    T m_hat = (*m)(i,j) / (1 - std::pow(beta1, t));
                    T v_hat = (*v)(i,j) / (1 - std::pow(beta2, t));
                    v_hat = std::max(v_hat, T(1e-12));

                    params(i,j) -= (learning_rate * m_hat / (std::sqrt(v_hat) + epsilon));
                }
            }
        }

        void step() override {}
    };
}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H