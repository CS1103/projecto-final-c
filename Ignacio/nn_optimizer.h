//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T>
class SGD final : public IOptimizer<T> {
    T learning_rate;
public:
    explicit SGD(T lr = 0.1) : learning_rate(lr) {}

    void update(Tensor<T,2>& params, const Tensor<T,2>& grads) override {
        for (size_t i = 0; i < params.shape()[0]; ++i) {
            for (size_t j = 0; j < params.shape()[1]; ++j) {
                params(i,j) -= learning_rate * grads(i,j);
            }
        }
    }

    void update(Tensor<T,1>& params, const Tensor<T,1>& grads) override {
        for (size_t i = 0; i < params.shape()[0]; ++i) {
            params[i] -= learning_rate * grads[i];
        }
    }
};

template<typename T>
class Adam final : public IOptimizer<T> {
    T learning_rate;
    T beta1, beta2;
    T epsilon;
    size_t t = 0;
    bool initialized = false;

    Tensor<T,2> m_W, v_W;
    Tensor<T,1> m_b, v_b;

public:
    explicit Adam(T lr = 0.001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    void init(const Tensor<T,2>& W, const Tensor<T,1>& b) {
        m_W = Tensor<T,2>(W.shape());
        v_W = Tensor<T,2>(W.shape());
        m_b = Tensor<T,1>(b.shape());
        v_b = Tensor<T,1>(b.shape());
        m_W.fill(0);
        v_W.fill(0);
        m_b.fill(0);
        v_b.fill(0);

        initialized = true;
    }

    void update(Tensor<T,2>& W, const Tensor<T,2>& dW) override {
        if (!initialized)
            init(W, Tensor<T,1>(std::array<size_t, 1>{W.shape()[1]}));

        t++;
        for (size_t i = 0; i < W.shape()[0]; ++i) {
            for (size_t j = 0; j < W.shape()[1]; ++j) {
                m_W(i,j) = beta1 * m_W(i,j) + (1 - beta1) * dW(i,j);
                v_W(i,j) = beta2 * v_W(i,j) + (1 - beta2) * dW(i,j) * dW(i,j);

                T m_hat = m_W(i,j) / (1 - std::pow(beta1, t));
                T v_hat = v_W(i,j) / (1 - std::pow(beta2, t));

                W(i,j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }

    void update(Tensor<T,1>& b, const Tensor<T,1>& db) override {
        if (!initialized) init(Tensor<T,2>(std::array<size_t, 2>{1, b.shape()[0]}), b);

        for (size_t i = 0; i < b.shape()[0]; ++i) {
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i];
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * db[i] * db[i];

            T m_hat = m_b[i] / (1 - std::pow(beta1, t));
            T v_hat = v_b[i] / (1 - std::pow(beta2, t));

            b[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void step() override { t++; }
};

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
