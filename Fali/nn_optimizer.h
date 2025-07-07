#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#pragma once

#include "nn_interfaces.h"
#include <unordered_map>
#include <cmath>

namespace utec::neural_network {

template <typename T>
class SGD final : public IOptimizer<T> {
public:
    explicit SGD(T lr) : lr_(lr) {}
    void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
        auto it_p = param.begin();
        auto it_g = grad.cbegin();
        for (; it_p != param.cend() && it_g != grad.cend(); ++it_p, ++it_g) {
            *it_p -= lr_ * (*it_g);
        }
    }

private:
    T lr_;
};

template<typename T>
class Adam : public IOptimizer<T> {
private:
    T lr_;
    T beta1_;
    T beta2_;
    T epsilon_;
    size_t t_;

    std::unordered_map<void*, Tensor<T,2>> m_map_;
    std::unordered_map<void*, Tensor<T,2>> v_map_;

public:
    explicit Adam(T learning_rate,
                  T beta1 = static_cast<T>(0.9),
                  T beta2 = static_cast<T>(0.999),
                  T epsilon = static_cast<T>(1e-8))
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void update(Tensor<T,2>& param, const Tensor<T,2>& grad) override {
        ++t_;
        void* key = static_cast<void*>(&param);

        // Si no hay m y v aún, inicialízalos en 0
        if (m_map_.find(key) == m_map_.end()) {
            m_map_[key] = Tensor<T,2>(param.shape());
            m_map_[key].fill(0);

            v_map_[key] = Tensor<T,2>(param.shape());
            v_map_[key].fill(0);
        }

        auto& m = m_map_[key];
        auto& v = v_map_[key];

        auto it_p = param.begin();
        auto it_g = grad.cbegin();
        auto it_m = m.begin();
        auto it_v = v.begin();

        for (; it_p != param.end(); ++it_p, ++it_g, ++it_m, ++it_v) {
            // m = beta1 * m + (1 - beta1) * grad
            *it_m = beta1_ * (*it_m) + (1 - beta1_) * (*it_g);

            // v = beta2 * v + (1 - beta2) * grad^2
            *it_v = beta2_ * (*it_v) + (1 - beta2_) * (*it_g) * (*it_g);

            // m_hat and v_hat
            T m_hat = (*it_m) / (1 - std::pow(beta1_, static_cast<T>(t_)));
            T v_hat = (*it_v) / (1 - std::pow(beta2_, static_cast<T>(t_)));

            // Update param
            *it_p -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
};





}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
