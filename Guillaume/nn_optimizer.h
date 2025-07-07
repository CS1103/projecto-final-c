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
    public:
        explicit SGD(T lr = T(0.01)) : lr_(lr) {}
        void update(utec::algebra::Tensor<T,2>& params,
                    const utec::algebra::Tensor<T,2>& grads) override {
            auto scaled = grads * lr_;
            params = params - scaled;
        }
    private:
        T lr_;
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
    public:
        explicit Adam(T lr = T(0.001),
                      T b1 = T(0.9),
                      T b2 = T(0.999),
                      T eps = T(1e-8))
            : lr_(lr), beta1_(b1), beta2_(b2), eps_(eps), t_(1) {}

        void update(utec::algebra::Tensor<T,2>& params,
                    const utec::algebra::Tensor<T,2>& grads) override {
            if (m_.shape() != params.shape()) {
                m_ = utec::algebra::Tensor<T,2>(params.shape());
                v_ = utec::algebra::Tensor<T,2>(params.shape());
                m_.fill(T(0));
                v_.fill(T(0));
            }
            auto it_g = grads.begin(), it_m = m_.begin(), it_v = v_.begin();
            for (; it_g != grads.end(); ++it_g, ++it_m, ++it_v) {
                *it_m = beta1_ * (*it_m) + (T(1) - beta1_) * (*it_g);
                *it_v = beta2_ * (*it_v) + (T(1) - beta2_) * ((*it_g) * (*it_g));
            }
            T bias1_corr = T(1) - std::pow(beta1_, T(t_));
            T bias2_corr = T(1) - std::pow(beta2_, T(t_));
            auto m_hat = m_ * (T(1) / bias1_corr);
            auto v_hat = v_ * (T(1) / bias2_corr);
            auto it_p = params.begin(), it_mh = m_hat.begin(), it_vh = v_hat.begin();
            for (; it_p != params.end(); ++it_p, ++it_mh, ++it_vh) {
                *it_p -= lr_ * (*it_mh / (std::sqrt(*it_vh) + eps_));
            }
        }

        void step() override { ++t_; }

    private:
        T lr_, beta1_, beta2_, eps_;
        size_t t_;
        utec::algebra::Tensor<T,2> m_, v_;
    };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
