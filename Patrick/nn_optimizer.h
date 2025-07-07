#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#pragma once
#include "nn_interfaces.h"
namespace utec::neural_network {
    template <typename T>
    class SGD : public IOptimizer<T> {
    private:
        T _lr;
    public:
        explicit SGD(T learning_rate = 0.01) : _lr(learning_rate) {}
        void update(utec::algebra::Tensor<T,2>& params, const utec::algebra::Tensor<T,2>& grads) override {
            params = params - (grads * _lr);
        }
    };
    template <typename T>
    class Adam : public IOptimizer<T> {
    private:
        T _lr, _beta1, _beta2, _eps;
        utec::algebra::Tensor<T, 2> _m, _v;
        int _t;
    public:
        Adam(T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T eps = 1e-8)
            : _lr(lr), _beta1(beta1), _beta2(beta2), _eps(eps), _t(0) {}
        void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) override {
            if (_t == 0) {
                _m = Tensor2D<T>(params.shape());
                _v = Tensor2D<T>(params.shape());
                _m.fill(0);
                _v.fill(0);
            }
            _t++;
            _m = _m * _beta1 + grads * (1 - _beta1);
            _v = _v * _beta2 + (grads*grads) * (1 - _beta2);
            auto m_hat = _m * (1.0 / (1.0 - std::pow(_beta1, _t)));
            auto v_hat = _v * (1.0 / (1.0 - std::pow(_beta2, _t)));
            for(size_t i=0; i < params.size(); ++i) {
                params[i] -= _lr * m_hat[i] / (std::sqrt(v_hat[i]) + _eps);
            }
        }
    };
}
#endif
