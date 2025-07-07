#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#pragma once
#include "nn_interfaces.h"
namespace utec::neural_network {
template<typename T>
class MSELoss : public ILoss<T> {
private:
    Tensor2D<T> _last_pred;
    Tensor2D<T> _last_target;
public:
    MSELoss() = default;
    MSELoss(const Tensor2D<T>& pred, const Tensor2D<T>& target)
        : _last_pred(pred), _last_target(target) {}
    T forward(const Tensor2D<T>& pred, const Tensor2D<T>& target) {
        _last_pred = pred;
        _last_target = target;
        return loss();
    }
    Tensor2D<T> backward() {
        return loss_gradient();
    }
    T loss() const override {
        assert(_last_pred.shape() == _last_target.shape());
        T sum_sq_err = 0;
        for (size_t i = 0; i < _last_pred.size(); ++i) {
            T err = _last_pred[i] - _last_target[i];
            sum_sq_err += err * err;
        }
        return sum_sq_err / _last_pred.size();
    }
    Tensor2D<T> loss_gradient() const override {
        Tensor2D<T> grad(_last_pred.shape());
        T n = static_cast<T>(_last_pred.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] = 2.0 * (_last_pred[i] - _last_target[i]) / n;
        }
        return grad;
    }
};

// BCELoss (Required by tests)
template<typename T>
class BCELoss : public ILoss<T> {
private:
    Tensor2D<T> _last_pred;
    Tensor2D<T> _last_target;
    T epsilon = 1e-12; // To avoid log(0)
public:
    BCELoss() = default;
    BCELoss(const Tensor2D<T>& pred, const Tensor2D<T>& target)
        : _last_pred(pred), _last_target(target) {}

    T loss() const override {
        T total_loss = 0;
        for(size_t i = 0; i < _last_pred.size(); ++i) {
            T p = std::max(epsilon, std::min(1.0 - epsilon, _last_pred[i]));
            T y = _last_target[i];
            total_loss += - (y * std::log(p) + (1-y) * std::log(1-p));
        }
        return total_loss / _last_pred.size();
    }
    Tensor2D<T> loss_gradient() const override {
        Tensor2D<T> grad(_last_pred.shape());
        for(size_t i=0; i < grad.size(); ++i) {
            T p = std::max(epsilon, std::min(1.0 - epsilon, _last_pred[i]));
            T y = _last_target[i];
            grad[i] = - (y / p - (1-y) / (1-p));
        }
        return grad * (1.0 / _last_pred.size());
    }
};

}
#endif
