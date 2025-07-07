//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class MSELoss final : public ILoss<T,2> {
    public:
        MSELoss(const utec::algebra::Tensor<T,2>& y_pred,
                const utec::algebra::Tensor<T,2>& y_true)
            : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            auto diff = y_pred_ - y_true_;
            T sum = T();
            for (auto v : diff) sum += v * v;
            auto dims = diff.shape();
            T n = static_cast<T>(dims[0] * dims[1]);
            return sum / n;
        }

        utec::algebra::Tensor<T,2> loss_gradient() const override {
            auto diff = y_pred_ - y_true_;
            auto dims = diff.shape();
            T n = static_cast<T>(dims[0] * dims[1]);
            return diff * (T(2) / n);
        }

    private:
        utec::algebra::Tensor<T,2> y_pred_, y_true_;
    };

    template<typename T>
    class BCELoss final : public ILoss<T,2> {
    public:
        BCELoss(const utec::algebra::Tensor<T,2>& y_pred,
                const utec::algebra::Tensor<T,2>& y_true)
            : y_pred_(y_pred), y_true_(y_true) {}

        T loss() const override {
            auto dims = y_pred_.shape();
            size_t total = dims[0] * dims[1];
            T sum = T();
            for (size_t i = 0; i < total; ++i) {
                T p = y_pred_.cbegin()[i];
                T y = y_true_.cbegin()[i];
                sum += -(y * std::log(p) + (T(1) - y) * std::log(T(1) - p));
            }
            return sum / static_cast<T>(total);
        }

        utec::algebra::Tensor<T,2> loss_gradient() const override {
            auto dims = y_pred_.shape();
            size_t total = dims[0] * dims[1];
            utec::algebra::Tensor<T,2> grad(dims);
            T inv_n = T(1) / static_cast<T>(total);
            for (size_t i = 0; i < total; ++i) {
                T p = y_pred_.cbegin()[i];
                T y = y_true_.cbegin()[i];
                grad.begin()[i] = inv_n * (-y / p + (T(1) - y) / (T(1) - p));
            }
            return grad;
        }

    private:
        utec::algebra::Tensor<T,2> y_pred_, y_true_;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
