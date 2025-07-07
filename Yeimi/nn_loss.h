#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "tensor.h"
#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

template<typename T, size_t DIMS>
using Tensor = algebra::Tensor<T, DIMS>;

template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    Tensor<T, 2> y_pred, y_true;
    const T epsilon = 1e-12;

public:
    BCELoss(const Tensor<T, 2>& y_prediction, const Tensor<T, 2>& y_true_value) : y_pred(y_prediction), y_true(y_true_value) {}

    T loss() const override {
        T sum = 0;
        const size_t rows = y_pred.shape()[0];
        const size_t cols = y_pred.shape()[1];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T p = std::max(epsilon, std::min(1 - epsilon, y_pred(i, j)));
                sum += y_true(i, j) * std::log(p) + (1 - y_true(i, j)) * std::log(1 - p);
            }
        }
        return -sum / (rows * cols);
    }

    Tensor<T, 2> loss_gradient() const override {
        Tensor<T, 2> grad(y_pred.shape());
        const size_t rows = y_pred.shape()[0];
        const size_t cols = y_pred.shape()[1];
        const size_t total_elements = rows * cols;

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T p = std::max(epsilon, std::min(1 - epsilon, y_pred(i, j)));
                grad(i, j) = (p - y_true(i, j)) / (p * (1 - p) * total_elements);
            }
        }
        return grad;
    }
};

template<typename T>
class MSELoss : public ILoss<T, 2> {
private:
    const Tensor<T, 2>& predictions;
    const Tensor<T, 2>& targets;

public:
    MSELoss(const Tensor<T, 2>& pred, const Tensor<T, 2>& target) : predictions(pred), targets(target) {}

    T loss() const override {
        T total_loss = 0;
        const size_t rows = predictions.shape()[0];
        const size_t cols = predictions.shape()[1];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T diff = predictions(i, j) - targets(i, j);
                total_loss += diff * diff;
            }
        }
        return total_loss / (rows * cols);
    }

    Tensor<T, 2> loss_gradient() const override {
        Tensor<T, 2> grad(predictions.shape());
        const size_t rows = predictions.shape()[0];
        const size_t cols = predictions.shape()[1];
        const size_t total_elements = rows * cols;

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                grad(i, j) = 2 * (predictions(i, j) - targets(i, j)) / total_elements;
            }
        }
        return grad;
    }
};

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H