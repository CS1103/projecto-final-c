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
    algebra::Tensor<T, 2> y_pred;
    algebra::Tensor<T, 2> y_true;
    const T epsilon = 1e-12; // Para evitar log(0)

public:
    BCELoss(const algebra::Tensor<T, 2>& y_prediction,
            const algebra::Tensor<T, 2>& y_true_value)
        : y_pred(y_prediction), y_true(y_true_value) {
        if (y_pred.shape() != y_true.shape()) {
            throw std::invalid_argument("Shapes of predicted and true values must match");
        }
    }

    T loss() const override {
        T sum = 0;
        const size_t batch_size = y_pred.shape()[0];
        const size_t total_elements = y_pred.size();

        for (size_t i = 0; i < total_elements; ++i) {
            // Clip valores para evitar log(0)
            T p = std::max(epsilon, std::min(1 - epsilon, y_pred.data[i]));
            sum += y_true.data[i] * std::log(p) + (1 - y_true.data[i]) * std::log(1 - p);
        }
        return -sum / total_elements; // Dividir por número total de elementos
    }

    algebra::Tensor<T, 2> loss_gradient() const override {
        algebra::Tensor<T, 2> grad(y_pred.shape());
        const size_t total_elements = y_pred.size();

        for (size_t i = 0; i < total_elements; ++i) {
            // Clip valores para evitar división por 0
            T p = std::max(epsilon, std::min(1 - epsilon, y_pred.data[i]));

            // Gradiente: (p - y) / (p * (1 - p)) normalizado por total de elementos
            grad.data[i] = (p - y_true.data[i]) / (p * (1 - p) * total_elements);
        }
        return grad;
    }
};

template<typename T>
class MSELoss : public ILoss<T, 2> {
private:
    const algebra::Tensor<T,2>& predictions;
    const algebra::Tensor<T,2>& targets;

public:
    MSELoss(const algebra::Tensor<T,2>& pred, const algebra::Tensor<T,2>& target)
        : predictions(pred), targets(target) {}

    T loss() const override {
        T total_loss = 0;
        const size_t total_elements = predictions.size();

        for (size_t i = 0; i < total_elements; ++i) {
            T diff = predictions.data[i] - targets.data[i];
            total_loss += diff * diff;
        }

        return total_loss / total_elements; // Dividir por número total de elementos
    }

    algebra::Tensor<T,2> loss_gradient() const override {
        algebra::Tensor<T,2> grad(predictions.shape());
        const size_t total_elements = predictions.size();

        for (size_t i = 0; i < total_elements; ++i) {
            // Gradiente: 2 * (pred - target) / total_elements
            grad.data[i] = 2 * (predictions.data[i] - targets.data[i]) / total_elements;
        }

        return grad;
    }
};

} // namespace neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H