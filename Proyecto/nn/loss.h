#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "interfaces.h"
#include <cmath>
#include <limits>


namespace utec::neural_network {

template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    Tensor<T,2> y_pred_;
    Tensor<T,2> y_true_;
    T cached_loss_ = 0;
public:
    MSELoss(const Tensor<T,2>& y_pred, const Tensor<T,2>& y_true)
      : y_pred_(y_pred), y_true_(y_true) {
        // calcular el loss al construir
        cached_loss_ = T{0};
        const size_t n = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < n; ++i) {
            T diff = y_pred_.begin()[i] - y_true_.begin()[i];
            cached_loss_ += diff * diff;
        }
        cached_loss_ /= n;
    }

    T loss() const override {
        return cached_loss_;
    }

    Tensor<T,2> loss_gradient() const override {
        Tensor<T,2> grad = y_pred_ - y_true_;
        T scale = static_cast<T>(2.0) / static_cast<T>(grad.shape()[0] * grad.shape()[1]);
        for (auto& val : grad)
            val *= scale;
        return grad;
    }
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    Tensor<T,2> y_pred_;
    Tensor<T,2> y_true_;
    T cached_loss_ = 0;
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();  // para evitar log(0)
public:
    BCELoss(const Tensor<T,2>& y_pred, const Tensor<T,2>& y_true)
      : y_pred_(y_pred), y_true_(y_true) {
        const size_t n = y_pred_.shape()[0] * y_pred_.shape()[1];
        cached_loss_ = 0;
        for (size_t i = 0; i < n; ++i) {
            T p = std::clamp(y_pred_.begin()[i], epsilon, 1 - epsilon); // evitar log(0)
            T y = y_true_.begin()[i];
            cached_loss_ += - (y * std::log(p) + (1 - y) * std::log(1 - p));
        }
        cached_loss_ /= n;
    }

    T loss() const override {
        return cached_loss_;
    }

    Tensor<T,2> loss_gradient() const override {
        Tensor<T,2> grad(y_pred_.shape());
        const size_t n = y_pred_.shape()[0] * y_pred_.shape()[1];
        for (size_t i = 0; i < n; ++i) {
            T p = std::clamp(y_pred_.cbegin()[i], epsilon, 1 - epsilon);
            T y = y_true_.cbegin()[i];
            grad.begin()[i] = (p - y) / (p * (1 - p)) / n;
        }
        return grad;
    }
};

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H