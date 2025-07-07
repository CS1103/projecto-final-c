//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#include <nn_interfaces.h>
#include <tensor.h>

namespace utec::neural_network {

    template<typename T>
    class MSELoss : public ILoss<T, 2>{
        algebra::Tensor<T, 2> last_pred, last_target;
        public:

        MSELoss() = default;

        MSELoss(const Tensor<T, 2>& y_pred, const Tensor<T, 2>& y_target) : last_pred(y_pred), last_target(y_target) {}

        T loss() const override {
            T sum = 0;

            for (size_t i = 0; i < last_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < last_pred.shape()[1]; ++j) {
                    T diff = last_pred(i, j) - last_target(i, j);
                    sum += diff * diff;
                }
            }

            return sum / (last_pred.shape()[0] * last_pred.shape()[1]);
        }

        Tensor<T,2> loss_gradient() const override {
            Tensor<T,2> grad(last_pred.shape());

            for (size_t i = 0; i < last_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < last_pred.shape()[1]; ++j) {
                    grad(i,j) = 2 * (last_pred(i,j) - last_target(i,j)) / (last_pred.shape()[0] * last_pred.shape()[1]);
                }
            }
            return grad;
        }

        T forward(const Tensor<T,2>& pred, const Tensor<T,2>& target) {
            last_pred = pred;
            last_target = target;
            return loss();
        }

        Tensor<T,2> backward() {
            return loss_gradient();
        }
    };

    template<typename T>
class BCELoss final : public ILoss<T, 2> {
        Tensor<T,2> y_pred;
        Tensor<T,2> y_true;
        static constexpr T epsilon = 1e-12;
    public:
        BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_target)
            : y_pred(y_prediction), y_true(y_target) {
            // Clip predictions to avoid log(0)
            for (auto& val : y_pred) {
                val = std::max(epsilon, std::min(1 - epsilon, val));
            }
        }

        T loss() const override {
            T sum = 0;
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    sum += y_true(i,j) * std::log(y_pred(i,j)) +
                          (1 - y_true(i,j)) * std::log(1 - y_pred(i,j));
                }
            }
            return -sum / (y_pred.shape()[0] * y_pred.shape()[1]);
        }

        Tensor<T,2> loss_gradient() const override {
            Tensor<T,2> grad(y_pred.shape());
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    grad(i,j) = (y_pred(i,j) - y_true(i,j)) /
                               (y_pred(i,j) * (1 - y_pred(i,j)) * (y_pred.shape()[0] * y_pred.shape()[1]));
                }
            }
            return grad;
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
