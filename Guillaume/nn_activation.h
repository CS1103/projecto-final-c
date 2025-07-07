//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
    public:
        utec::algebra::Tensor<T,2> forward(
            const utec::algebra::Tensor<T,2>& z) override {
            z_ = z;
            auto result = z;
            for (auto& v : result)
                v = (v > T(0) ? v : T(0));
            return result;
        }

        utec::algebra::Tensor<T,2> backward(
            const utec::algebra::Tensor<T,2>& g) override {
            auto grad = g;
            auto it_z = z_.cbegin();
            for (auto it = grad.begin(); it != grad.end(); ++it, ++it_z)
                if (*it_z <= T(0)) *it = T(0);
            return grad;
        }

        void update_params(IOptimizer<T>&) override {}

    private:
        utec::algebra::Tensor<T,2> z_;
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
    public:
        utec::algebra::Tensor<T,2> forward(
            const utec::algebra::Tensor<T,2>& z) override {
            auto result = z;
            for (auto& v : result)
                v = T(1) / (T(1) + std::exp(-v));
            s_ = result;
            return result;
        }

        utec::algebra::Tensor<T,2> backward(
            const utec::algebra::Tensor<T,2>& g) override {
            auto grad = g;
            auto it_s = s_.cbegin();
            for (auto it = grad.begin(); it != grad.end(); ++it, ++it_s) {
                T sig = *it_s;
                *it = *it * (sig * (T(1) - sig));
            }
            return grad;
        }

        void update_params(IOptimizer<T>&) override {}

    private:
        utec::algebra::Tensor<T,2> s_;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
