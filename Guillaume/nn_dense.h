//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <algorithm>

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
    public:
        template<typename InitW, typename InitB>
        Dense(size_t in_f, size_t out_f, InitW init_w, InitB init_b)
            : weights_(in_f, out_f), bias_(1, out_f) {
            init_w(weights_);
            init_b(bias_);
        }

        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
            x_ = x;
            auto z = utec::algebra::matrix_product(x, weights_);
            return z + bias_;  // broadcasting
        }

        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& dZ) override {
            auto x_t = utec::algebra::transpose_2d(x_);
            dW_ = utec::algebra::matrix_product(x_t, dZ);

            auto dims = dZ.shape();
            size_t batch = dims[0], out_f = dims[1];
            dB_ = utec::algebra::Tensor<T,2>(1, out_f);
            for (size_t j = 0; j < out_f; ++j) {
                T sum = T();
                for (size_t i = 0; i < batch; ++i)
                    sum += dZ(i, j);
                dB_(0, j) = sum;
            }

            auto w_t = utec::algebra::transpose_2d(weights_);
            return utec::algebra::matrix_product(dZ, w_t);
        }

        void update_params(IOptimizer<T>& opt) override {
            opt.update(weights_, dW_);
            opt.step();
            opt.update(bias_, dB_);
            opt.step();
        }

    private:
        utec::algebra::Tensor<T,2> weights_, bias_, x_, dW_, dB_;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
