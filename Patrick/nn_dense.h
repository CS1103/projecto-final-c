#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#pragma once

#include "nn_interfaces.h"
#include <random>
#include <functional>

namespace utec::neural_network {

    template <typename T>
    class Dense : public ILayer<T> {
    private:
        Tensor2D<T> _weights;
        Tensor2D<T> _biases;
        Tensor2D<T> _dW;
        Tensor2D<T> _db;
        Tensor2D<T> _last_input;
    public:
        Dense(size_t in, size_t out) : _weights(in, out), _biases(1, out) {
            std::random_device rd;
            std::mt19937 gen(rd());
            T stddev = std::sqrt(2.0 / in);
            std::normal_distribution<T> d(0, stddev);
            for (auto& val : _weights) val = d(gen);
            _biases.fill(0);
        }

        template<typename InitW, typename InitB>
        Dense(size_t in, size_t out, InitW init_w, InitB init_b)
            : _weights(in, out), _biases(1, out) {
            init_w(_weights);
            init_b(_biases);
        }

        Tensor2D<T> forward(const Tensor2D<T>& x) override {
            _last_input = x;
            return x.matmul(_weights) + _biases;
        }

        Tensor2D<T> backward(const Tensor2D<T>& grad) override {
            auto input_T = _last_input.transpose_2d();
            _dW = input_T.matmul(grad);

            _db = Tensor2D<T>(1, grad.shape()[1]);
            _db.fill(0);
            for(size_t j = 0; j < grad.shape()[1]; ++j) {
                for(size_t i = 0; i < grad.shape()[0]; ++i) {
                    _db(0, j) += grad(i, j);
                }
            }
            auto weights_T = _weights.transpose_2d();
            return grad.matmul(weights_T);
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(_weights, _dW);
            optimizer.update(_biases, _db);
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
