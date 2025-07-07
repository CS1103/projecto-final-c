#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#pragma once
#include "nn_interfaces.h"
namespace utec::neural_network {
    template <typename T>
    class ReLU : public ILayer<T> {
    private:
        Tensor2D<T> _mask;
    public:
        ReLU() = default;
        Tensor2D<T> forward(const Tensor2D<T>& x) override {
            _mask = Tensor2D<T>(x.shape());
            Tensor2D<T> output(x.shape());
            for (size_t i = 0; i < x.size(); ++i) {
                if (x[i] > 0) {
                    output[i] = x[i];
                    _mask[i] = 1;
                } else {
                    output[i] = 0;
                    _mask[i] = 0;
                }
            }
            return output;
        }
        Tensor2D<T> backward(const Tensor2D<T>& grad) override {
            return grad * _mask;
        }
    };
    template <typename T>
    class Sigmoid : public ILayer<T> {
    private:
        Tensor2D<T> _last_output;
    public:
        Sigmoid() = default;
        Tensor2D<T> forward(const Tensor2D<T>& x) override {
            _last_output = Tensor2D<T>(x.shape());
            for(size_t i = 0; i < x.size(); ++i) {
                _last_output[i] = 1 / (1 + std::exp(-x[i]));
            }
            return _last_output;
        }
        Tensor2D<T> backward(const Tensor2D<T>& grad) override {
            Tensor2D<T> sigmoid_grad(_last_output.shape());
            for(size_t i = 0; i < _last_output.size(); ++i) {
                sigmoid_grad[i] = _last_output[i] * (1 - _last_output[i]);
            }
            return grad * sigmoid_grad;
        }
    };
}
#endif
