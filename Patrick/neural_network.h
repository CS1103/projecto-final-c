#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#pragma once
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <vector>
#include <memory>
namespace utec::neural_network {
    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> _layers;
    public:
        NeuralNetwork() = default;

        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            _layers.push_back(std::move(layer));
        }

        Tensor2D<T> forward(const Tensor2D<T>& x) {
            Tensor2D<T> current_output = x;
            for (const auto& layer : _layers) {
                current_output = layer->forward(current_output);
            }
            return current_output;
        }
        Tensor2D<T> predict(const Tensor2D<T>& x) {
            return forward(x);
        }
        void backward(const Tensor2D<T>& grad) {
            Tensor2D<T> current_grad = grad;
            for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
                current_grad = (*it)->backward(current_grad);
            }
        }
        template<template<typename> class LossType>
        void train(const Tensor2D<T>& X, const Tensor2D<T>& Y, size_t epochs, size_t batch_size, T lr) {
            SGD<T> optimizer(lr);
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                Tensor2D<T> Y_pred = this->forward(X);
                LossType<T> criterion(Y_pred, Y);
                Tensor2D<T> grad = criterion.loss_gradient();
                this->backward(grad);
                for(auto& layer : _layers) {
                    layer->update_params(optimizer);
                }
            }
        }
    };
}
#endif
